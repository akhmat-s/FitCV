"""Integration tests for the end-to-end tailoring pipeline.

`generate_tailored_cv` is exercised through the conftest ``mock_llm`` seam: a router
returns the per-section tool-call JSON the model would have produced (keyed by tool
name), so plan → generate → validate → regen → assemble → score → flags runs without
any network access. These cover convergence, the honest-gap did-not-converge path, and
factual-integrity regeneration on an invented company.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import openai

from cv_generator import (
    COVERAGE_TARGET_PCT,
    REGEN_CAP,
    FlagKind,
    TailoredResult,
    generate_tailored_cv,
)
from helprers.llm_model import ProviderResponseError
from schemas import (
    CandidateLevel,
    CVFacts,
    ExtractResult,
    FactsExperience,
    FactsLanguage,
    FactsPersonalInfo,
    JdAnalysis,
    TargetSection,
)

_KEYWORDS = ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"]


def _facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        experiences=[
            FactsExperience(
                role="Analyst",
                company="Acme",
                start_date="2020",
                end_date="2022",
                # evidence the JD keywords so the truth-preserving skills filter keeps them
                bullets=[
                    "Built Python and FastAPI services with Pydantic on PostgreSQL and Docker"
                ],
            )
        ],
    )


def _jd(keywords: list[str]) -> JdAnalysis:
    return JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=list(keywords),
        requirements_must=["3+ years of distributed systems experience"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )


def _responses(experience_company: str = "Acme") -> dict:
    return {
        "generate_summary": {"text": "line a\nline b\nline c", "relevant_skills": ["Python"]},
        "generate_skills": {"categories": [{"category": "Languages", "keywords": list(_KEYWORDS)}]},
        "generate_experience": {
            "experiences": [
                {
                    "role": "Senior Analyst",
                    "company": experience_company,
                    "company_description": "A large enterprise software company today",
                    "start_date": "2020",
                    "end_date": "2022",
                    "bullets": [
                        {"action_verb": "Developed", "description": "built Python services"}
                    ],
                }
            ]
        },
    }


def _router(responses: dict, counter: dict | None = None):
    def call(_system: str, _user: str, schema: dict) -> dict:
        name = schema["name"]
        if counter is not None:
            counter[name] = counter.get(name, 0) + 1
        return responses[name]

    return call


async def test_pipeline_converges_above_target(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python", "FastAPI"]), flags=[])
    mock_llm.call_tool.side_effect = _router(_responses())

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, TailoredResult)
    assert result.ats_score.after_pct >= COVERAGE_TARGET_PCT
    # A fully-valid, converged extract must produce a CLEAN flag set — no
    # did-not-converge AND no phantom capped/one-page flags. This is the guard against
    # a phantom languages cap and a mislabeled cap section.
    assert result.flags == []


async def test_pipeline_did_not_converge_surfaces_flag_and_missing(mock_llm: MagicMock) -> None:
    extract = ExtractResult(
        facts=_facts(),
        jd=_jd(["Python", "Kubernetes", "Terraform", "Go"]),  # mostly absent → < target
        flags=[],
    )
    mock_llm.call_tool.side_effect = _router(_responses())

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, TailoredResult)  # CV still rendered, never an error
    assert result.ats_score.after_pct < COVERAGE_TARGET_PCT
    assert any(f.kind is FlagKind.DID_NOT_CONVERGE for f in result.flags)
    assert result.ats_score.missing  # honest gaps surfaced, not omitted


def _facts_many_roles() -> CVFacts:
    role = lambda n: FactsExperience(  # noqa: E731
        role=f"Engineer {n}",
        company=f"Co{n}",
        start_date="2020",
        end_date="2022",
        bullets=["Built Python and FastAPI services with Pydantic on PostgreSQL and Docker"],
    )
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        experiences=[role(1), role(2), role(3), role(4)],  # > MAX_ROLES_NO_WARNING
    )


async def test_pipeline_omission_panel_single_coverage_and_single_missing(
    mock_llm: MagicMock,
) -> None:
    requirement_phrase = "3+ years of distributed systems experience"
    extract = ExtractResult(
        facts=_facts_many_roles(),
        jd=_jd(["Python", "Kubernetes", "Terraform", "Go"]),  # mostly absent → < target
        flags=[],
    )
    responses = _responses()
    responses["generate_experience"] = {
        "experiences": [
            {
                "role": f"Engineer {n}",
                "company": f"Co{n}",
                "company_description": "A large enterprise software company today",
                "start_date": "2020",
                "end_date": "2022",
                "bullets": [{"action_verb": "Developed", "description": "built Python services"}],
            }
            for n in (1, 2, 3, 4)  # > MAX_ROLES_NO_WARNING → one-page-pressure warning
        ]
    }
    mock_llm.call_tool.side_effect = _router(responses)

    result = await generate_tailored_cv(extract, llm=mock_llm)
    messages = [f.message for f in result.flags]

    coverage = [f for f in result.flags if f.kind is FlagKind.DID_NOT_CONVERGE]
    missing_flags = [f for f in result.flags if f.kind is FlagKind.UNMET_COVERAGE]
    pressure = [f for f in result.flags if f.kind is FlagKind.ONE_PAGE_PRESSURE]

    # exactly one coverage entry, and it carries NO embedded keyword list
    assert len(coverage) == 1
    for gap in result.ats_score.missing:
        assert gap not in coverage[0].message
    # the before→after numbers are on the coverage line
    assert f"{result.ats_score.before_pct}" in coverage[0].message

    # exactly one missing entry; each missing keyword appears in exactly one flag total
    assert len(missing_flags) == 1
    assert "no cv evidence" in missing_flags[0].message.lower()
    for gap in result.ats_score.missing:
        assert sum(gap in m for m in messages) == 1

    # no per-requirement prose leaks into any flag
    assert not any(requirement_phrase in m for m in messages)

    # the >3-roles one-page-pressure warning is still its own separate entry
    assert len(pressure) == 1


async def test_pipeline_invented_company_triggers_regeneration(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    counter: dict = {}
    mock_llm.call_tool.side_effect = _router(_responses(experience_company="Ghost Inc"), counter)

    result = await generate_tailored_cv(extract, llm=mock_llm)

    # The invented company failed factual integrity every attempt → regenerated to cap.
    assert counter["generate_experience"] == REGEN_CAP + 1
    assert any(
        f.kind is FlagKind.CAPPED_SECTION and f.section is TargetSection.EXPERIENCE
        for f in result.flags
    )


async def test_pipeline_with_languages_converges_with_no_flags(mock_llm: MagicMock) -> None:
    facts = _facts()
    facts.languages = [FactsLanguage(language="English", level="Native")]
    extract = ExtractResult(facts=facts, jd=_jd(["Python", "FastAPI"]), flags=[])
    responses = _responses()
    responses["generate_language"] = {"languages": [{"language": "English", "level": "Native"}]}
    mock_llm.call_tool.side_effect = _router(responses)

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, TailoredResult)
    assert result.flags == []  # no phantom capped_section for languages


async def test_pipeline_missing_required_field_returns_actionable_error(
    mock_llm: MagicMock,
) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    responses = _responses()
    responses["generate_summary"] = {"relevant_skills": ["Python"]}  # omits required 'text'
    mock_llm.call_tool.side_effect = _router(responses)

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, dict)  # error envelope, never a crash
    assert result["stage"] == "generate"
    assert "missing required field" in result["error"].lower()
    assert "text" in result["error"]  # names the missing field


async def test_pipeline_empty_experience_does_not_silently_pass(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    responses = _responses()
    responses["generate_experience"] = {"experiences": []}  # source has work history
    counter: dict = {}
    mock_llm.call_tool.side_effect = _router(responses, counter)

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, TailoredResult)
    assert counter["generate_experience"] == REGEN_CAP + 1  # regenerated, not accepted
    assert any(
        f.kind is FlagKind.CAPPED_SECTION and f.section is TargetSection.EXPERIENCE
        for f in result.flags
    )


async def test_pipeline_null_summary_field_returns_actionable_error(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    responses = _responses()
    responses["generate_summary"] = {"text": None, "relevant_skills": ["Python"]}
    mock_llm.call_tool.side_effect = _router(responses)

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, dict)  # envelope, never a 500
    assert result["stage"] == "generate"
    assert "text" in result["error"]


async def test_pipeline_null_skills_field_returns_actionable_error(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    responses = _responses()
    responses["generate_skills"] = {"categories": None}
    mock_llm.call_tool.side_effect = _router(responses)

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, dict)
    assert result["stage"] == "generate"
    assert "categories" in result["error"].lower()


async def test_pipeline_transport_error_returns_envelope(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    mock_llm.call_tool.side_effect = openai.APITimeoutError(
        request=httpx.Request("POST", "https://openrouter.ai/api/v1")
    )

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, dict)
    assert result["stage"] == "generate"


async def test_pipeline_propagates_extract_keyword_gap_flag(mock_llm: MagicMock) -> None:
    extract = ExtractResult(
        facts=_facts(),
        jd=_jd(["Python", "FastAPI"]),
        flags=["Only 2 JD keywords found (fewer than 5)."],
    )
    mock_llm.call_tool.side_effect = _router(_responses())

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, TailoredResult)
    assert any(
        f.kind is FlagKind.UNMET_COVERAGE and "keyword" in f.message.lower()
        for f in result.flags
    )


async def test_did_not_converge_flag_not_attributed_to_skills(mock_llm: MagicMock) -> None:
    extract = ExtractResult(
        facts=_facts(),
        jd=_jd(["Python", "Kubernetes", "Terraform", "Go"]),  # mostly absent → below target
        flags=[],
    )
    mock_llm.call_tool.side_effect = _router(_responses())

    result = await generate_tailored_cv(extract, llm=mock_llm)

    did_not_converge = [f for f in result.flags if f.kind is FlagKind.DID_NOT_CONVERGE]
    assert did_not_converge
    assert all(f.section != TargetSection.SKILLS for f in did_not_converge)


async def test_pipeline_provider_failure_returns_redacted_envelope(mock_llm: MagicMock) -> None:
    extract = ExtractResult(facts=_facts(), jd=_jd(["Python"]), flags=[])
    mock_llm.call_tool.side_effect = ProviderResponseError(
        "upstream 500 leaked sk-or-test-dummy in the body"
    )

    result = await generate_tailored_cv(extract, llm=mock_llm)

    assert isinstance(result, dict)
    assert result["stage"] == "generate"
    assert "sk-or-test-dummy" not in result["error"]  # key redacted
    assert "***REDACTED***" in result["error"]
