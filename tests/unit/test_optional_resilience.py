"""Finish the half-applied ``fields → Optional`` resilience refactor (single commit).

Reproduce-then-fix coverage for the downstream consumers that still assumed the now-Optional
extract fields are present: the skills-shape crash (A), the languages-only non-vacuous gate (B),
the strict generator/schema vs. Optional extract mismatch (C), the bare-URL/None render leaks (D),
the catch-all stage label (E), the cross-category dedup dropping a DECLARED skill (F), the
over-grabbing year coercer (G), and the render KeyError on a contract-skewed 200 (H).

Field-agnostic by construction — no profession lexicon appears in any assertion.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import app
from cv_generator import (
    SkillProvenance,
    _build_link,
    _dedup_subsumed_skills,
    _generate_certificate,
    _generate_project,
    _generate_skills,
    _validate_skills,
)
from helprers.cv_template import Category, Skills
from main import _on_unexpected_error
from schemas import (
    CVFacts,
    FactsCertificate,
    FactsEducation,
    FactsLanguage,
    FactsPersonalInfo,
    FactsProject,
    FactsSkillGroup,
    JdAnalysis,
    KeywordTier,
    _coerce_optional_year,
)


def _facts(**overrides: object) -> CVFacts:
    base: dict = {
        "personal_info": FactsPersonalInfo(name="Sam Doe", email="sam@example.com"),
    }
    base.update(overrides)
    return CVFacts(**base)


# --- A. Non-string skill keyword must not crash the whole request -------------


async def test_non_string_skill_keyword_is_dropped_not_a_crash() -> None:
    """A model that emits ``{"keyword": 3}`` must drop that item, not raise TypeError."""
    facts = _facts(
        skills=[FactsSkillGroup(category="Group", items=["Alpha"])],
    )
    jd = JdAnalysis(keywords=["alpha"], keyword_tiers={"alpha": "concrete"})
    llm = AsyncMock()
    llm.call_tool.return_value = {
        "categories": [
            {
                "category": "Group",
                "keywords": [
                    {"keyword": 3, "tier": "concrete", "category": "Group"},  # malformed shape
                    {"keyword": "Alpha", "tier": "concrete", "category": "Group"},
                ],
            }
        ]
    }

    skills = await _generate_skills(facts, jd, llm)  # must NOT raise

    rendered = [kw for c in skills.categories for kw in c.keywords]
    assert "Alpha" in rendered
    assert 3 not in rendered


async def test_non_list_categories_degrades_to_clean_section_not_a_crash() -> None:
    """A non-list ``categories`` must degrade to an empty surface, never crash."""
    facts = _facts()
    jd = JdAnalysis(keywords=["alpha"], keyword_tiers={"alpha": "concrete"})
    llm = AsyncMock()
    llm.call_tool.return_value = {"categories": "not-a-list"}

    skills = await _generate_skills(facts, jd, llm)  # must NOT raise

    assert all(not c.keywords for c in skills.categories)


# --- B. A Skills section of ONLY spoken languages is not vacuously valid -------


def test_validate_skills_languages_only_fails_non_vacuous_gate() -> None:
    """Zero REAL skill keywords (only Spoken Languages) → blocking error, not a pass."""
    skills = Skills(
        categories=[Category(category="Spoken Languages", keywords=["english fluent"])],
        provenance=[],
    )
    facts = _facts(languages=[FactsLanguage(language="English", level="Fluent")])
    jd = JdAnalysis(keywords=["alpha"], keyword_tiers={"alpha": "concrete"})

    result = _validate_skills(skills, facts, jd)

    assert result.errors, "languages-only Skills must fail the non-vacuous gate"


# --- C. Generators/schemas loosen to MATCH the Optional extract ---------------


async def test_certificate_without_issuer_generates_cleanly() -> None:
    facts = _facts(certificates=[FactsCertificate(title="Standard A")])
    jd = JdAnalysis(keywords=[])
    llm = AsyncMock()
    llm.call_tool.return_value = {"certificates": [{"title": "Standard A"}]}  # issuer omitted

    certs = await _generate_certificate(facts, jd, llm)  # must NOT raise

    assert certs[0].title == "Standard A"
    assert certs[0].issuer is None  # omitted, NOT fabricated


async def test_project_without_description_generates_cleanly() -> None:
    facts = _facts(projects=[FactsProject(name="Showcase")])
    jd = JdAnalysis(keywords=[])
    llm = AsyncMock()
    llm.call_tool.return_value = {"projects": [{"name": "Showcase"}]}  # description omitted

    projects = await _generate_project(facts, jd, llm)  # must NOT raise

    assert projects[0].name == "Showcase"
    assert projects[0].description is None  # omitted, NOT fabricated


def test_bare_url_link_round_trips_without_title() -> None:
    link = _build_link({"url": "https://example.com/sam"})  # title omitted — must not raise

    assert link is not None
    assert link.url == "https://example.com/sam"
    assert link.title is None


def test_generator_schemas_no_longer_require_optional_fields() -> None:
    from tool_schemas import (
        _LINK_SCHEMA,
        generate_certificate_tool_schema,
        generate_project_tool_schema,
    )

    proj_item = generate_project_tool_schema()["parameters"]["properties"]["projects"]["items"]
    cert_item = generate_certificate_tool_schema()["parameters"]["properties"]["certificates"][
        "items"
    ]
    assert proj_item["required"] == ["name"]
    assert cert_item["required"] == ["title"]
    assert _LINK_SCHEMA["required"] == ["url"]


# --- D. Render must never leak a literal "None" for an absent Optional ---------


def test_contact_text_bare_url_link_has_no_none_prefix() -> None:
    cv = {
        "personal_info": {
            "name": "SAM DOE",
            "location": "Remote",
            "email": "sam@example.com",
            "links": [{"title": None, "url": "https://example.com/sam"}],
        }
    }
    text = app._contact_text(cv)
    assert "None" not in text
    assert "https://example.com/sam" in text


def test_projects_text_absent_description_has_no_none() -> None:
    cv = {"projects": [{"name": "Showcase", "description": None, "skills": []}]}
    text = app._projects_text(cv)
    assert "None" not in text
    assert "Showcase" in text


def test_certificates_text_absent_issuer_has_no_none() -> None:
    cv = {"certificates": [{"title": "Standard A", "issuer": None, "year": None}]}
    text = app._certificates_text(cv)
    assert "None" not in text
    assert "Standard A" in text


# --- E. Unexpected exception → upstream fallback stage, not "assemble" ---------


def test_unexpected_exception_uses_fallback_stage_not_assemble() -> None:
    request = MagicMock()
    request.url.path = "/generate"
    response = asyncio.run(_on_unexpected_error(request, RuntimeError("boom-internal-detail")))

    body = json.loads(response.body)
    assert body["stage"] != "assemble"
    assert body["stage"] == "extract"  # _FALLBACK_STAGE
    assert response.status_code == 502
    assert "boom-internal-detail" not in body["error"]  # generic, no raw exc leak


# --- F. Cross-category dedup must never drop a DECLARED skill ------------------


def _skills_with_provenance(categories: list[Category], prov: list[SkillProvenance]) -> Skills:
    return Skills(categories=categories, provenance=prov)


def test_dedup_keeps_declared_skill_even_when_subsumed_cross_category() -> None:
    skills = _skills_with_provenance(
        [
            Category(category="One", keywords=["Alpha"]),
            Category(category="Two", keywords=["Alpha pipeline"]),
        ],
        [
            SkillProvenance(keyword="Alpha", tier=KeywordTier.CONCRETE, declared=True),
            SkillProvenance(keyword="Alpha pipeline", tier=KeywordTier.CONCRETE, declared=False),
        ],
    )
    _dedup_subsumed_skills(skills)

    rendered = [kw for c in skills.categories for kw in c.keywords]
    assert "Alpha" in rendered  # declared skill is a fact — never dropped


def test_dedup_still_drops_an_added_subsumed_keyword() -> None:
    skills = _skills_with_provenance(
        [Category(category="One", keywords=["Alpha", "Alpha pipeline"])],
        [
            SkillProvenance(keyword="Alpha", tier=KeywordTier.CONCRETE, declared=False),
            SkillProvenance(keyword="Alpha pipeline", tier=KeywordTier.CONCRETE, declared=False),
        ],
    )
    _dedup_subsumed_skills(skills)

    rendered = [kw for c in skills.categories for kw in c.keywords]
    assert "Alpha" not in rendered  # added + subsumed → dropped (coverage-neutral)
    assert "Alpha pipeline" in rendered


# --- G. Year coercer must not over-grab a 4-digit run from a range/id ----------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026", 2026),
        ("Expected 2026", 2026),
        ("  2026  ", 2026),
        ("Cohort 2024-2025", None),  # range — ambiguous, don't guess
        ("Batch 2019-A", None),  # extra alphanumerics around the year
        ("2024-present", None),  # range-like — don't grab 2024
        ("Present", None),
        ("", None),
    ],
)
def test_coerce_optional_year_only_accepts_a_clean_year(value: str, expected: int | None) -> None:
    assert _coerce_optional_year(value) == expected


# --- G2. Year coercer must never RAISE on a non-finite float (json inf/nan) ----


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_coerce_optional_year_non_finite_float_is_none(value: float) -> None:
    # json.loads parses 1e400 → inf and the NaN token → nan; int(inf) raises OverflowError and
    # int(nan) raises ValueError (NOT a pydantic error), escaping model_validate and breaking the
    # never-502 resilience contract. A non-finite year must coerce to None, never raise.
    assert _coerce_optional_year(value) is None


def test_coerce_optional_year_finite_float_truncates() -> None:
    assert _coerce_optional_year(2023.0) == 2023


def test_model_validate_tolerates_non_finite_year() -> None:
    # The live extract path: a non-finite year must drop to None, never bubble an OverflowError.
    edu = FactsEducation.model_validate(
        {"institution": "X", "degree": "Y", "start_year": float("inf"), "end_year": float("nan")}
    )
    assert edu.start_year is None and edu.end_year is None
    cert = FactsCertificate.model_validate({"title": "Z", "year": float("inf")})
    assert cert.year is None


# --- H. Render tolerates a contract-skewed 200 (missing optional sub-fields) ---


def test_render_ats_panel_tolerates_missing_matched_missing() -> None:
    ats = {"after_pct": 50.0, "before_pct": 20.0}  # matched/missing absent (skewed payload)
    app.render_ats_panel(MagicMock(), ats)  # must NOT KeyError


def test_render_result_tolerates_missing_flags() -> None:
    result = {
        "ats_score": {"matched": [], "missing": [], "after_pct": 0.0, "before_pct": 0.0},
        "cv": {"section_order": []},
        "cover_letter": "",
    }  # 'flags' absent — is_success_response does not require it
    app.render_result(MagicMock(), result)  # must NOT KeyError
