"""Bounded intra-request concurrency: determinism, concurrency bound, coverage parity.

Domain-neutral (the fixture is a generic engineer; no profession vocabulary in the assertions).
These tests prove the concurrent section fan-out (``cv_generator._generate_all_sections`` via
``bounded_gather``) produces output IDENTICAL to a sequential run regardless of which section's
provider call finishes first, never exceeds ``MAX_CONCURRENT_LLM_CALLS`` in-flight calls, and does
not change the ATS coverage number.
"""

from __future__ import annotations

import asyncio

from cv_generator import (
    COVERAGE_TARGET_PCT,
    TailoredResult,
    generate_tailored_cv,
)
from main import _map_ats, _map_cv
from schemas import (
    MAX_CONCURRENT_LLM_CALLS,
    CandidateLevel,
    CVFacts,
    ExtractResult,
    FactsCertificate,
    FactsEducation,
    FactsExperience,
    FactsLanguage,
    FactsPersonalInfo,
    FactsProject,
    JdAnalysis,
)

_KEYWORDS = ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"]

# Per-section tool name → its definition-order index (used to build scrambling delay maps).
_TOOL_NAMES = [
    "generate_summary",
    "generate_skills",
    "generate_experience",
    "generate_education",
    "generate_project",
    "generate_certificate",
    "generate_language",
]


def _extract() -> ExtractResult:
    """A rich extract that populates ALL seven sections, so the fan-out is maximal (> the cap)."""
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Sam Dev", location="Remote", email="sam@x.io"),
        experiences=[
            FactsExperience(
                role="Engineer",
                company="Acme",
                start_date="2020",
                end_date="2022",
                # bullets evidence every JD keyword so the truth-preserving skills gate keeps them
                bullets=[
                    "Built Python and FastAPI services with Pydantic on PostgreSQL and Docker"
                ],
            )
        ],
        education=[FactsEducation(institution="MIT", degree="BSc CS")],
        projects=[FactsProject(name="Svc", description="Python FastAPI service")],
        certificates=[FactsCertificate(title="AWS SAA", issuer="Amazon", year=2021)],
        languages=[FactsLanguage(language="English", level="Native")],
    )
    jd = JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=list(_KEYWORDS),
        requirements_must=["Python"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )
    return ExtractResult(facts=facts, jd=jd, flags=[])


def _responses() -> dict:
    return {
        "generate_summary": {"text": "line a\nline b\nline c", "relevant_skills": ["Python"]},
        "generate_skills": {
            "categories": [{"category": "Languages", "keywords": list(_KEYWORDS)}]
        },
        "generate_experience": {
            "experiences": [
                {
                    "role": "Senior Engineer",
                    "company": "Acme",
                    "company_description": "A large enterprise software company today",
                    "start_date": "2020",
                    "end_date": "2022",
                    "bullets": [
                        {"action_verb": "Developed", "description": "built Python services"}
                    ],
                }
            ]
        },
        "generate_education": {"education": [{"institution": "MIT", "degree": "BSc CS"}]},
        "generate_project": {
            "projects": [{"name": "Svc", "description": "Python FastAPI service"}]
        },
        "generate_certificate": {
            "certificates": [{"title": "AWS SAA", "issuer": "Amazon", "year": 2021}]
        },
        "generate_language": {"languages": [{"language": "English", "level": "Native"}]},
    }


class _FakeLLM:
    """Async fake provider: routes a response by tool name, optionally delays each call to scramble
    completion order, and records peak in-flight concurrency + the order the calls completed in."""

    def __init__(self, responses: dict, delays: dict[str, float] | None = None) -> None:
        self._responses = responses
        self._delays = delays or {}
        self._in_flight = 0
        self.max_in_flight = 0
        self.completion_order: list[str] = []

    async def call_tool(self, _system: str, _user: str, schema: dict) -> dict:
        name = schema["name"]
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._delays.get(name, 0.0))
            return self._responses[name]
        finally:
            self._in_flight -= 1
            self.completion_order.append(name)


def _snapshot(result: TailoredResult) -> tuple:
    """A thorough, completion-order-independent fingerprint of the pipeline output.

    Maps the CV + ATS to their Pydantic mirrors (full structure incl. section_order) and freezes the
    flags, so two runs are byte-for-byte comparable regardless of which section finished first.
    """
    return (
        _map_cv(result.cv).model_dump(mode="json"),
        _map_ats(result.ats_score).model_dump(mode="json"),
        tuple((str(f.section), f.kind.value, f.message) for f in result.flags),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  DETERMINISM LOCK — concurrent output equals sequential, regardless of finish order
# ──────────────────────────────────────────────────────────────────────────────


async def test_concurrent_section_generation_is_deterministic() -> None:
    """The #1 risk: two runs whose sections COMPLETE in different orders must produce byte-identical
    output (CV + ats_score + flags). Section assembly follows section_order, never arrival order."""
    extract = _extract()
    # Two pronounced, opposite delay gradients → the sections finish in genuinely different orders.
    delays_fwd = {name: 0.002 * i for i, name in enumerate(_TOOL_NAMES)}  # summary finishes first
    delays_rev = {name: 0.002 * (len(_TOOL_NAMES) - i) for i, name in enumerate(_TOOL_NAMES)}

    fake_fwd = _FakeLLM(_responses(), delays_fwd)
    fake_rev = _FakeLLM(_responses(), delays_rev)
    result_fwd = await generate_tailored_cv(extract, llm=fake_fwd)
    result_rev = await generate_tailored_cv(extract, llm=fake_rev)

    assert isinstance(result_fwd, TailoredResult)
    assert isinstance(result_rev, TailoredResult)
    # The scramble was REAL — the two runs completed their calls in different orders ...
    assert fake_fwd.completion_order != fake_rev.completion_order
    # ... yet the assembled output is identical (determinism preserved under concurrency).
    assert _snapshot(result_fwd) == _snapshot(result_rev)


async def test_section_order_is_fixed_not_completion_order() -> None:
    """Assembly orders sections by the level-driven section_order, never by which finished first."""
    extract = _extract()
    delays = {name: 0.002 * (len(_TOOL_NAMES) - i) for i, name in enumerate(_TOOL_NAMES)}
    result = await generate_tailored_cv(extract, llm=_FakeLLM(_responses(), delays))

    assert isinstance(result, TailoredResult)
    # senior_ic order: contact, summary, skills, experience, education, projects, (+ certificates,
    # languages appended when populated) — fixed irrespective of the scrambled completion order.
    assert result.cv.section_order == [
        "contact",
        "summary",
        "skills",
        "experience",
        "education",
        "projects",
        "certificates",
        "languages",
    ]


# ──────────────────────────────────────────────────────────────────────────────
#  CONCURRENCY BOUND — never more than MAX_CONCURRENT_LLM_CALLS in flight
# ──────────────────────────────────────────────────────────────────────────────


async def test_in_flight_calls_never_exceed_cap() -> None:
    """With 7 sections fanning out but the semaphore capped at MAX_CONCURRENT_LLM_CALLS, the peak
    in-flight provider calls must equal the cap (proving the bound is active) and never exceed it
    (rate-limit safety — _RATE_LIMIT_ERROR must not be provoked)."""
    extract = _extract()
    # A uniform delay forces the slots to overlap so the peak is actually reached and observable.
    fake = _FakeLLM(_responses(), {name: 0.02 for name in _TOOL_NAMES})

    result = await generate_tailored_cv(extract, llm=fake)

    assert isinstance(result, TailoredResult)
    assert fake.max_in_flight <= MAX_CONCURRENT_LLM_CALLS  # never exceeds the cap
    assert fake.max_in_flight == MAX_CONCURRENT_LLM_CALLS  # cap genuinely reached (7 sections > 5)


# ──────────────────────────────────────────────────────────────────────────────
#  COVERAGE NEUTRALITY — the ATS number is unchanged by the concurrency migration
# ──────────────────────────────────────────────────────────────────────────────


async def test_coverage_after_pct_is_migration_invariant() -> None:
    """score_ats.after_pct on an identical CV is unchanged vs the pre-migration result: every JD
    keyword is evidenced and surfaced, so after_pct is exactly 100.0 (deterministic, no concurrency
    drift), and the result converges above target with no honest-gap flags."""
    extract = _extract()
    result = await generate_tailored_cv(extract, llm=_FakeLLM(_responses()))

    assert isinstance(result, TailoredResult)
    assert result.ats_score.after_pct == 100.0
    assert result.ats_score.after_pct >= COVERAGE_TARGET_PCT
    assert sorted(result.ats_score.matched, key=str.lower) == sorted(_KEYWORDS, key=str.lower)
    assert result.ats_score.missing == []
    assert result.flags == []  # clean convergence — no caps, no did-not-converge, no page pressure
