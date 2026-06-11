"""Unit tests for the section-local regeneration loop.

The loop regenerates ONLY a failing section, bounded by REGEN_CAP, and on cap-hit emits
the section with a `capped_section` flag instead of hanging (no infinite loop).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import cv_generator
from cv_generator import (
    REGEN_CAP,
    FlagKind,
    SectionValidation,
    _resolve_section,
    generate_tailored_cv,
)
from helprers.cv_template import Category, Language, Skills, Summary
from schemas import (
    CandidateLevel,
    CVFacts,
    ExtractResult,
    FactsPersonalInfo,
    FactsProject,
    JdAnalysis,
    TargetSection,
)

_KEYWORDS = ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"]


def _extract() -> ExtractResult:
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        # a source project evidences the JD keywords so the mocked passing skills survive the
        # anti-fabrication gate (the experience generator is mocked to [], so evidence can't
        # live in experience without tripping the empty-experience gate)
        projects=[
            FactsProject(name="Svc", description="Python FastAPI Pydantic PostgreSQL Docker")
        ],
    )
    jd = JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=list(_KEYWORDS),
        requirements_must=["Python"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )
    return ExtractResult(facts=facts, jd=jd, flags=[])


def _as_async(produce):
    """Wrap a value-producing callable as an awaited async section generator.

    ``_resolve_section`` now ``await``s the section generator, so a monkeypatched replacement
    must be a coroutine function; each call still invokes ``produce`` so the MagicMock call
    counts the test asserts on are preserved.
    """

    async def _generate(_facts, _jd, _llm):
        return produce()

    return _generate


async def test_failing_section_regenerates_at_most_cap_times() -> None:
    generate_fn = AsyncMock(return_value=Summary(text="x"))
    validate_fn = MagicMock(return_value=SectionValidation(errors=["boom"]))

    _obj, flag = await _resolve_section(TargetSection.SUMMARY, generate_fn, validate_fn)

    assert generate_fn.call_count == REGEN_CAP + 1
    assert flag is not None


async def test_passing_section_is_generated_once() -> None:
    generate_fn = AsyncMock(return_value=Summary(text="a\nb\nc"))
    validate_fn = MagicMock(return_value=SectionValidation())  # no errors

    _obj, flag = await _resolve_section(TargetSection.SUMMARY, generate_fn, validate_fn)

    assert generate_fn.call_count == 1
    assert flag is None


async def test_always_failing_section_emits_capped_flag() -> None:
    sentinel = Summary(text="still broken")
    generate_fn = AsyncMock(return_value=sentinel)
    validate_fn = MagicMock(return_value=SectionValidation(errors=["boom"]))

    obj, flag = await _resolve_section(TargetSection.SUMMARY, generate_fn, validate_fn)

    assert generate_fn.call_count == REGEN_CAP + 1
    assert flag is not None
    assert flag.kind is FlagKind.CAPPED_SECTION
    assert obj is sentinel  # the section is still emitted, not dropped


async def test_capped_string_section_reports_own_section() -> None:
    generate_fn = AsyncMock(return_value=[Language(language="Klingon", level="Native")])
    validate_fn = MagicMock(return_value=SectionValidation(errors=["boom"]))

    _obj, flag = await _resolve_section("languages", generate_fn, validate_fn)

    assert flag is not None
    assert flag.section == "languages"
    assert flag.section != TargetSection.SUMMARY


async def test_only_failing_section_regenerates(monkeypatch) -> None:
    failing_summary = MagicMock(return_value=Summary(text="single line fails validation"))
    passing_skills = MagicMock(
        return_value=Skills(categories=[Category(category="Languages", keywords=list(_KEYWORDS))])
    )
    empty = MagicMock(return_value=[])

    monkeypatch.setattr(cv_generator, "_generate_summary", _as_async(failing_summary))
    monkeypatch.setattr(cv_generator, "_generate_skills", _as_async(passing_skills))
    for attr in (
        "_generate_experience",
        "_generate_education",
        "_generate_project",
        "_generate_certificate",
        "_generate_language",
    ):
        monkeypatch.setattr(cv_generator, attr, _as_async(empty))

    result = await generate_tailored_cv(_extract(), llm=MagicMock())

    # Summary fails every attempt → regenerated REGEN_CAP times; skills passes once.
    assert failing_summary.call_count == REGEN_CAP + 1
    assert passing_skills.call_count == 1
    # The capped summary is surfaced as a flag, never an error.
    assert any(f.kind is FlagKind.CAPPED_SECTION for f in result.flags)
