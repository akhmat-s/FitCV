"""Adversarial tests for deterministic Skills category-header normalization.

A clean single-concept header is an INVARIANT BY CONSTRUCTION (``_normalize_header`` /
``_normalize_skill_headers``) — not validated-and-rejected. These tests feed deliberately
MALFORMED headers and assert the clean shape is produced WITHOUT a regen, that the change is
coverage-neutral (the scorer never reads the header), that no internal diagnostic leaks to the
user-facing flags, and that the rule is domain-NEUTRAL (a nurse header normalizes identically).
"""

from __future__ import annotations

import logging

import pytest

from cv_generator import (
    SPOKEN_LANGUAGES_CATEGORY,
    SectionValidation,
    _normalize_header,
    _normalize_skill_headers,
    _resolve_section,
    score_ats,
    validate_section,
)
from helprers.cv_template import (
    Category,
    CVTemplate,
    PersonalInfo,
    Skills,
    Summary,
)
from schemas import (
    CandidateLevel,
    CVFacts,
    FactsPersonalInfo,
    JdAnalysis,
    TargetSection,
)

# ── Part A: the pure structural normalizer (domain-neutral) ───────────────────


@pytest.mark.parametrize(
    "header",
    [
        "CI/CD",
        "TCP/IP",
        "A/B Testing",
        "R&D",
        "Health & Safety",
        "Frameworks/Libraries",
        "Research and Development",
        "UX/UI Design",
    ],
)
def test_normalize_keeps_atomic_single_joiner_header_unchanged(header: str) -> None:
    # A label is multi-concept ONLY with a comma OR >1 conjunction joiner. A SINGLE "&", "/",
    # or "and" is part of an ATOMIC header — kept verbatim (each below is ≤3 words / ≤24 chars).
    # A split-on-any-joiner approach would corrupt these (CI/CD→CI, TCP/IP→TCP, R&D→R).
    assert _normalize_header(header) == header


def test_normalize_single_joiner_header_kept_whole_then_capped() -> None:
    # A 4-word single-"&" header stays atomic, then the ≤3-word cap + dangling-joiner strip apply.
    assert _normalize_header("Quality Infrastructure & Testing") == "Quality Infrastructure"
    assert _normalize_header("AI & Quality Engineering") == "AI & Quality"
    # "Programming Languages & Tools" → cap → "Programming Languages &" → strip the dangling "&".
    assert _normalize_header("Programming Languages & Tools") == "Programming Languages"


def test_normalize_multi_joiner_keeps_single_clean_segment() -> None:
    out = _normalize_header("CI/CD, Testing & AI Integration")
    assert out == "AI Integration"  # the only 2-word segment after splitting on /, comma, &
    assert "," not in out and "&" not in out and "/" not in out
    assert len(out.split()) <= 3


def test_normalize_word_count_tie_breaks_to_first() -> None:
    # "Build and Ship and Deploy" → split on "and" → 3 one-word segments → FIRST wins.
    assert _normalize_header("Build and Ship and Deploy") == "Build"


def test_normalize_strips_markdown_to_plain() -> None:
    assert _normalize_header("**Skills**") == "Skills"
    assert "*" not in _normalize_header("__Tooling__")


def test_normalize_caps_to_three_words_and_24_chars() -> None:
    out = _normalize_header("Senior Quality Assurance Test Automation Specialist")
    assert len(out.split()) <= 3
    assert len(out) <= 24


def test_normalize_degenerate_blank_or_markup_only_is_empty() -> None:
    # The SOLE headerless case: markup-only / blank → empty (renders flat).
    assert _normalize_header("***") == ""
    assert _normalize_header("   ") == ""
    assert _normalize_header(None) == ""
    assert _normalize_header(123) == ""


def test_normalize_does_not_split_the_word_inside_a_token() -> None:
    # Domain-neutral word-boundary: the "and" inside "Standards" / "Brand" is never a joiner.
    assert _normalize_header("Standards") == "Standards"
    assert _normalize_header("Brand") == "Brand"


def test_normalize_is_idempotent() -> None:
    for raw in ("AI & Quality Engineering", "**Skills**", "CI/CD, Testing & AI Integration"):
        once = _normalize_header(raw)
        assert _normalize_header(once) == once


# ── Universality: identical structural behavior for a non-software field ───────


def test_normalize_nurse_compound_header_no_software_bias() -> None:
    # A single-"&" clinical header is atomic (kept whole) by the SAME domain-neutral rule, then the
    # ≤3-word cap + dangling-joiner strip reduce a 4+-word header (no profession-specific logic).
    assert _normalize_header("Clinical Skills & Patient Care") == "Clinical Skills"
    assert _normalize_header("Patient Care & Triage") == "Patient Care"


# ── _normalize_skill_headers applies to every category, exempts Spoken Languages ─


def test_normalize_skill_headers_applies_and_exempts_spoken_languages() -> None:
    skills = Skills(
        categories=[
            Category(category="AI & Quality Engineering", keywords=["Python"]),
            Category(category="***", keywords=["Docker"]),
            Category(category=SPOKEN_LANGUAGES_CATEGORY, keywords=["English"]),
        ]
    )
    _normalize_skill_headers(skills)
    assert skills.categories[0].category == "AI & Quality"  # single "&" atomic, capped to 3 words
    assert skills.categories[1].category == ""  # degenerate → flat
    assert skills.categories[2].category == SPOKEN_LANGUAGES_CATEGORY  # exempt, untouched


def test_normalize_skill_headers_merges_siblings_with_same_normalized_label() -> None:
    # FIX 3: two raw headers that normalize to the SAME label must MERGE into one category, not
    # render as duplicate sibling lines. Keywords combine with case-insensitive first-seen dedup.
    skills = Skills(
        categories=[
            Category(category="Programming Languages & Tools", keywords=["Python", "Bash"]),
            Category(category="Programming Languages, Frameworks", keywords=["Python", "Django"]),
            Category(category=SPOKEN_LANGUAGES_CATEGORY, keywords=["English"]),
        ]
    )
    _normalize_skill_headers(skills)
    non_spoken = [c for c in skills.categories if c.category != SPOKEN_LANGUAGES_CATEGORY]
    assert len(non_spoken) == 1  # both normalized to "Programming Languages" → merged
    assert non_spoken[0].category == "Programming Languages"
    assert non_spoken[0].keywords == ["Python", "Bash", "Django"]  # first-seen dedup, order-stable
    # the spoken category stays separate and LAST
    assert skills.categories[-1].category == SPOKEN_LANGUAGES_CATEGORY
    assert skills.categories[-1].keywords == ["English"]


def test_sibling_merge_is_coverage_neutral() -> None:
    # FIX 3: merging siblings only regroups/dedups keyword TEXT the scorer already reads — the
    # before→after coverage numbers must not move (a cross-category exact dup is deduped anyway).
    jd = _jd(["Python", "Django", "Bash"])
    raw = _cv_from_skills(
        Skills(
            categories=[
                Category(category="Programming Languages & Tools", keywords=["Python", "Bash"]),
                Category(category="Programming Languages, Frameworks", keywords=["Django"]),
            ]
        )
    )
    merged_skills = Skills(
        categories=[
            Category(category="Programming Languages & Tools", keywords=["Python", "Bash"]),
            Category(category="Programming Languages, Frameworks", keywords=["Django"]),
        ]
    )
    _normalize_skill_headers(merged_skills)
    merged = _cv_from_skills(merged_skills)

    before = score_ats(raw, jd, original_cv_text="none")
    after = score_ats(merged, jd, original_cv_text="none")

    assert before.after_pct == after.after_pct
    assert before.before_pct == after.before_pct
    assert before.matched == after.matched
    assert before.missing == after.missing


# ── Coverage-neutrality: the scorer never reads the header ────────────────────


def _jd(keywords: list[str]) -> JdAnalysis:
    return JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=list(keywords),
        requirements_must=["3+ years experience"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )


def _cv(header: str, keywords: list[str]) -> CVTemplate:
    return CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=Skills(categories=[Category(category=header, keywords=keywords)]),
    )


def _cv_from_skills(skills: Skills) -> CVTemplate:
    return CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=skills,
    )


def test_coverage_identical_before_and_after_header_normalization() -> None:
    # The malformed header even CONTAINS a JD keyword ("Testing") — proving the scorer counts only
    # the rendered keyword TEXT (word-boundary), never the header. Normalizing the header must not
    # move a single coverage number, field by field.
    jd = _jd(["Python", "FastAPI", "Testing"])
    keywords = ["Python", "FastAPI"]
    cv_raw = _cv("Testing & Quality Engineering", keywords)
    cv_norm = _cv(_normalize_header("Testing & Quality Engineering"), list(keywords))

    before = score_ats(cv_raw, jd, original_cv_text="Original mentions Python.")
    after = score_ats(cv_norm, jd, original_cv_text="Original mentions Python.")

    assert cv_norm.skills.categories[0].category != cv_raw.skills.categories[0].category
    assert before.after_pct == after.after_pct
    assert before.before_pct == after.before_pct
    assert before.matched == after.matched
    assert before.missing == after.missing  # "Testing" stays missing — header never counts


# ── Part C: regen-exhaustion diagnostic logs server-side, never leaks to flags ─


async def test_capped_section_logs_raw_errors_and_flag_carries_no_rule_text(
    caplog: object,
) -> None:
    async def generate_fn():  # always-failing summary (awaited by _resolve_section)
        return Summary(text="x")

    validate_fn = lambda _o: SectionValidation(  # noqa: E731
        errors=["Skills header violates format rules: AI & Quality Engineering"]
    )
    with caplog.at_level(logging.WARNING):
        _obj, flag = await _resolve_section(TargetSection.SUMMARY, generate_fn, validate_fn)

    # the raw internal rule text is LOGGED server-side (debuggable)
    assert "violates format rules" in caplog.text
    assert "still failing after" in caplog.text
    # but it NEVER reaches the user-facing flag message
    assert flag is not None
    assert "violates format rules" not in flag.message
    assert "still failing after" not in flag.message


def test_validate_skills_still_blocks_real_failures() -> None:
    # Part B removed ONLY header format. Non-vacuous + concrete-literal rules still validate/regen.
    facts = CVFacts(personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"))
    jd = _jd(["Python"])

    empty = Skills(categories=[])
    result_empty = validate_section(TargetSection.SKILLS, empty, facts, jd)
    assert any("empty" in e.lower() for e in result_empty.errors)

    fabricated = Skills(categories=[Category(category="Tooling", keywords=["Kubernetes"])])
    result_fab = validate_section(TargetSection.SKILLS, fabricated, facts, jd)
    assert any("fabrication" in e.lower() for e in result_fab.errors)
