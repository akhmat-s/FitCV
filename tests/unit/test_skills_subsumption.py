"""Unit tests for intra-section subsumed-substring dedup in the Skills section.

A short skill phrase wholly contained — on word boundaries — in a longer phrase kept in the
SAME section adds zero ATS coverage (the scorer matches the keyword via the longer phrase) and
only eats the one-page budget. ``_dedup_subsumed_skills`` drops the subsumed phrase, keeps the
longer subsuming one, runs across ALL categories, reuses the scorer's word-boundary matcher
(so "Java" is never subsumed by "JavaScript"), is coverage-neutral, and never empties a section.
"""

from __future__ import annotations

from copy import deepcopy

from cv_generator import SPOKEN_LANGUAGES_CATEGORY, _dedup_subsumed_skills, score_ats
from helprers.cv_template import (
    Category,
    CVTemplate,
    PersonalInfo,
    Skills,
    Summary,
)
from schemas import CandidateLevel, JdAnalysis


def _skills(*categories: Category) -> Skills:
    return Skills(categories=list(categories))


def _cv(skills: Skills) -> CVTemplate:
    return CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=skills,
    )


def _jd(keywords: list[str]) -> JdAnalysis:
    return JdAnalysis(
        role_title="QA Engineer",
        company="Globex",
        keywords=list(keywords),
        requirements_must=["5+ years of test automation experience"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )


def _rendered(skills: Skills) -> list[str]:
    return [kw for cat in skills.categories for kw in cat.keywords]


# --- subsumption drop ----------------------------------------------------


def test_subsumed_shorter_phrase_is_dropped_longer_kept() -> None:
    skills = _skills(
        Category(
            category="Quality Engineering",
            keywords=[
                "test automation",
                "SDET",
                "regression testing",
                "Mobile/Web/API test automation",
            ],
        )
    )
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    assert "test automation" not in rendered  # subsumed by the longer phrase
    assert "Mobile/Web/API test automation" in rendered  # the longer phrase survives
    assert "SDET" in rendered  # not subsumed → kept
    assert "regression testing" in rendered  # not subsumed → kept


def test_spoken_language_is_exempt_from_subsumption() -> None:
    # the Spoken Languages category is exempt from subsumption (matching _validate_skills /
    # _normalize_skill_headers). A real spoken language must NEVER be dropped just because it is a
    # whole-word substring of a longer skill phrase in another category (truth violation).
    skills = _skills(
        Category(category="Standards", keywords=["English communication standards"]),
        Category(category=SPOKEN_LANGUAGES_CATEGORY, keywords=["English"]),
    )
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    assert "English" in rendered  # the real spoken language survives
    assert "English communication standards" in rendered  # the longer skill phrase also kept


def test_subsumption_applies_across_categories() -> None:
    skills = _skills(
        Category(category="Core", keywords=["test automation"]),
        Category(category="Specialties", keywords=["Mobile/Web/API test automation"]),
    )
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    assert rendered == ["Mobile/Web/API test automation"]


# --- word-boundary safety ------------------------------------------------


def test_java_not_subsumed_by_javascript() -> None:
    skills = _skills(Category(category="Languages", keywords=["Java", "JavaScript"]))
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    assert "Java" in rendered  # NOT a whole-word substring of "JavaScript"
    assert "JavaScript" in rendered


# --- non-subsumption is kept ---------------------------------------------


def test_non_subsumed_overlapping_phrases_both_survive() -> None:
    skills = _skills(
        Category(category="QA", keywords=["test automation", "test cases"])
    )
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    assert "test automation" in rendered  # neither contains the other whole-word
    assert "test cases" in rendered


# --- coverage-neutral ----------------------------------------------------


def test_dedup_is_coverage_neutral() -> None:
    skills = _skills(
        Category(
            category="Quality Engineering",
            keywords=[
                "test automation",
                "SDET",
                "regression testing",
                "Mobile/Web/API test automation",
            ],
        )
    )
    jd = _jd(["test automation", "SDET", "regression testing"])

    before_cv = _cv(deepcopy(skills))
    before = score_ats(before_cv, jd, original_cv_text="nothing relevant here")

    _dedup_subsumed_skills(skills)
    after_cv = _cv(skills)
    after = score_ats(after_cv, jd, original_cv_text="nothing relevant here")

    assert after.before_pct == before.before_pct
    assert after.after_pct == before.after_pct
    assert after.matched == before.matched
    assert after.missing == before.missing


# --- degenerate guard (never empty) --------------------------------------


def test_chain_of_subsumed_phrases_keeps_single_longest() -> None:
    skills = _skills(
        Category(category="Skills", keywords=["A", "A B", "A B C"])
    )
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    assert rendered == ["A B C"]  # only the maximal phrase survives, section never empty


# --- universality intact (non-software fixture unaffected) ---------------


def test_nurse_skills_unaffected_by_subsumption() -> None:
    skills = _skills(
        Category(
            category="Clinical Skills",
            keywords=["triage", "medication administration", "wound care"],
        ),
        Category(category="Certifications", keywords=["ACLS", "BLS"]),
    )
    _dedup_subsumed_skills(skills)
    rendered = _rendered(skills)

    for kept in ("triage", "medication administration", "wound care", "ACLS", "BLS"):
        assert kept in rendered  # none is a whole-word substring of another
