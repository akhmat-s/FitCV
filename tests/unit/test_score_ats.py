"""Unit tests for before→after ATS keyword scoring.

`score_ats` computes `|JD_must ∩ CV_keywords| / |JD_must| × 100` for the tailored CV
(after) and the original CV text (before), on post-clean()/remove_ai_tells() text, and
returns the honest-gap `missing[]` list in full.
"""

from __future__ import annotations

import pytest

from cv_generator import (
    COVERAGE_TARGET_PCT,
    AtsScore,
    FlagKind,
    _coverage_flags,
    _cv_to_text,
    _facts_to_text,
    score_ats,
)
from helprers.cv_template import (
    BulletPoint,
    Category,
    Certificate,
    CVTemplate,
    Experience,
    Language,
    PersonalInfo,
    Project,
    Skills,
    Summary,
)
from schemas import (
    CandidateLevel,
    CVFacts,
    FactsCertificate,
    FactsEducation,
    FactsExperience,
    FactsLanguage,
    FactsPersonalInfo,
    FactsProject,
    JdAnalysis,
)


def _cv(relevant: list[str]) -> CVTemplate:
    return CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=Skills(categories=[Category(category="Languages", keywords=relevant)]),
    )


def _jd(keywords: list[str]) -> JdAnalysis:
    return JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=list(keywords),
        requirements_must=["3+ years of distributed systems experience"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )


def test_coverage_formula_before_and_after_and_lift() -> None:
    jd = _jd(["Python", "FastAPI", "Docker", "AWS"])  # 4 must-haves
    cv = _cv(["Python", "FastAPI", "Docker"])  # 3 of 4 present after tailoring

    score = score_ats(cv, jd, original_cv_text="My original CV mentions Python only.")

    assert score.after_pct == 75.0  # 3/4 × 100
    assert score.before_pct == 25.0  # 1/4 × 100 (only Python in the original)
    assert score.coverage_pct == score.after_pct
    assert round(score.after_pct - score.before_pct, 2) == 50.0  # lift


def test_coverage_matches_after_homoglyph_cleanup() -> None:
    jd = _jd(["Python"])
    cv = _cv(["Pythоn"])  # Cyrillic 'о' — cleaned to Latin before matching

    score = score_ats(cv, jd, original_cv_text="nothing relevant here")

    assert "Python" in score.matched
    assert score.after_pct == 100.0


def test_missing_keywords_are_surfaced() -> None:
    jd = _jd(["Python", "Kubernetes"])
    cv = _cv(["Python"])

    score = score_ats(cv, jd, original_cv_text="Python")

    assert "Kubernetes" in score.missing
    assert "Python" in score.matched


def test_short_token_does_not_match_inside_larger_word() -> None:
    jd = _jd(["Go", "Python"])
    cv = _cv(["Google Cloud", "Python"])  # 'Go' only as a substring of Google

    score = score_ats(cv, jd, original_cv_text="nothing relevant")

    assert "Python" in score.matched  # whole-word match counts
    assert "Go" in score.missing  # not falsely matched inside 'Google'


def test_java_does_not_match_inside_javascript() -> None:
    jd = _jd(["Java"])
    cv = _cv(["JavaScript", "React"])

    score = score_ats(cv, jd, original_cv_text="nothing relevant")

    assert "Java" in score.missing


def test_before_basis_includes_passthrough_sections() -> None:
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        education=[FactsEducation(institution="MIT", degree="Computer Science")],
        languages=[FactsLanguage(language="French", level="Native")],
    )

    before = _facts_to_text(facts).lower()

    assert "computer science" in before  # education present in the before basis
    assert "french" in before  # languages present in the before basis


def test_empty_keywords_scores_full_coverage() -> None:
    jd = _jd([])
    cv = _cv(["Python"])

    score = score_ats(cv, jd, original_cv_text="nothing relevant")

    assert score.after_pct == 100.0
    assert score.before_pct == 100.0
    assert score.missing == []


def test_requirement_phrases_are_not_the_coverage_basis() -> None:
    jd = _jd(["Python"])  # requirements_must holds a phrase that the CV never quotes
    cv = _cv(["Python"])

    score = score_ats(cv, jd, original_cv_text="nothing relevant")

    assert score.after_pct == 100.0  # scored on the keyword, not the phrase
    assert score.missing == []


def test_dotted_and_symbol_keywords_match_in_real_tokens() -> None:
    jd = _jd([".NET", "C#"])
    cv = _cv(["ASP.NET Core", "C#"])

    score = score_ats(cv, jd, original_cv_text="nothing relevant")

    assert ".NET" in score.matched
    assert "C#" in score.matched


def test_keyword_in_company_description_is_not_fake_lift() -> None:
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        experiences=[
            FactsExperience(
                role="Engineer",
                company="Acme",
                company_description="A fintech leader",
                start_date="2020",
                end_date="2022",
                bullets=["shipped work"],
            )
        ],
    )
    jd = _jd(["fintech"])
    cv = CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=Skills(),
        experiences=[
            Experience(
                role="Engineer",
                company="Acme",
                company_description="A fintech leader",
                start_date="2020",
                end_date="2022",
                bullets=[BulletPoint(action_verb="Developed", description="shipped work")],
            )
        ],
    )

    score = score_ats(cv, jd, original_cv_text=_facts_to_text(facts))

    assert score.before_pct == score.after_pct  # present on both sides → no fake lift


def test_keyword_only_in_bullet_skills_tag_is_not_counted() -> None:
    jd = _jd(["Kubernetes"])
    cv = CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=Skills(),
        experiences=[
            Experience(
                role="Engineer",
                company="Acme",
                company_description="A cloud platform",
                start_date="2020",
                end_date="2022",
                bullets=[
                    BulletPoint(
                        action_verb="Built",
                        description="internal services",  # prose: no JD keyword
                        skills=["Kubernetes"],  # structured tag only
                    )
                ],
            )
        ],
    )

    score = score_ats(cv, jd, original_cv_text="nothing relevant")

    assert "Kubernetes" in score.missing  # tag-only keyword is not part of the rendered basis
    assert score.after_pct == 0.0


def test_facts_to_text_tolerates_null_optional_fields() -> None:
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        projects=[FactsProject(name="Atlas")],  # description=None
        certificates=[FactsCertificate(title="AWS SAA")],  # issuer=None
        languages=[FactsLanguage(language="English")],  # level=None
    )

    before = _facts_to_text(facts)  # must not raise TypeError

    assert "atlas" in before.lower()  # the named entry still serializes
    assert "english" in before.lower()
    assert "None" not in before  # a null field contributes "", never the literal "None"


def test_cv_to_text_omits_null_optional_fields_without_none_literal() -> None:
    cv = CVTemplate(
        personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        summary=Summary(text="a\nb\nc"),
        skills=Skills(),
        projects=[Project(name="Atlas")],  # description=None
        certificates=[Certificate(title="AWS SAA")],  # issuer=None
        languages=[Language(language="English")],  # level=None
    )

    text = _cv_to_text(cv)

    assert "atlas" in text.lower()
    assert "None" not in text  # null fields must not pollute the scored text


def test_coverage_flags_below_target_with_no_missing_emits_single_flag() -> None:
    ats = AtsScore(
        before_pct=10.0,
        after_pct=float(COVERAGE_TARGET_PCT - 5),  # below target
        matched=["Python"],
        missing=[],  # nothing honestly omitted
    )

    flags = _coverage_flags(ats)

    assert len(flags) == 1  # the coverage line only — no empty "Missing:" entry
    assert flags[0].kind is FlagKind.DID_NOT_CONVERGE
    assert not any(f.kind is FlagKind.UNMET_COVERAGE for f in flags)
    assert "Missing" not in flags[0].message


def test_coverage_flags_at_target_emits_nothing() -> None:
    ats = AtsScore(before_pct=50.0, after_pct=float(COVERAGE_TARGET_PCT), matched=[], missing=[])

    assert _coverage_flags(ats) == []  # at/above target → no omission panel


@pytest.mark.parametrize(("before", "after"), [(-1.0, 50.0), (50.0, 101.0)])
def test_ats_score_rejects_out_of_range_percentages(before: float, after: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 100\]"):
        AtsScore(before_pct=before, after_pct=after)


def test_ats_score_accepts_full_coverage_bounds() -> None:
    ats = AtsScore(before_pct=0.0, after_pct=100.0)  # both bounds inclusive

    assert ats.coverage_pct == 100.0


def test_after_pct_identical_with_and_without_experience_dates() -> None:
    jd = _jd(["Python"])

    def _cv_dated(dated: bool) -> CVTemplate:
        return CVTemplate(
            personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
            summary=Summary(text="a\nb\nc"),
            skills=Skills(),
            experiences=[
                Experience(
                    role="Engineer",
                    company="Acme",
                    company_description="A platform",
                    start_date="2020" if dated else None,
                    end_date="2022" if dated else None,
                    bullets=[BulletPoint(action_verb="Built", description="Python services")],
                )
            ],
        )

    with_dates = score_ats(_cv_dated(True), jd, original_cv_text="nothing relevant")
    without_dates = score_ats(_cv_dated(False), jd, original_cv_text="nothing relevant")

    assert with_dates.after_pct == without_dates.after_pct
