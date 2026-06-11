"""Unit tests for the extract-pass Pydantic schemas (schemas.py).

Covers CVFacts (+ sub-models), JdAnalysis (+ to_job_target), and ExtractResult.
These are pure validation tests — no LLM or I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from helprers.cv_template import JobTarget
from schemas import (
    CandidateLevel,
    CVFacts,
    ExtractResult,
    FactsExperience,
    FactsPersonalInfo,
    FactsSkillGroup,
    JdAnalysis,
    KeywordTier,
    TargetSection,
)

def _full_cv_facts_payload() -> dict:
    """A fully-populated CVFacts payload exercising every sub-model."""
    return {
        "personal_info": {
            "name": "Ada Lovelace",
            "location": "London, UK",
            "email": "ada@example.com",
            "phone": "+44 20 7946 0958",
            "links": [{"title": "LinkedIn", "url": "https://linkedin.com/in/ada"}],
        },
        "experiences": [
            {
                "role": "Analyst",
                "company": "Analytical Engines",
                "company_description": "Pioneering computation firm",
                "start_date": "2020-01",
                "end_date": "Present",
                "location": "London",
                "bullets": ["Wrote the first algorithm", "Designed loop notation"],
            }
        ],
        "education": [
            {
                "institution": "Self-taught",
                "degree": "Mathematics",
                "start_year": 1830,
                "end_year": 1835,
                "gpa": "N/A",
            }
        ],
        "projects": [
            {
                "name": "Note G",
                "description": "Bernoulli number algorithm",
                "skills": ["Mathematics"],
                "link": {"title": "Notes", "url": "https://example.com/notes"},
            }
        ],
        "certificates": [
            {
                "title": "Honorary Fellow",
                "issuer": "Royal Society",
                "year": 1843,
                "link": {"title": "Cert", "url": "https://example.com/cert"},
            }
        ],
        "languages": [{"language": "English", "level": "Native"}],
        "skills": [
            {"category": "Mathematics", "items": ["Calculus", "Number theory"]},
            {"category": "Computing", "items": ["Algorithm design"]},
        ],
    }


def test_cv_facts_validates_with_full_data() -> None:
    facts = CVFacts.model_validate(_full_cv_facts_payload())

    assert facts.personal_info.name == "Ada Lovelace"
    assert facts.personal_info.links[0].title == "LinkedIn"
    assert facts.experiences[0].bullets == [
        "Wrote the first algorithm",
        "Designed loop notation",
    ]
    assert facts.education[0].end_year == 1835
    assert facts.projects[0].link is not None
    assert facts.certificates[0].issuer == "Royal Society"
    assert facts.languages[0].level == "Native"
    assert facts.skills[0].category == "Mathematics"
    assert facts.skills[0].items == ["Calculus", "Number theory"]
    assert facts.skills[1].items == ["Algorithm design"]


def test_cv_facts_skills_capture_groups_verbatim_category_optional() -> None:
    """Skill groups are captured verbatim; category is optional; identity = >=1 non-blank item."""
    facts = CVFacts.model_validate(
        {
            "personal_info": {
                "name": "Dev",
                "location": "Berlin",
                "email": "dev@example.com",
            },
            "skills": [
                {"items": ["Welding", "Blueprint reading"]},  # no category — a flat list
                {"category": "Languages", "items": ["German"]},
                {"category": "Empty", "items": ["", "  "]},  # all-blank → dropped (hollow)
                {"category": "Garbage"},  # no items → dropped
            ],
        }
    )

    assert facts.skills[0].category is None
    assert facts.skills[0].items == ["Welding", "Blueprint reading"]
    assert [g.category for g in facts.skills] == [None, "Languages"]


def test_factsskillgroup_drops_non_string_items() -> None:
    group = FactsSkillGroup.model_validate({"category": "Mix", "items": ["ok", None, 3, "  fine "]})
    assert group.items == ["ok", "fine"]


def test_cv_facts_tolerates_missing_optional_fields_lists_default_empty() -> None:
    minimal = {
        "personal_info": {
            "name": "Grace Hopper",
            "location": "New York, USA",
            "email": "grace@example.com",
        }
    }

    facts = CVFacts.model_validate(minimal)

    # All collection fields default to empty lists when omitted.
    assert facts.experiences == []
    assert facts.education == []
    assert facts.projects == []
    assert facts.certificates == []
    assert facts.languages == []
    assert facts.skills == []
    # Optional scalar/link fields tolerate absence.
    assert facts.personal_info.phone is None
    assert facts.personal_info.links == []


def test_cv_facts_requires_personal_info() -> None:
    with pytest.raises(ValidationError):
        CVFacts.model_validate({"experiences": []})


def test_personal_info_requires_name_location_email() -> None:
    with pytest.raises(ValidationError):
        FactsPersonalInfo.model_validate({"name": "Solo Name"})


def test_experience_dates_accept_free_form_strings() -> None:
    # no enforced YYYY-MM — free-form date strings must be accepted as-is.
    exp = FactsExperience.model_validate(
        {
            "role": "Engineer",
            "company": "Acme",
            "start_date": "Summer 2019",
            "end_date": "Present",
        }
    )

    assert exp.start_date == "Summer 2019"
    assert exp.end_date == "Present"
    assert exp.bullets == []


@pytest.mark.parametrize(
    ("start_date", "end_date", "expected_start", "expected_end"),
    [
        ("Jan 2020", "Present", "Jan 2020", "Present"),
        ("2020", "2021", "2020", "2021"),
        # An empty date string is tolerated (never rejected) and normalized to None by the
        # resilience coercer — a truthful "no date", not a literal empty-string value.
        ("", "", None, None),
    ],
)
def test_experience_dates_accept_arbitrary_free_form_strings(
    start_date: str, end_date: str, expected_start: str | None, expected_end: str | None
) -> None:
    # extract tolerates any date string (month-name, year-only, empty);
    # YYYY-MM/Present normalization is owned by CV generation, not the extract stage.
    exp = FactsExperience.model_validate(
        {
            "role": "Engineer",
            "company": "Acme",
            "start_date": start_date,
            "end_date": end_date,
        }
    )

    assert exp.start_date == expected_start
    assert exp.end_date == expected_end


def test_experience_omitting_date_and_location_keys_parses_to_none() -> None:
    cv = CVFacts.model_validate(
        {
            "personal_info": {"name": "Mercy Otieno", "email": "mercy@example.com"},
            "experiences": [
                {
                    "role": "Registered Nurse",
                    "company": "St. Mary's Hospital",
                    "start_date": "2019",
                    "bullets": ["Triaged patients across a 40-bed emergency ward"],
                }
            ],
        }
    )

    exp = cv.experiences[0]
    assert exp.start_date == "2019"  # a present date is kept verbatim
    assert exp.end_date is None  # omitted end_date -> truthful absence, never fabricated
    assert exp.location is None  # omitted location header -> None, not an invented city


def test_jd_analysis_omitting_company_parses_to_none() -> None:
    jd = JdAnalysis.model_validate({"keywords": ["Python"], "candidate_level": "mid"})

    assert jd.company is None  # a JD naming no company is not fabricated
    assert jd.role_title is None


def _jd_payload() -> dict:
    return {
        "role_title": "Senior Backend Engineer",
        "company": "Globex",
        "keywords": ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
        "requirements_must": ["5y Python"],
        "requirements_nice": ["GraphQL"],
        "keyword_plan": {"Python": "skills", "FastAPI": "experience"},
        "candidate_level": "senior_ic",
    }


def test_jd_analysis_validates_required_fields() -> None:
    jd = JdAnalysis.model_validate(_jd_payload())

    assert jd.role_title == "Senior Backend Engineer"
    assert jd.company == "Globex"
    assert jd.keywords == ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"]
    assert jd.candidate_level is CandidateLevel.SENIOR_IC
    assert jd.keyword_plan["Python"] is TargetSection.SKILLS
    assert jd.keyword_plan["FastAPI"] is TargetSection.EXPERIENCE


def test_jd_analysis_defaults_optional_collections() -> None:
    jd = JdAnalysis.model_validate(
        {
            "role_title": "Engineer",
            "company": "Acme",
            "keywords": ["Python"],
            "candidate_level": "mid",
        }
    )

    assert jd.requirements_must == []
    assert jd.requirements_nice == []
    assert jd.keyword_plan == {}


def test_jd_analysis_defaults_unknown_candidate_level() -> None:
    payload = _jd_payload()
    payload["candidate_level"] = "staff"  # real seniority outside the 6-member enum

    jd = JdAnalysis.model_validate(payload)

    assert jd.candidate_level is CandidateLevel.MID


def test_jd_analysis_drops_unknown_keyword_plan_section() -> None:
    payload = _jd_payload()
    payload["keyword_plan"] = {"Python": "skills", "Weird": "not_a_section"}

    jd = JdAnalysis.model_validate(payload)

    # The valid entry is kept; the out-of-vocab section is dropped, not fatal.
    assert jd.keyword_plan == {"Python": TargetSection.SKILLS}


def test_jd_analysis_tags_keywords_concrete_or_competency() -> None:
    payload = _jd_payload()
    payload["keywords"] = [
        "AWS",
        "Kafka",
        "Kubernetes",
        "React",
        "Systems Thinking",
        "SDLC Automation",
        "Prototyping",
        "Production Monitoring",
    ]
    payload["keyword_tiers"] = {
        "AWS": "concrete",
        "Kafka": "concrete",
        "Kubernetes": "concrete",
        "React": "concrete",
        "Systems Thinking": "competency",
        "SDLC Automation": "competency",
        "Prototyping": "competency",
        "Production Monitoring": "competency",
    }

    jd = JdAnalysis.model_validate(payload)

    for concrete in ("AWS", "Kafka", "Kubernetes", "React"):
        assert jd.tier_of(concrete) is KeywordTier.CONCRETE
    for competency in (
        "Systems Thinking",
        "SDLC Automation",
        "Prototyping",
        "Production Monitoring",
    ):
        assert jd.tier_of(competency) is KeywordTier.COMPETENCY


def test_jd_tier_of_defaults_concrete_and_is_case_insensitive() -> None:
    payload = _jd_payload()
    payload["keyword_tiers"] = {"Systems Thinking": "competency"}
    jd = JdAnalysis.model_validate(payload)

    # case-insensitive lookup of a tagged competency
    assert jd.tier_of("systems thinking") is KeywordTier.COMPETENCY
    # an untagged keyword defaults to CONCRETE (literal evidence required)
    assert jd.tier_of("Python") is KeywordTier.CONCRETE


def test_jd_analysis_drops_unknown_keyword_tier() -> None:
    payload = _jd_payload()
    payload["keyword_tiers"] = {"Python": "concrete", "FastAPI": "not_a_tier"}

    jd = JdAnalysis.model_validate(payload)

    # the valid tier is kept; the out-of-vocab tier is dropped (defaults to CONCRETE), not fatal
    assert jd.keyword_tiers == {"Python": KeywordTier.CONCRETE}
    assert jd.tier_of("FastAPI") is KeywordTier.CONCRETE


def test_jd_to_job_target_returns_job_target() -> None:
    jd = JdAnalysis.model_validate(_jd_payload())

    target = jd.to_job_target()

    assert isinstance(target, JobTarget)
    assert target.title == "Senior Backend Engineer"
    assert target.company == "Globex"
    assert target.keywords == ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"]


def test_extract_result_requires_facts_and_jd() -> None:
    with pytest.raises(ValidationError):
        ExtractResult.model_validate({"flags": []})


def test_extract_result_flags_default_empty() -> None:
    result = ExtractResult(
        facts=CVFacts.model_validate(
            {
                "personal_info": {
                    "name": "Linus",
                    "location": "Portland, USA",
                    "email": "linus@example.com",
                }
            }
        ),
        jd=JdAnalysis.model_validate(
            {
                "role_title": "Engineer",
                "company": "Acme",
                "keywords": ["Linux"],
                "candidate_level": "senior_ic",
            }
        ),
    )

    assert result.flags == []


def test_extract_result_accepts_flags() -> None:
    result = ExtractResult(
        facts=CVFacts.model_validate(
            {
                "personal_info": {
                    "name": "Linus",
                    "location": "Portland, USA",
                    "email": "linus@example.com",
                }
            }
        ),
        jd=JdAnalysis.model_validate(
            {
                "role_title": "Engineer",
                "company": "Acme",
                "keywords": ["Linux"],
                "candidate_level": "senior_ic",
            }
        ),
        flags=["keyword-gap: only 1 of 5 keywords"],
    )

    assert result.flags == ["keyword-gap: only 1 of 5 keywords"]
