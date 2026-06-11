"""Unit tests for the section-wise generators.

The OpenRouter client is the conftest ``mock_llm`` seam — ``call_tool`` returns the
tool-call dict the model would have produced, so the generators are exercised without
any network access. These cover truth-preservation (every generated company/date traces
to source), the color-methodology field surfaces, source-gated pass-through for
projects/certificates/languages, and the truthful carry of ``personal_info``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cv_generator import (
    _carry_personal_info,
    _facts_to_text,
    _generate_certificate,
    _generate_education,
    _generate_experience,
    _generate_language,
    _generate_project,
    _generate_skills,
    _generate_summary,
    _normalize_for_match,
    _term_present,
)
from helprers.cv_template import BulletPoint, PersonalInfo, Summary
from schemas import (
    CandidateLevel,
    CVFacts,
    FactsCertificate,
    FactsEducation,
    FactsExperience,
    FactsLanguage,
    FactsLink,
    FactsPersonalInfo,
    FactsProject,
    JdAnalysis,
    TargetSection,
)


def _facts(**overrides: object) -> CVFacts:
    base = {
        "personal_info": FactsPersonalInfo(
            name="Ada Lovelace",
            location="London, UK",
            email="ada@example.com",
            phone="+44 20 1234",
            links=[FactsLink(title="LinkedIn", url="https://linkedin.com/in/ada")],
        ),
        "experiences": [
            FactsExperience(
                role="Analyst",
                company="Analytical Engines",
                start_date="1840",
                end_date="1843",
                # bullets evidence the JD keywords so the truth-preserving skills filter keeps them
                bullets=["Wrote the first algorithm in Python and FastAPI"],
            ),
            FactsExperience(
                role="Researcher",
                company="Royal Society",
                start_date="1843",
                end_date="1845",
                bullets=["Published notes on Pydantic, PostgreSQL and Docker"],
            ),
        ],
    }
    base.update(overrides)
    return CVFacts(**base)


def _jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="Senior Engineer",
        company="Globex",
        keywords=["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
        requirements_must=["Python", "FastAPI"],
        keyword_plan={"Python": TargetSection.SKILLS, "FastAPI": TargetSection.EXPERIENCE},
        candidate_level=CandidateLevel.SENIOR_IC,
    )


async def test_generated_experience_companies_and_dates_trace_to_source(
    mock_llm: MagicMock,
) -> None:
    facts = _facts()
    source_companies = {e.company for e in facts.experiences}
    source_dates = {(e.company, e.start_date, e.end_date) for e in facts.experiences}
    mock_llm.call_tool.return_value = {
        "experiences": [
            {
                "role": "Senior Analyst",
                "company": "Analytical Engines",
                "company_description": "Pioneering computation lab of the era",
                "start_date": "1840",
                "end_date": "1843",
                "bullets": [
                    {"action_verb": "Developed", "description": "the first published algorithm"}
                ],
            },
            {
                "role": "Lead Researcher",
                "company": "Royal Society",
                "company_description": "The premier scientific society",
                "start_date": "1843",
                "end_date": "1845",
                "bullets": [
                    {"action_verb": "Published", "description": "analytical notes on the engine"}
                ],
            },
        ]
    }

    experiences = await _generate_experience(facts, _jd(), mock_llm)

    assert len(experiences) == 2
    for exp in experiences:
        assert exp.company in source_companies
        assert (exp.company, exp.start_date, exp.end_date) in source_dates


async def test_color_methodology_fields_are_populated(mock_llm: MagicMock) -> None:
    facts = _facts()
    jd = _jd()

    mock_llm.call_tool.return_value = {
        "text": "Senior engineer with deep Python and FastAPI experience.\nProven impact.\nReady.",
        "relevant_skills": ["Python", "FastAPI"],
    }
    summary = await _generate_summary(facts, jd, mock_llm)
    assert isinstance(summary, Summary)
    assert summary.relevant_skills == ["Python", "FastAPI"]  # 🟡

    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "ignored outer header",
                "keywords": [
                    {"keyword": "Python", "tier": "concrete", "category": "Languages"},
                    {"keyword": "FastAPI", "tier": "concrete", "category": "Frameworks"},
                    {"keyword": "PostgreSQL", "tier": "concrete", "category": "Tools"},
                    {"keyword": "Docker", "tier": "concrete", "category": "Tools"},
                ],
            }
        ]
    }
    skills = await _generate_skills(facts, jd, mock_llm)
    all_keywords = [kw for category in skills.categories for kw in category.keywords]
    # every JD keyword the facts evidence surfaces, regrouped under emergent headers (🟡🟠)
    assert set(all_keywords) == {"Python", "FastAPI", "PostgreSQL", "Docker"}
    # the model's per-keyword emergent header drives grouping, in stable first-surfaced order
    assert [c.category for c in skills.categories] == [
        "Languages",
        "Frameworks",
        "Tools",
    ]

    mock_llm.call_tool.return_value = {
        "experiences": [
            {
                "role": "Senior Analyst",
                "company": "Analytical Engines",
                "company_description": "Pioneering computation lab of the era",  # 🟣
                "start_date": "1840",
                "end_date": "1843",
                "bullets": [
                    {
                        "action_verb": "Developed",  # 🔴
                        "description": "the first published algorithm in Python",
                        "skills": ["Python"],  # 🟠
                        "impact": "reduced runtime by 40%",  # 🟢
                        "benefit": "accelerated downstream research",  # 🔵
                    }
                ],
            }
        ]
    }
    experiences = await _generate_experience(facts, jd, mock_llm)
    bullet = experiences[0].bullets[0]
    assert isinstance(bullet, BulletPoint)
    assert str(bullet.action_verb) and bullet.action_verb  # 🔴 populated
    assert bullet.impact and any(c.isdigit() for c in bullet.impact)  # 🟢
    assert bullet.benefit  # 🔵
    assert experiences[0].company_description  # 🟣
    assert bullet.skills == ["Python"]  # 🟠


async def test_passthrough_sections_empty_when_absent_from_source(mock_llm: MagicMock) -> None:
    facts = _facts()  # no projects / certificates / languages

    assert await _generate_project(facts, _jd(), mock_llm) == []
    assert await _generate_certificate(facts, _jd(), mock_llm) == []
    assert await _generate_language(facts, _jd(), mock_llm) == []
    # Never invents content — no provider call for an absent source section.
    assert mock_llm.call_tool.call_count == 0


async def test_passthrough_sections_generated_when_present_in_source(mock_llm: MagicMock) -> None:
    facts = _facts(
        projects=[FactsProject(name="Engine Notes", description="Translation with notes")],
        certificates=[FactsCertificate(title="Mathematics", issuer="Royal Society", year=1842)],
        languages=[FactsLanguage(language="English", level="Native")],
    )

    mock_llm.call_tool.return_value = {
        "projects": [{"name": "Engine Notes", "description": "Translation with analytical notes"}]
    }
    projects = await _generate_project(facts, _jd(), mock_llm)
    assert len(projects) == 1
    assert projects[0].name == "Engine Notes"

    mock_llm.call_tool.return_value = {
        "certificates": [{"title": "Mathematics", "issuer": "Royal Society", "year": 1842}]
    }
    certificates = await _generate_certificate(facts, _jd(), mock_llm)
    assert len(certificates) == 1
    assert certificates[0].title == "Mathematics"

    mock_llm.call_tool.return_value = {
        "languages": [{"language": "English", "level": "Native"}]
    }
    languages = await _generate_language(facts, _jd(), mock_llm)
    assert len(languages) == 1
    assert languages[0].language == "English"


async def test_generate_experience_with_date_less_source_yields_null_dates(
    mock_llm: MagicMock,
) -> None:
    # Non-software, ongoing role: the source carries NO dates at all.
    facts = _facts(
        experiences=[
            FactsExperience(
                role="Head Chef",
                company="Le Bernardin",
                bullets=["Ran a brigade of twenty across two dinner services"],
            )
        ]
    )
    mock_llm.call_tool.return_value = {
        "experiences": [
            {
                "role": "Head Chef",
                "company": "Le Bernardin",
                "bullets": [{"action_verb": "Led", "description": "a brigade of twenty cooks"}],
            }
        ]
    }

    experiences = await _generate_experience(facts, _jd(), mock_llm)

    assert len(experiences) == 1
    assert experiences[0].start_date is None  # absent source date -> null, not fabricated
    assert experiences[0].end_date is None


async def test_generate_language_with_level_less_item_yields_null_level(
    mock_llm: MagicMock,
) -> None:
    facts = _facts(languages=[FactsLanguage(language="English")])  # no stated proficiency
    mock_llm.call_tool.return_value = {"languages": [{"language": "English"}]}

    languages = await _generate_language(facts, _jd(), mock_llm)

    assert len(languages) == 1
    assert languages[0].language == "English"
    assert languages[0].level is None  # honest absence, not a fabricated proficiency


def test_personal_info_carried_truthfully_from_facts() -> None:
    facts = _facts()

    personal_info = _carry_personal_info(facts)

    assert isinstance(personal_info, PersonalInfo)
    assert personal_info.name == "Ada Lovelace"
    assert personal_info.email == "ada@example.com"
    assert personal_info.location == "London, UK"
    assert personal_info.phone == "+44 20 1234"
    assert personal_info.links[0].url == "https://linkedin.com/in/ada"


async def test_generate_education_builds_from_source(mock_llm: MagicMock) -> None:
    facts = _facts(education=[FactsEducation(institution="MIT", degree="CS")])
    mock_llm.call_tool.return_value = {
        "education": [
            {
                "institution": "MIT",
                "degree": "BSc Computer Science",
                "start_year": 1837,
                "end_year": 1841,
                "gpa": "4.0",
            }
        ]
    }

    education = await _generate_education(facts, _jd(), mock_llm)

    assert len(education) == 1
    assert education[0].institution == "MIT"
    assert education[0].gpa == "4.0"


async def test_generate_project_builds_link_when_present(mock_llm: MagicMock) -> None:
    facts = _facts(projects=[FactsProject(name="Engine Notes", description="d")])
    mock_llm.call_tool.return_value = {
        "projects": [
            {
                "name": "Engine Notes",
                "description": "d",
                "link": {"title": "repo", "url": "https://git.example/x"},
            }
        ]
    }

    projects = await _generate_project(facts, _jd(), mock_llm)

    assert projects[0].link is not None
    assert projects[0].link.url == "https://git.example/x"


async def test_skills_string_not_iterated_per_character(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.return_value = {
        "categories": [
            {"category": "Languages", "keywords": "Python"},  # bare string, not a list
            {"category": "Tools", "keywords": ["Docker"]},
        ]
    }

    skills = await _generate_skills(_facts(), _jd(), mock_llm)

    all_keywords = [kw for category in skills.categories for kw in category.keywords]
    # the bare-string keyword list degraded to empty — NOT ['P', 'y', 't', 'h', 'o', 'n']
    assert all_keywords == ["Docker"]  # only the well-formed keyword list survives


def _sdet_facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"),
        experiences=[
            FactsExperience(
                role="Mobile Engineer",
                company="Appco",
                start_date="2020",
                end_date="2023",
                bullets=[
                    "Built a test automation suite in Python with Selenium",
                    "Maintained CI/CD pipelines for release builds",
                    "Shipped iOS apps in Swift and Android apps in Kotlin using Flutter",
                ],
            )
        ],
        languages=[FactsLanguage(language="English", level="Fluent")],
    )


def _sdet_jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="SDET",
        company="AICorp",
        keywords=["Python", "Selenium", "Test Automation", "CI/CD", "Machine Learning"],
        requirements_must=["Python"],
        candidate_level=CandidateLevel.MID,
    )


async def test_skills_excludes_role_irrelevant_noise_and_unevidenced_gaps(
    mock_llm: MagicMock,
) -> None:
    facts = _sdet_facts()
    mock_llm.call_tool.return_value = {
        "categories": [
            {"category": "Languages", "keywords": ["Python", "Swift", "Kotlin"]},
            {
                "category": "Testing",
                "keywords": ["Selenium", "Test Automation", "Mobile Automation"],
            },
            {"category": "Practices", "keywords": ["CI/CD", "Machine Learning"]},
        ]
    }

    skills = await _generate_skills(facts, _sdet_jd(), mock_llm)
    all_keywords = [kw for category in skills.categories for kw in category.keywords]

    # evidenced AI/SDET keywords surface
    for kept in ("Python", "Selenium", "Test Automation", "CI/CD"):
        assert kept in all_keywords
    # role-irrelevant mobile noise (not named by the JD) is dropped
    for noise in ("Swift", "Kotlin", "Flutter", "Mobile Automation"):
        assert noise not in all_keywords
    # a JD keyword with NO CV evidence stays omitted (true gap, never fabricated)
    assert "Machine Learning" not in all_keywords
    # invariant: zero rendered skill keyword is absent from CV evidence
    evidence = _normalize_for_match(_facts_to_text(facts))
    skill_keywords = [
        kw
        for category in skills.categories
        if category.category != "Spoken Languages"
        for kw in category.keywords
    ]
    for keyword in skill_keywords:
        assert _term_present(keyword, evidence), f"{keyword} is not evidenced in the CV facts"


async def test_skills_appends_spoken_languages_from_facts(mock_llm: MagicMock) -> None:
    facts = _sdet_facts()  # languages: English / Fluent
    mock_llm.call_tool.return_value = {
        "categories": [{"category": "Languages", "keywords": ["Python"]}]
    }

    skills = await _generate_skills(facts, _sdet_jd(), mock_llm)

    spoken = [c for c in skills.categories if c.category == "Spoken Languages"]
    assert spoken, "Spoken Languages category is missing"
    assert any("English" in kw for kw in spoken[0].keywords)


def _mobile_qa_facts() -> CVFacts:
    """A mobile/QA candidate who architected test infra and ran CI/CD — no AWS/Kafka/K8s."""
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"),
        experiences=[
            FactsExperience(
                role="QA Engineer",
                company="Appco",
                start_date="2019",
                end_date="2023",
                bullets=[
                    "Architected the test automation infrastructure across three mobile teams",
                    "Built and maintained CI/CD pipelines that run autotests on every commit",
                    "Wrote UI and API tests in Python with Selenium and Appium",
                ],
            )
        ],
        languages=[FactsLanguage(language="English", level="Fluent")],
    )


def _ai_sdet_jd() -> JdAnalysis:
    """AI/SDET JD: concrete techs the CV lacks + competencies the CV evidences differently."""
    return JdAnalysis(
        role_title="AI SDET",
        company="AICorp",
        keywords=[
            "Python",
            "Selenium",
            "AWS",
            "Kafka",
            "Kubernetes",
            "Systems Thinking",
            "SDLC Automation",
            "Prototyping",
            "Production Monitoring",
        ],
        keyword_tiers={
            "Python": "concrete",
            "Selenium": "concrete",
            "AWS": "concrete",
            "Kafka": "concrete",
            "Kubernetes": "concrete",
            "Systems Thinking": "competency",
            "SDLC Automation": "competency",
            "Prototyping": "competency",
            "Production Monitoring": "competency",
        },
        requirements_must=["Python"],
        candidate_level=CandidateLevel.MID,
    )


async def test_skills_surfaces_competency_only_with_traceable_anchor(mock_llm: MagicMock) -> None:
    facts = _mobile_qa_facts()
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "Languages & Tools",
                "keywords": [
                    {"keyword": "Python", "tier": "concrete"},
                    {"keyword": "Selenium", "tier": "concrete"},
                    # concrete techs the CV does NOT contain — must not surface
                    {"keyword": "AWS", "tier": "concrete"},
                    {"keyword": "Kafka", "tier": "concrete"},
                    {"keyword": "Kubernetes", "tier": "concrete"},
                ],
            },
            {
                "category": "Engineering Practices",
                "keywords": [
                    {
                        "keyword": "Systems Thinking",
                        "tier": "competency",
                        "anchor_ref": "Architected the test automation infrastructure across three mobile teams",
                    },
                    {
                        "keyword": "SDLC Automation",
                        "tier": "competency",
                        "anchor_ref": "Built and maintained CI/CD pipelines that run autotests on every commit",
                    },
                    # competency the CV does NOT evidence — bogus anchor not in the source
                    {
                        "keyword": "Prototyping",
                        "tier": "competency",
                        "anchor_ref": "Designed and shipped rapid product prototypes",
                    },
                    # competency emitted with NO anchor — must never reach output
                    {"keyword": "Production Monitoring", "tier": "competency"},
                ],
            },
        ]
    }

    skills = await _generate_skills(facts, _ai_sdet_jd(), mock_llm)
    rendered = [kw for category in skills.categories for kw in category.keywords]

    # concrete literally present + anchored competencies surface
    for kept in ("Python", "Selenium", "Systems Thinking", "SDLC Automation"):
        assert kept in rendered, f"{kept} should surface"
    # concrete techs absent from the CV never surface (the interview-failure case)
    for gap in ("AWS", "Kafka", "Kubernetes"):
        assert gap not in rendered, f"{gap} must not surface (not literal in CV)"
    # competency with an untraceable anchor, and one with no anchor, are dropped
    assert "Prototyping" not in rendered
    assert "Production Monitoring" not in rendered

    # provenance accompanies every surfaced skill keyword; competency anchors trace to source
    facts_text = _normalize_for_match(_facts_to_text(facts))
    prov_by_kw = {p.keyword: p for p in skills.provenance}
    assert prov_by_kw["Systems Thinking"].anchor_ref in (
        "Architected the test automation infrastructure across three mobile teams",
    )
    for kw in ("Systems Thinking", "SDLC Automation"):
        anchor = prov_by_kw[kw].anchor_ref
        assert anchor and anchor.lower() in facts_text.lower()
    # concrete provenance carries no anchor
    assert prov_by_kw["Python"].anchor_ref is None


async def test_skills_provenance_is_not_rendered_or_in_keywords(mock_llm: MagicMock) -> None:
    facts = _mobile_qa_facts()
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "Practices",
                "keywords": [
                    {
                        "keyword": "SDLC Automation",
                        "tier": "competency",
                        "anchor_ref": "Built and maintained CI/CD pipelines that run autotests on every commit",
                    }
                ],
            }
        ]
    }

    skills = await _generate_skills(facts, _ai_sdet_jd(), mock_llm)
    # the rendered keyword list is plain strings — never the tier/anchor provenance dicts
    for category in skills.categories:
        for kw in category.keywords:
            assert isinstance(kw, str)
    assert "SDLC Automation" in [kw for c in skills.categories for kw in c.keywords]


def _testing_jd() -> JdAnalysis:
    """An SDET JD whose four testing keywords are all CONCRETE and literally evidenced."""
    return JdAnalysis(
        role_title="SDET",
        company="AICorp",
        keywords=["SDET", "Test Automation", "Regression Testing", "Load Testing"],
        keyword_tiers={
            "SDET": "concrete",
            "Test Automation": "concrete",
            "Regression Testing": "concrete",
            "Load Testing": "concrete",
        },
        requirements_must=["SDET"],
        candidate_level=CandidateLevel.MID,
    )


def _testing_facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"),
        experiences=[
            FactsExperience(
                role="QA Engineer",
                company="Appco",
                start_date="2019",
                end_date="2023",
                bullets=[
                    "Worked as an SDET building Test Automation across teams",
                    "Owned Regression Testing suites and Load Testing pipelines",
                ],
            )
        ],
    )


async def test_testing_keywords_aggregate_under_shared_emergent_header(mock_llm: MagicMock) -> None:
    # The model splits the four testing keywords across two OUTER categories but tags each with the
    # SAME emergent per-keyword header ("Testing"); they must aggregate under that ONE header.
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer A",
                "keywords": [
                    {"keyword": "Test Automation", "tier": "concrete", "category": "Testing"},
                    {"keyword": "SDET", "tier": "concrete", "category": "Testing"},
                ],
            },
            {
                "category": "outer B",
                "keywords": [
                    {"keyword": "Regression Testing", "tier": "concrete", "category": "Testing"},
                    {"keyword": "Load Testing", "tier": "concrete", "category": "Testing"},
                ],
            },
        ]
    }
    skills = await _generate_skills(_testing_facts(), _testing_jd(), mock_llm)
    testing = [c for c in skills.categories if c.category == "Testing"]
    assert len(testing) == 1, "keywords sharing an emergent header aggregate into ONE group"
    assert set(testing[0].keywords) == {
        "SDET",
        "Test Automation",
        "Regression Testing",
        "Load Testing",
    }


async def test_anchored_competencies_keep_their_emergent_header(mock_llm: MagicMock) -> None:
    # No closed taxonomy: anchored competencies render under the model's OWN emergent header,
    # verbatim (format-valid) — never coerced into a hardcoded "Practices & Concepts" bucket.
    facts = _mobile_qa_facts()
    infra_anchor = "Architected the test automation infrastructure across three mobile teams"
    cicd_anchor = "Built and maintained CI/CD pipelines that run autotests on every commit"
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer",
                "keywords": [
                    {
                        "keyword": "Systems Thinking",
                        "tier": "competency",
                        "category": "Engineering",
                        "anchor_ref": infra_anchor,
                    },
                    {
                        "keyword": "SDLC Automation",
                        "tier": "competency",
                        "category": "Engineering",
                        "anchor_ref": cicd_anchor,
                    },
                ],
            }
        ]
    }
    skills = await _generate_skills(facts, _ai_sdet_jd(), mock_llm)
    engineering = [c for c in skills.categories if c.category == "Engineering"]
    assert len(engineering) == 1
    assert set(engineering[0].keywords) == {"Systems Thinking", "SDLC Automation"}
    # no closed-set header leaked in
    assert "Practices & Concepts" not in [c.category for c in skills.categories]


def _nurse_facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Pat Carer", email="pat@example.com"),
        experiences=[
            FactsExperience(
                role="Registered Nurse",
                company="St. Mary's Hospital",
                start_date="2018",
                end_date="2024",
                bullets=[
                    "Charted patient vitals in Epic EHR across a 30-bed medical-surgical unit",
                    "Performed triage and wound care during high-acuity shifts",
                    "Maintained ACLS certification and led code-blue responses",
                ],
            )
        ],
        languages=[FactsLanguage(language="Spanish", level="Fluent")],
    )


def _nurse_jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="ICU Nurse",
        company="General Hospital",
        keywords=["Epic EHR", "Triage", "Wound Care", "ACLS", "Telemetry"],
        keyword_tiers={
            "Epic EHR": "concrete",
            "Triage": "competency",
            "Wound Care": "competency",
            "ACLS": "concrete",
            "Telemetry": "concrete",  # a real clinical tech the CV does NOT mention → gap
        },
        requirements_must=["Epic EHR"],
        candidate_level=CandidateLevel.MID,
    )


async def test_nursing_cv_renders_field_appropriate_headers_no_software_words(
    mock_llm: MagicMock,
) -> None:
    facts = _nurse_facts()
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer",
                "keywords": [
                    {"keyword": "Epic EHR", "tier": "concrete", "category": "Clinical Systems"},
                    {"keyword": "ACLS", "tier": "concrete", "category": "Certifications"},
                    {
                        "keyword": "Triage",
                        "tier": "competency",
                        "category": "Patient Care",
                        "anchor_ref": "Performed triage and wound care during high-acuity shifts",
                    },
                    {
                        "keyword": "Wound Care",
                        "tier": "competency",
                        "category": "Patient Care",
                        "anchor_ref": "Performed triage and wound care during high-acuity shifts",
                    },
                    # a concrete clinical tech NOT literally in the CV → must not surface
                    {"keyword": "Telemetry", "tier": "concrete", "category": "Clinical Systems"},
                ],
            }
        ]
    }
    skills = await _generate_skills(facts, _nurse_jd(), mock_llm)
    headers = [c.category for c in skills.categories]
    rendered = [kw for c in skills.categories for kw in c.keywords]

    # field-appropriate emergent headers — derived from the nursing material
    assert "Clinical Systems" in headers
    assert "Patient Care" in headers
    assert "Certifications" in headers
    # ZERO forced software vocabulary in any header (the closed taxonomy is gone)
    for software_word in (
        "Languages",
        "Frameworks & Libraries",
        "Tools & Platforms",
        "Testing & QA",
        "Practices & Concepts",
    ):
        assert software_word not in headers
    # concrete clinical terms surface only when literally in the CV; the gap stays missing
    assert "Epic EHR" in rendered and "ACLS" in rendered
    assert "Triage" in rendered and "Wound Care" in rendered
    assert "Telemetry" not in rendered, "a concrete tech absent from the CV must stay a gap"
    # spoken language is facts-sourced and appended last
    assert skills.categories[-1].category == "Spoken Languages"


def _format_jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="Engineer",
        company="Co",
        keywords=["Python"],
        keyword_tiers={"Python": "concrete"},
        requirements_must=["Python"],
        candidate_level=CandidateLevel.MID,
    )


def _python_facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"),
        experiences=[
            FactsExperience(
                role="Engineer",
                company="Appco",
                start_date="2020",
                end_date="2023",
                bullets=["Built Python services in production"],
            )
        ],
    )


async def _resolve_skills(facts: CVFacts, jd: JdAnalysis, llm: MagicMock):
    """Resolve ONLY the skills section through the real regen loop + header normalization.

    Mirrors what ``_generate_all_sections`` does for skills (generate → validate → regen to cap →
    deterministic header normalization), but in isolation so a single skills mock drives it
    without the other sections demanding their own tool-call payloads.
    """
    from cv_generator import (
        _generate_skills,
        _normalize_skill_headers,
        _resolve_section,
        validate_section,
    )

    obj, flag = await _resolve_section(
        TargetSection.SKILLS,
        lambda: _generate_skills(facts, jd, llm),
        lambda o: validate_section(TargetSection.SKILLS, o, facts, jd),
    )
    _normalize_skill_headers(obj)
    return obj, flag


@pytest.mark.parametrize(
    ("bad_label", "expected_header"),
    [
        # > 3 words / > 24 chars → capped to the first 3 words (single concept, no joiners)
        ("An Extraordinarily Long Editorialized Header Of Doom", "An Extraordinarily Long"),
        # comma + multi-conjunction → keep the GREATEST-word-count segment ("AI Integration")
        ("CI/CD, Testing & AI Integration", "AI Integration"),
        # multi "and" + > 3 words → split on "and"; all 1-word segments → FIRST ("Build")
        ("Build and Ship and Deploy", "Build"),
        # single "&" is ATOMIC, but > 3 words → cap drops the trailing word + dangling "&"
        ("Quality Infrastructure & Testing", "Quality Infrastructure"),
    ],
)
async def test_malformed_header_normalized_by_construction_no_regen(
    mock_llm: MagicMock, bad_label: str, expected_header: str
) -> None:
    # A malformed header is NORMALIZED at assembly, never a regen trigger: the section validates
    # and resolves in ONE call (no header-driven retries), and the rendered header is the clean
    # single-concept label — no comma, no "&"/"/" joiner, no markup, <= 3 words.
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer",
                "keywords": [{"keyword": "Python", "tier": "concrete", "category": bad_label}],
            }
        ]
    }
    skills, flag = await _resolve_skills(_python_facts(), _format_jd(), mock_llm)
    rendered = [kw for c in skills.categories for kw in c.keywords]
    headers = [c.category for c in skills.categories if c.keywords]

    assert mock_llm.call_tool.call_count == 1  # NO header regen — invariant by construction
    assert flag is None
    assert "Python" in rendered  # the keyword still surfaces
    assert headers == [expected_header]
    for header in headers:
        assert "," not in header and "&" not in header and "/" not in header
        assert "*" not in header and "_" not in header
        assert len(header.split()) <= 3 and len(header) <= 24


async def test_markup_only_header_renders_flat_others_keep_clean_header(
    mock_llm: MagicMock,
) -> None:
    # DEGENERATE case: a markup-only header normalizes to empty → that ONE category renders flat
    # (no header), while a sibling category with a normal atomic header still keeps its header.
    # No inconsistent mix beyond the single intentional flat group; no internal diagnostic.
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer",
                "keywords": [
                    {"keyword": "Python", "tier": "concrete", "category": "***"},
                    {"keyword": "Docker", "tier": "concrete", "category": "Tooling & Platforms"},
                ],
            }
        ]
    }
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"),
        experiences=[
            FactsExperience(
                role="Engineer",
                company="Appco",
                start_date="2020",
                end_date="2023",
                bullets=["Built Python services with Docker in production"],
            )
        ],
    )
    jd = JdAnalysis(
        role_title="Engineer",
        company="Co",
        keywords=["Python", "Docker"],
        keyword_tiers={"Python": "concrete", "Docker": "concrete"},
        requirements_must=["Python"],
        candidate_level=CandidateLevel.MID,
    )
    skills, flag = await _resolve_skills(facts, jd, mock_llm)
    by_kw = {kw: c.category for c in skills.categories for kw in c.keywords}
    assert by_kw["Python"] == ""  # markup-only header → flat (sole headerless case)
    assert by_kw["Docker"] == "Tooling & Platforms"  # single-"&" atomic header kept whole
    assert flag is None


async def test_emergent_header_with_markdown_renders_plain(mock_llm: MagicMock) -> None:
    # plain-text invariant: a model-emitted header wrapped in markdown is stripped to plain text
    # before it is stored, so the rendered AND copied Skills header carries ZERO markup.
    mock_llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer",
                "keywords": [
                    {"keyword": "Python", "tier": "concrete", "category": "**Languages**"}
                ],
            }
        ]
    }
    skills = await _generate_skills(_python_facts(), _format_jd(), mock_llm)
    headers = [c.category for c in skills.categories]
    assert "Languages" in headers
    for category in skills.categories:
        assert "*" not in category.category and "_" not in category.category


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
