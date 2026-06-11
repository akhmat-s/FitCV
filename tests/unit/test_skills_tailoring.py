"""Skills TAILORING tests — the candidate's real skills are tailored, not synthesized.

Reproduces the root-cause bug two ways (field-agnostic): the generator must surface the
candidate's DECLARED skills (``facts.skills``) — reframed/foregrounded toward the JD — instead
of rebuilding Skills from ``JD_keywords ∩ evidence`` and discarding everything that is not a
literal JD keyword. Two fixtures (a software CV and a nursing CV) prove the fix holds from the
SAME code with no hardcoded taxonomy. Field-specific tokens (Python / triage / ACLS …) live
ONLY in these fixtures, never in tool logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from cv_generator import (
    _facts_to_text,
    _generate_skills,
    _normalize_for_match,
    _term_present,
)
from schemas import (
    CandidateLevel,
    CVFacts,
    FactsExperience,
    FactsLanguage,
    FactsPersonalInfo,
    FactsSkillGroup,
    JdAnalysis,
)

# --- SOFTWARE fixture (the real reproduction) ---------------------------------


def _automation_lead_facts() -> CVFacts:
    """Automation-lead CV that DECLARES its skills in a dedicated Skills section."""
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Lead Dev", email="lead@example.com"),
        experiences=[
            FactsExperience(
                role="Automation Lead",
                company="Shipfast",
                start_date="2019",
                end_date="2024",
                bullets=[
                    "Built test automation and CI/CD across three product teams",
                    "Drove LLM evaluation harnesses for release gating",
                ],
            )
        ],
        skills=[
            FactsSkillGroup(
                category="AI & LLM",
                items=["LLM integration", "fine-tuning", "LLM evaluation", "AI tooling adoption"],
            ),
            FactsSkillGroup(
                category="QA & Automation",
                items=["test automation", "CI/CD", "release automation", "load testing"],
            ),
            FactsSkillGroup(
                category="Programming",
                items=["Python", "Dart", "Swift", "Kotlin", "Java", "C++"],
            ),
            FactsSkillGroup(category="Leadership", items=["team management", "hiring & mentoring"]),
        ],
        languages=[FactsLanguage(language="English", level="Fluent")],
    )


def _ai_qa_jd() -> JdAnalysis:
    """AI-native QA / agentic JD: names some of the candidate's skills + things they lack."""
    return JdAnalysis(
        role_title="AI-native QA Engineer",
        company="AICorp",
        keywords=[
            "test automation",
            "CI/CD",
            "Python",
            "LLM evaluation",
            "agentic workflows",
            "cloud infrastructure",
        ],
        keyword_tiers={
            "test automation": "concrete",
            "CI/CD": "concrete",
            "Python": "concrete",
            "LLM evaluation": "concrete",
            "agentic workflows": "concrete",
            "cloud infrastructure": "concrete",
        },
        requirements_must=["test automation"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )


async def test_software_real_skills_kept_even_when_not_jd_keywords() -> None:
    """Core: declared skills survive even when they are NOT literal JD keywords."""
    facts = _automation_lead_facts()
    jd = _ai_qa_jd()
    llm = AsyncMock()
    # The generator reframes/foregrounds the candidate's own declared skills under emergent
    # headers, and tries to ADD two concrete JD terms the candidate lacks.
    llm.call_tool.return_value = {
        "categories": [
            {
                "category": "AI & Testing",
                "keywords": [
                    {"keyword": "LLM evaluation", "tier": "concrete", "category": "AI & LLM"},
                    {"keyword": "LLM integration", "tier": "concrete", "category": "AI & LLM"},
                    {"keyword": "fine-tuning", "tier": "concrete", "category": "AI & LLM"},
                    {"keyword": "test automation", "tier": "concrete", "category": "QA"},
                    {"keyword": "CI/CD", "tier": "concrete", "category": "QA"},
                    {"keyword": "Python", "tier": "concrete", "category": "Programming"},
                    {"keyword": "Java", "tier": "concrete", "category": "Programming"},
                    # ADD attempts the candidate does NOT have — must be dropped (no CV evidence)
                    {"keyword": "agentic workflows", "tier": "concrete", "category": "AI & LLM"},
                    {"keyword": "cloud infrastructure", "tier": "concrete", "category": "Infra"},
                ],
            }
        ]
    }

    skills = await _generate_skills(facts, jd, llm)
    rendered = [kw for c in skills.categories for kw in c.keywords]

    # real domain skills are kept — including ones the JD never names (the bug fix)
    for kept in ("LLM integration", "fine-tuning", "LLM evaluation", "Python", "Java", "CI/CD"):
        assert kept in rendered, f"{kept} is a real declared skill and must be kept"
    # a JD term the candidate genuinely lacks stays missing (truth-preserving)
    for gap in ("agentic workflows", "cloud infrastructure"):
        assert gap not in rendered, f"{gap} is not evidenced — must not be fabricated in"


async def test_software_every_declared_skill_is_backfilled_even_if_model_drops_it() -> None:
    """Declared skills are facts: included by default even when the model omits them."""
    facts = _automation_lead_facts()
    jd = _ai_qa_jd()
    llm = AsyncMock()
    # the model returns ONLY one keyword; every other declared skill must still appear
    llm.call_tool.return_value = {
        "categories": [
            {
                "category": "QA",
                "keywords": [{"keyword": "Python", "tier": "concrete", "category": "Programming"}],
            }
        ]
    }

    skills = await _generate_skills(facts, jd, llm)
    rendered = {kw.lower() for c in skills.categories for kw in c.keywords}

    for declared in ("dart", "swift", "kotlin", "c++", "release automation", "team management"):
        assert declared in rendered, f"{declared} declared by the candidate must be backfilled"


async def test_only_in_declared_skills_jd_keyword_surfaces_raising_coverage() -> None:
    """A concrete JD keyword present ONLY in the declared Skills block now surfaces.

    The old intersect-and-discard baseline scored evidence WITHOUT the skills block, so such a
    keyword was dropped and after-coverage understated. It must now be in the scored render text.
    """
    facts = CVFacts(
        personal_info=FactsPersonalInfo(name="Dev", email="dev@example.com"),
        experiences=[
            FactsExperience(
                role="Engineer",
                company="Co",
                start_date="2020",
                end_date="2024",
                bullets=["Shipped backend services"],  # NOTE: no mention of Kubernetes
            )
        ],
        skills=[FactsSkillGroup(category="Platforms", items=["Kubernetes", "Docker"])],
    )
    jd = JdAnalysis(
        role_title="Platform Engineer",
        company="Co",
        keywords=["Kubernetes", "Docker"],
        keyword_tiers={"Kubernetes": "concrete", "Docker": "concrete"},
        requirements_must=["Kubernetes"],
        candidate_level=CandidateLevel.MID,
    )
    llm = AsyncMock()
    llm.call_tool.return_value = {
        "categories": [
            {
                "category": "Platforms",
                "keywords": [
                    {"keyword": "Kubernetes", "tier": "concrete", "category": "Platforms"},
                    {"keyword": "Docker", "tier": "concrete", "category": "Platforms"},
                ],
            }
        ]
    }

    skills = await _generate_skills(facts, jd, llm)
    scored = _normalize_for_match(" ".join(kw for c in skills.categories for kw in c.keywords))
    assert _term_present("Kubernetes", scored)  # surfaces from the declared skills block
    assert _term_present("Docker", scored)
    # the before-basis includes the declared skills, so this is honest coverage, not false lift
    before = _facts_to_text(facts)
    assert "Kubernetes" in before and "Docker" in before


# --- NON-SOFTWARE fixture (the universality guard) ----------------------------


def _nurse_facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Pat Carer", email="pat@example.com"),
        experiences=[
            FactsExperience(
                role="Registered Nurse",
                company="St. Mary's",
                start_date="2018",
                end_date="2024",
                bullets=["Performed triage and medication administration on a busy ward"],
            )
        ],
        skills=[
            FactsSkillGroup(
                category="Clinical Skills",
                items=["triage", "medication administration", "wound care"],
            ),
            FactsSkillGroup(category="Certifications", items=["ACLS", "BLS"]),
        ],
        languages=[FactsLanguage(language="Spanish", level="Fluent")],
    )


def _clinical_jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="ICU Nurse",
        company="General Hospital",
        keywords=["triage", "medication administration", "ACLS", "ECMO"],
        keyword_tiers={
            "triage": "competency",
            "medication administration": "competency",
            "ACLS": "concrete",
            "ECMO": "concrete",  # a real clinical tech the CV does NOT mention → gap
        },
        requirements_must=["triage"],
        candidate_level=CandidateLevel.MID,
    )


async def test_nursing_real_skills_kept_and_absent_term_stays_missing() -> None:
    facts = _nurse_facts()
    jd = _clinical_jd()
    llm = AsyncMock()
    llm.call_tool.return_value = {
        "categories": [
            {
                "category": "outer",
                "keywords": [
                    {"keyword": "triage", "tier": "competency", "category": "Clinical Skills",
                     "anchor_ref": "Performed triage and medication administration on a busy ward"},
                    {"keyword": "wound care", "tier": "competency", "category": "Clinical Skills",
                     "anchor_ref": "Performed triage and medication administration on a busy ward"},
                    {"keyword": "ACLS", "tier": "concrete", "category": "Certifications"},
                    # a concrete clinical tech NOT in the CV → must not surface
                    {"keyword": "ECMO", "tier": "concrete", "category": "Clinical Skills"},
                ],
            }
        ]
    }

    skills = await _generate_skills(facts, jd, llm)
    headers = [c.category for c in skills.categories]
    rendered = [kw for c in skills.categories for kw in c.keywords]

    # real clinical skills kept (triage / medication administration / ACLS / BLS / wound care)
    for kept in ("triage", "medication administration", "wound care", "ACLS", "BLS"):
        assert kept in rendered, f"{kept} is a real declared skill and must be kept"
    # field-appropriate emergent headers, no forced software vocabulary
    assert "Clinical Skills" in headers and "Certifications" in headers
    for software_word in ("Languages", "Frameworks", "Tools & Platforms"):
        assert software_word not in headers
    # a clinical term the CV lacks stays a gap
    assert "ECMO" not in rendered
    # spoken language is facts-sourced, appended last (not duplicated into skills)
    assert skills.categories[-1].category == "Spoken Languages"
    assert "triage" not in [c.category for c in skills.categories]  # sanity: header != skill
