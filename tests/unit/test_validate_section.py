"""Unit tests for deterministic section validation.

`validate_section` is pure and deterministic (no LLM): it returns a SectionValidation
partitioning blocking errors from non-blocking warnings. These cover writing checks,
ATS checks (heading / keyword floor / parse-friendly contact), factual integrity,
the summary 3–5 line rule, and the ActionVerb warning-only decision.
"""

from __future__ import annotations

from cv_generator import MIN_KEYWORDS, validate_section
from helprers.cv_template import (
    BulletPoint,
    Certificate,
    Education,
    Experience,
    Language,
    Link,
    PersonalInfo,
    Project,
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
    TargetSection,
)

_KEYWORDS = ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"]


def _facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        experiences=[
            FactsExperience(
                role="Analyst",
                company="Analytical Engines",
                start_date="1840",
                end_date="1843",
                # bullets evidence the JD keywords so the anti-fabrication gate keeps them
                bullets=["Built Python, FastAPI, Pydantic, PostgreSQL and Docker services"],
            )
        ],
        projects=[FactsProject(name="Engine Notes", description="d")],
        certificates=[FactsCertificate(title="Mathematics", issuer="Royal Society")],
        languages=[FactsLanguage(language="English", level="Native")],
    )


def _jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=list(_KEYWORDS),
        requirements_must=["Python"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )


def _experience(**kw: object) -> Experience:
    base = dict(
        role="Senior Analyst",
        company="Analytical Engines",
        company_description="Pioneering computation laboratory of the era",
        start_date="1840",
        end_date="1843",
        bullets=[BulletPoint(action_verb="Developed", description="the first algorithm")],
    )
    base.update(kw)
    return Experience(**base)


def test_link_url_must_be_http() -> None:
    facts, jd = _facts(), _jd()
    contact = PersonalInfo(name="Ada", location="London", email="ada@x.io",
                           links=[Link(title="site", url="ftp://nope")])
    result = validate_section(TargetSection.CONTACT, contact, facts, jd)
    assert any("https" in e for e in result.errors)


def test_language_level_must_be_allowed() -> None:
    facts, jd = _facts(), _jd()
    langs = [Language(language="English", level="Z9")]
    result = validate_section("languages", langs, facts, jd)
    assert any("level" in e.lower() for e in result.errors)


def test_language_without_a_level_is_not_an_error() -> None:
    # The Optional refactor: an absent proficiency is truthful, not a validation error. The level
    # check now fires ONLY when a level is present (a bogus, present level still errors above).
    facts, jd = _facts(), _jd()
    langs = [Language(language="English", level=None)]
    result = validate_section("languages", langs, facts, jd)
    assert not any("level" in e.lower() for e in result.errors)


def _date_less_source_facts() -> CVFacts:
    return CVFacts(
        personal_info=FactsPersonalInfo(name="Ada", location="London", email="ada@x.io"),
        experiences=[
            FactsExperience(
                role="Senior Analyst",
                company="Analytical Engines",
                bullets=["Built Python, FastAPI, Pydantic, PostgreSQL and Docker services"],
            )
        ],
    )


def test_date_less_source_with_null_generated_dates_passes_integrity() -> None:
    # A source role with NO dates, generated with null dates, must NOT raise a spurious
    # "dates do not match" error (symmetric absence matches via the normalized key).
    facts, jd = _date_less_source_facts(), _jd()
    exp = _experience(start_date=None, end_date=None)
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert not any("do not match source" in e.lower() for e in result.errors)


def test_fabricated_date_on_date_less_source_still_fails_integrity() -> None:
    # De-requiring dates removed the fabrication PRESSURE, not the integrity guard: an invented
    # date absent from a date-less source is still caught.
    facts, jd = _date_less_source_facts(), _jd()
    exp = _experience(start_date="2099", end_date="2100")
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert any("do not match source" in e.lower() for e in result.errors)


def test_overlong_bullet_is_warning_not_error() -> None:
    facts, jd = _facts(), _jd()
    long_desc = "x" * 200
    exp = _experience(bullets=[BulletPoint(action_verb="Developed", description=long_desc)])
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert result.errors == []
    assert any("exceeds" in w for w in result.warnings)


def test_short_company_description_is_warning() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(company_description="tiny")
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert any("company_description" in w for w in result.warnings)


def test_impact_without_digit_is_warning() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(
        bullets=[BulletPoint(action_verb="Developed", description="d", impact="big gains")]
    )
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert any("digit" in w for w in result.warnings)


def test_nonstandard_section_heading_is_error() -> None:
    facts, jd = _facts(), _jd()
    result = validate_section("references", Summary(text="a\nb\nc"), facts, jd)
    assert any("ATS-standard" in e for e in result.errors)


def test_skills_below_keyword_floor_is_error() -> None:
    from helprers.cv_template import Category, Skills

    facts, jd = _facts(), _jd()
    # only 2 < MIN_KEYWORDS
    too_few = Skills(categories=[Category(category="Languages", keywords=["Python", "FastAPI"])])
    result = validate_section(TargetSection.SKILLS, too_few, facts, jd)
    assert any(str(MIN_KEYWORDS) in e for e in result.errors)


def test_skills_meeting_keyword_floor_has_no_keyword_error() -> None:
    from helprers.cv_template import Category, Skills

    facts, jd = _facts(), _jd()
    # all 5
    enough = Skills(categories=[Category(category="Languages", keywords=list(_KEYWORDS))])
    result = validate_section(TargetSection.SKILLS, enough, facts, jd)
    assert result.errors == []


def test_skills_floor_uses_word_boundary_not_substring() -> None:
    from helprers.cv_template import Category, Skills

    facts = _facts()
    jd = JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=["Go", "Java", "Rust", "Scala", "Perl"],  # 5 keywords
        requirements_must=["Python"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )
    # Each entry merely CONTAINS a keyword as a substring; none is a whole-word match.
    sneaky = Skills(
        categories=[
            Category(
                category="Languages",
                keywords=["Golang", "JavaScript", "Rustaceans", "Scalable", "Perlite"],
            )
        ]
    )
    result = validate_section(TargetSection.SKILLS, sneaky, facts, jd)
    assert any(str(MIN_KEYWORDS) in e for e in result.errors)


def test_skills_floor_skipped_when_jd_has_few_keywords() -> None:
    from helprers.cv_template import Category, Skills

    facts = _facts()
    jd = JdAnalysis(
        role_title="Engineer",
        company="Globex",
        keywords=["Python", "FastAPI", "Docker"],  # only 3 < MIN_KEYWORDS
        requirements_must=["Python"],
        candidate_level=CandidateLevel.SENIOR_IC,
    )
    # surfaces every JD keyword
    skills = Skills(
        categories=[Category(category="Languages", keywords=["Python", "FastAPI", "Docker"])]
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    assert result.errors == []


def test_skills_fabricated_keyword_is_blocking_error() -> None:
    from helprers.cv_template import Category, Skills

    facts, jd = _facts(), _jd()  # facts evidence Python..Docker, never Kubernetes
    fabricated = Skills(
        categories=[Category(category="Languages", keywords=["Python", "Kubernetes"])]
    )
    result = validate_section(TargetSection.SKILLS, fabricated, facts, jd)
    assert any("Kubernetes" in e and "fabrication" in e.lower() for e in result.errors)


def test_empty_skills_is_not_a_vacuous_pass() -> None:
    from helprers.cv_template import Skills

    facts, jd = _facts(), _jd()
    result = validate_section(TargetSection.SKILLS, Skills(), facts, jd)
    assert result.errors  # an empty skills section must block, never pass silently


def test_skills_noise_keyword_is_warning_not_error() -> None:
    from helprers.cv_template import Category, Skills

    facts, jd = _facts(), _jd()  # "Analytical" is evidenced (company) but not a JD keyword
    noisy = Skills(
        categories=[Category(category="Languages", keywords=["Python", "Analytical"])]
    )
    result = validate_section(TargetSection.SKILLS, noisy, facts, jd)
    assert not any("Analytical" in e for e in result.errors)  # evidenced → not fabrication
    assert any("Analytical" in w and "noise" in w.lower() for w in result.warnings)


def test_spoken_languages_category_is_exempt_from_evidence_gate() -> None:
    from helprers.cv_template import Category, Skills

    facts, jd = _facts(), _jd()  # source language English / Native
    skills = Skills(
        categories=[
            Category(category="Languages", keywords=["Python"]),
            Category(category="Spoken Languages", keywords=["English native"]),
        ]
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    # "English native" is not contiguous in the facts text, but the spoken category is exempt
    assert not any("English native" in e for e in result.errors)


def _tiered_facts() -> CVFacts:
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
                    "Wrote tests in Python with Selenium",
                ],
            )
        ],
    )


def _tiered_jd() -> JdAnalysis:
    return JdAnalysis(
        role_title="AI SDET",
        company="AICorp",
        keywords=["Python", "Kubernetes", "Systems Thinking", "SDLC Automation"],
        keyword_tiers={
            "Python": "concrete",
            "Kubernetes": "concrete",
            "Systems Thinking": "competency",
            "SDLC Automation": "competency",
        },
        requirements_must=["Python"],
        candidate_level=CandidateLevel.MID,
    )


def _skills_with_provenance(categories, provenance):
    from helprers.cv_template import Skills

    skills = Skills(categories=categories)
    skills.provenance = provenance
    return skills


def test_validate_concrete_absent_from_cv_is_blocking_error() -> None:
    from cv_generator import SkillProvenance
    from helprers.cv_template import Category
    from schemas import KeywordTier

    facts, jd = _tiered_facts(), _tiered_jd()  # CV has no "Kubernetes"
    skills = _skills_with_provenance(
        [Category(category="Tools & Platforms", keywords=["Python", "Kubernetes"])],
        [
            SkillProvenance(keyword="Python", tier=KeywordTier.CONCRETE),
            SkillProvenance(keyword="Kubernetes", tier=KeywordTier.CONCRETE),
        ],
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    assert any("Kubernetes" in e and "fabrication" in e.lower() for e in result.errors)


def test_validate_competency_with_traceable_anchor_passes() -> None:
    from cv_generator import SkillProvenance
    from helprers.cv_template import Category
    from schemas import KeywordTier

    facts, jd = _tiered_facts(), _tiered_jd()
    skills = _skills_with_provenance(
        [Category(category="Practices & Concepts", keywords=["Systems Thinking"])],
        [
            SkillProvenance(
                keyword="Systems Thinking",
                tier=KeywordTier.COMPETENCY,
                anchor_ref="Architected the test automation infrastructure",
            )
        ],
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    assert result.errors == []
    # the competency survives (it is not dropped)
    rendered = [kw for c in skills.categories for kw in c.keywords]
    assert "Systems Thinking" in rendered


def test_validate_competency_with_untraceable_anchor_is_dropped() -> None:
    from cv_generator import SkillProvenance
    from helprers.cv_template import Category
    from schemas import KeywordTier

    facts, jd = _tiered_facts(), _tiered_jd()
    skills = _skills_with_provenance(
        [Category(category="Practices & Concepts", keywords=["Python", "SDLC Automation"])],
        [
            SkillProvenance(keyword="Python", tier=KeywordTier.CONCRETE),
            SkillProvenance(
                keyword="SDLC Automation",
                tier=KeywordTier.COMPETENCY,
                anchor_ref="Led an org-wide release-management overhaul",  # not in the CV
            ),
        ],
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    rendered = [kw for c in skills.categories for kw in c.keywords]
    # the untraceable competency is dropped (not an error), the valid concrete remains
    assert "SDLC Automation" not in rendered
    assert "Python" in rendered
    assert not any("SDLC Automation" in e for e in result.errors)


def test_validate_competency_with_missing_provenance_is_dropped() -> None:
    from helprers.cv_template import Category

    facts, jd = _tiered_facts(), _tiered_jd()
    # competency keyword rendered but NO provenance entry recorded at all
    skills = _skills_with_provenance(
        [Category(category="Practices & Concepts", keywords=["Systems Thinking"])], []
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    rendered = [kw for c in skills.categories for kw in c.keywords]
    assert "Systems Thinking" not in rendered  # dropped — no anchor to trace
    # dropping the only keyword leaves the section vacuous → blocking error
    assert result.errors


def test_validate_concrete_with_bogus_anchor_still_errors_not_laundered() -> None:
    from cv_generator import SkillProvenance
    from helprers.cv_template import Category
    from schemas import KeywordTier

    facts, jd = _tiered_facts(), _tiered_jd()  # CV has no "Kubernetes"
    # a real CV line handed as a bogus anchor on a CONCRETE keyword must NOT launder it
    skills = _skills_with_provenance(
        [Category(category="Tools & Platforms", keywords=["Python", "Kubernetes"])],
        [
            SkillProvenance(keyword="Python", tier=KeywordTier.CONCRETE),
            SkillProvenance(
                keyword="Kubernetes",
                tier=KeywordTier.COMPETENCY,  # model lies about the tier
                anchor_ref="Architected the test automation infrastructure",
            ),
        ],
    )
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    # tier governs (jd.tier_of('Kubernetes') == concrete) → literal check → still a fabrication
    assert any("Kubernetes" in e and "fabrication" in e.lower() for e in result.errors)


def test_validate_skills_does_not_check_header_format() -> None:
    # Header FORMAT left the validator. A header is NEVER a reason to fail validation or
    # regenerate — a clean shape is an invariant enforced by construction (_normalize_skill_headers)
    # at assembly, not validated-and-rejected. So even a compound/long header raises NO error here.
    from cv_generator import SkillProvenance
    from helprers.cv_template import Category
    from schemas import KeywordTier

    facts, jd = _tiered_facts(), _tiered_jd()  # CV evidences "Python"
    for header in (
        "Quality Engineering & SDET Tooling",  # compound + long
        "An Extraordinarily Long Editorialized Header Of Doom",  # > 3 words / > 24 chars
        "CI/CD, Testing & AI Integration",  # comma + multi-conjunction
        "Testing",
        "Clinical Skills",
    ):
        skills = _skills_with_provenance(
            [Category(category=header, keywords=["Python"])],
            [SkillProvenance(keyword="Python", tier=KeywordTier.CONCRETE)],
        )
        result = validate_section(TargetSection.SKILLS, skills, facts, jd)
        assert not any("format" in e.lower() for e in result.errors), header


async def test_validate_skills_guardrail_unevidenced_techs_never_rendered() -> None:
    """Unevidenced techs/competencies never reach rendered Skills."""
    from cv_generator import _generate_skills

    class _Stub:
        async def call_tool(self, *_a, **_k):
            return {
                "categories": [
                    {
                        "category": "Cloud & Infra",
                        "keywords": [
                            {"keyword": "AWS", "tier": "concrete"},
                            {"keyword": "Kafka", "tier": "concrete"},
                            {"keyword": "Kubernetes", "tier": "concrete"},
                            {
                                "keyword": "Agentic Frameworks",
                                "tier": "competency",
                                "anchor_ref": "Built agentic LLM orchestration frameworks",
                            },
                            {
                                "keyword": "AI Agents",
                                "tier": "competency",
                                "anchor_ref": "Shipped autonomous AI agents to production",
                            },
                        ],
                    }
                ]
            }

    facts = _tiered_facts()  # none of those techs appear in the CV
    jd = JdAnalysis(
        role_title="AI SDET",
        company="AICorp",
        keywords=["AWS", "Kafka", "Kubernetes", "Agentic Frameworks", "AI Agents"],
        keyword_tiers={
            "AWS": "concrete",
            "Kafka": "concrete",
            "Kubernetes": "concrete",
            "Agentic Frameworks": "competency",
            "AI Agents": "competency",
        },
        candidate_level=CandidateLevel.MID,
    )
    skills = await _generate_skills(facts, jd, _Stub())
    rendered = [kw for c in skills.categories for kw in c.keywords]
    for gap in ("AWS", "Kafka", "Kubernetes", "Agentic Frameworks", "AI Agents"):
        assert gap not in rendered
    # and the validator agrees the section is not a vacuous success
    result = validate_section(TargetSection.SKILLS, skills, facts, jd)
    assert result.errors


def test_score_ats_ignores_tier_and_anchor_only_rendered_keywords_count() -> None:
    """Coverage counts the rendered keyword text only — tier/anchor never score."""
    from cv_generator import SkillProvenance, score_ats
    from helprers.cv_template import (
        Category,
        CVTemplate,
        PersonalInfo,
        Skills,
        Summary,
    )
    from schemas import KeywordTier

    jd = _tiered_jd()
    skills = Skills(categories=[Category(category="Tools", keywords=["Python"])])
    # provenance referencing an unrendered keyword must NOT lift the score
    skills.provenance = [
        SkillProvenance(keyword="Python", tier=KeywordTier.CONCRETE),
        SkillProvenance(
            keyword="Kubernetes",
            tier=KeywordTier.COMPETENCY,
            anchor_ref="Architected the test automation infrastructure across three mobile teams",
        ),
    ]
    cv = CVTemplate(
        personal_info=PersonalInfo(name="Dev", email="dev@example.com"),
        summary=Summary(text="Engineer", relevant_skills=[]),
        skills=skills,
        experiences=[],
        section_order=["skills"],
    )
    ats = score_ats(cv, jd, original_cv_text="")
    # only the rendered "Python" matches; "Kubernetes" (in provenance, not rendered) stays missing
    assert "Python" in ats.matched
    assert "Kubernetes" in ats.missing


def test_contact_empty_name_is_error() -> None:
    facts, jd = _facts(), _jd()
    contact = PersonalInfo(name="", location="London", email="ada@x.io")
    result = validate_section(TargetSection.CONTACT, contact, facts, jd)
    assert any("name" in e.lower() for e in result.errors)


def test_contact_email_without_at_is_error() -> None:
    facts, jd = _facts(), _jd()
    contact = PersonalInfo(name="Ada", location="London", email="ada-at-x")
    result = validate_section(TargetSection.CONTACT, contact, facts, jd)
    assert any("email" in e.lower() for e in result.errors)


def test_contact_absent_phone_is_not_error() -> None:
    facts, jd = _facts(), _jd()
    contact = PersonalInfo(name="Ada", location="London", email="ada@x.io", phone=None)
    result = validate_section(TargetSection.CONTACT, contact, facts, jd)
    assert result.errors == []


def test_contact_any_format_phone_is_not_error() -> None:
    facts, jd = _facts(), _jd()
    contact = PersonalInfo(name="Ada", location="London", email="ada@x.io", phone="dial-me")
    result = validate_section(TargetSection.CONTACT, contact, facts, jd)
    assert result.errors == []


def test_invented_company_is_error() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(company="Imaginary Corp")
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert any("not found in source" in e.lower() for e in result.errors)


def test_mismatched_date_is_error() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(start_date="1999", end_date="2001")
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert any("do not match source" in e.lower() for e in result.errors)


def test_invented_project_is_error() -> None:
    facts, jd = _facts(), _jd()
    result = validate_section(
        TargetSection.PROJECTS, [Project(name="Fake", description="d")], facts, jd
    )
    assert any("not found in source" in e.lower() for e in result.errors)


def test_invented_certificate_is_error() -> None:
    facts, jd = _facts(), _jd()
    result = validate_section(
        "certificates", [Certificate(title="Fake", issuer="Nobody", year=2020)], facts, jd
    )
    assert any("not found in source" in e.lower() for e in result.errors)


def test_invented_language_is_error() -> None:
    facts, jd = _facts(), _jd()
    result = validate_section(
        "languages", [Language(language="Klingon", level="Native")], facts, jd
    )
    assert any("not found in source" in e.lower() for e in result.errors)


def test_valid_languages_section_has_no_errors() -> None:
    facts, jd = _facts(), _jd()
    langs = [Language(language="English", level="Native")]  # in source, valid level
    result = validate_section("languages", langs, facts, jd)
    assert result.errors == []


def _facts_with_education() -> CVFacts:
    facts = _facts()
    facts.education = [FactsEducation(institution="MIT", degree="BSc")]
    return facts


def test_fabricated_education_is_error() -> None:
    facts, jd = _facts_with_education(), _jd()
    fabricated = [Education(institution="Stanford", degree="PhD")]
    result = validate_section(TargetSection.EDUCATION, fabricated, facts, jd)
    assert any("not found in source" in e.lower() for e in result.errors)


def test_truthful_education_has_no_errors() -> None:
    facts, jd = _facts_with_education(), _jd()
    truthful = [Education(institution="MIT", degree="BSc")]
    result = validate_section(TargetSection.EDUCATION, truthful, facts, jd)
    assert result.errors == []


def test_empty_experience_with_source_history_is_error() -> None:
    facts, jd = _facts(), _jd()  # _facts() has one source experience
    result = validate_section(TargetSection.EXPERIENCE, [], facts, jd)
    assert result.errors  # must not be a clean pass


def test_recased_company_is_not_factual_error() -> None:
    facts, jd = _facts(), _jd()  # source company "Analytical Engines"
    exp = _experience(company="analytical engines")  # same company, different case
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert not any("not found in source" in e.lower() for e in result.errors)


def test_lowercase_language_level_is_valid() -> None:
    facts, jd = _facts(), _jd()  # source language English / Native
    langs = [Language(language="english", level="native")]  # recased, still truthful
    result = validate_section("languages", langs, facts, jd)
    assert result.errors == []


def test_absent_company_still_caught_after_normalization() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(company="Globex")  # genuinely not in source
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert any("not found in source" in e.lower() for e in result.errors)


def test_warnings_only_section_has_no_blocking_errors() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(company_description="tiny")  # triggers a warning, no error
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert result.errors == []
    assert result.warnings  # non-blocking signals still surface


def test_factual_error_blocks() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(company="Ghost LLC")
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert result.errors  # blocking → triggers regeneration


def test_summary_too_few_lines_is_error() -> None:
    facts, jd = _facts(), _jd()
    result = validate_section(TargetSection.SUMMARY, Summary(text="only one line"), facts, jd)
    assert result.errors


def test_summary_too_many_lines_is_error() -> None:
    facts, jd = _facts(), _jd()
    six = Summary(text="\n".join(f"line {i}" for i in range(6)))
    result = validate_section(TargetSection.SUMMARY, six, facts, jd)
    assert result.errors


def test_summary_in_range_passes() -> None:
    facts, jd = _facts(), _jd()
    four = Summary(text="\n".join(f"line {i}" for i in range(4)))
    result = validate_section(TargetSection.SUMMARY, four, facts, jd)
    assert result.errors == []


def test_single_paragraph_summary_in_range_passes() -> None:
    facts, jd = _facts(), _jd()
    paragraph = Summary(
        text=(
            "Senior engineer with deep Python and FastAPI experience. "
            "Proven measurable impact across distributed teams. "
            "Ready to lead from day one."
        )
    )
    result = validate_section(TargetSection.SUMMARY, paragraph, facts, jd)
    assert result.errors == []


def test_single_paragraph_summary_too_many_sentences_is_error() -> None:
    facts, jd = _facts(), _jd()
    seven = Summary(text="One. Two. Three. Four. Five. Six. Seven.")
    result = validate_section(TargetSection.SUMMARY, seven, facts, jd)
    assert result.errors


def test_action_verb_out_of_enum_is_warning_only() -> None:
    facts, jd = _facts(), _jd()
    exp = _experience(
        bullets=[BulletPoint(action_verb="Foobarred", description="did the thing")]
    )
    result = validate_section(TargetSection.EXPERIENCE, [exp], facts, jd)
    assert result.errors == []
    assert any("ActionVerb" in w for w in result.warnings)
