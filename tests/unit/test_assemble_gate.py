"""Unit tests for assembly + the one-page global gate.

`assemble_and_gate` builds the CV in a level-driven section order, deduplicates content
across sections, compresses overflow to a single page, and emits a one_page_pressure
flag when role count exceeds MAX_ROLES_NO_WARNING.
"""

from __future__ import annotations

from cv_generator import (
    MAX_PAGES,
    MAX_ROLES_NO_WARNING,
    SPOKEN_LANGUAGES_CATEGORY,
    FlagKind,
    assemble_and_gate,
    estimate_page_count,
)
from helprers.cv_template import (
    BulletPoint,
    Category,
    Certificate,
    CVTemplate,
    Education,
    Experience,
    Language,
    PersonalInfo,
    Skills,
    Summary,
)
from schemas import CandidateLevel


def _bullet(text: str) -> BulletPoint:
    return BulletPoint(action_verb="Developed", description=text)


def _exp(company: str, bullets: list[BulletPoint]) -> Experience:
    return Experience(
        role="Engineer",
        company=company,
        company_description="A solid company with a long enough description here",
        start_date="2020",
        end_date="2022",
        bullets=bullets,
    )


def _sections(**overrides: object) -> dict:
    base: dict = {
        "personal_info": PersonalInfo(name="Ada", location="London", email="ada@x.io"),
        "summary": Summary(text="line one\nline two\nline three"),
        "skills": Skills(categories=[Category(category="Languages", keywords=["Python"])]),
        "experiences": [],
        "education": [],
        "projects": [],
        "certificates": [],
        "languages": [],
    }
    base.update(overrides)
    return base


def test_section_order_junior_leads_with_education() -> None:
    sections = _sections(experiences=[_exp("Acme", [_bullet("a")])], education=[
        Education(institution="MIT", degree="BSc")
    ])
    cv, _flags = assemble_and_gate(sections, CandidateLevel.NEW_GRAD)
    order = cv.section_order
    assert order.index("education") < order.index("experience")


def test_section_order_senior_leads_with_experience() -> None:
    sections = _sections(experiences=[_exp("Acme", [_bullet("a")])], education=[
        Education(institution="MIT", degree="BSc")
    ])
    cv, _flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)
    order = cv.section_order
    assert order.index("experience") < order.index("education")


def test_global_gate_dedups_skills_and_bullets() -> None:
    sections = _sections(
        skills=Skills(categories=[Category(category="Languages", keywords=["Go", "go", "Rust"])]),
        experiences=[
            _exp("Acme", [_bullet("shared win"), _bullet("shared win"), _bullet("other")])
        ],
    )
    cv, _flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)

    assert cv.skills.categories[0].keywords == ["Go", "Rust"]  # case-insensitive dedup
    descriptions = [b.description for b in cv.experiences[0].bullets]
    assert descriptions.count("shared win") == 1
    assert "other" in descriptions


def test_global_gate_compresses_to_one_page() -> None:
    # Role-unique descriptions so dedup keeps them; the volume forces compression.
    bulky = [
        _exp(f"Company {c}", [_bullet(f"company {c} achievement {i}") for i in range(25)])
        for c in range(3)  # <= MAX_ROLES_NO_WARNING so no pressure flag here
    ]
    original_bullets = sum(len(e.bullets) for e in bulky)

    cv, _flags = assemble_and_gate(_sections(experiences=bulky), CandidateLevel.SENIOR_IC)

    assert estimate_page_count(cv) <= MAX_PAGES
    # Overflow was actually shed, and no role was emptied (>=1 bullet each).
    assert sum(len(exp.bullets) for exp in cv.experiences) < original_bullets
    assert all(len(exp.bullets) >= 1 for exp in cv.experiences)


def test_compression_preserves_keyword_bearing_bullet() -> None:
    filler = [_bullet(f"did routine task number {i}") for i in range(60)]
    keyword_bullet = BulletPoint(action_verb="Developed", description="ran Kubernetes clusters")
    sections = _sections(experiences=[_exp("Acme", [*filler, keyword_bullet])])

    cv, _flags = assemble_and_gate(
        sections, CandidateLevel.SENIOR_IC, keywords=["Kubernetes"]
    )

    assert estimate_page_count(cv) <= MAX_PAGES  # still compressed to fit
    surviving = " ".join(b.description for b in cv.experiences[0].bullets)
    assert "Kubernetes" in surviving  # keyword-bearing bullet was not trimmed away


def test_more_than_three_roles_emits_one_page_pressure() -> None:
    roles = [
        _exp(f"Company {c}", [_bullet("did a thing")])
        for c in range(MAX_ROLES_NO_WARNING + 1)
    ]
    sections = _sections(experiences=roles)

    cv, flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)

    assert any(f.kind is FlagKind.ONE_PAGE_PRESSURE for f in flags)
    assert estimate_page_count(cv) <= MAX_PAGES


def test_overflow_after_compression_emits_flag() -> None:
    roles = [_exp(f"Company {c}", [_bullet("did a thing")]) for c in range(3)]
    education = [Education(institution=f"School {i}", degree="BSc") for i in range(60)]
    sections = _sections(experiences=roles, education=education)

    cv, flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)

    assert estimate_page_count(cv) > MAX_PAGES  # genuinely cannot compress to one page
    assert any(f.kind is FlagKind.ONE_PAGE_PRESSURE for f in flags)  # never ships silently


def test_shared_bullet_across_roles_is_kept_in_both() -> None:
    sections = _sections(
        experiences=[
            _exp("Acme", [_bullet("led the migration")]),
            _exp("Globex", [_bullet("led the migration")]),
        ]
    )

    cv, _flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)

    assert [b.description for b in cv.experiences[0].bullets] == ["led the migration"]
    assert [b.description for b in cv.experiences[1].bullets] == ["led the migration"]


def test_skill_heavy_cv_overflows_one_page() -> None:
    bulk_skills = Skills(
        categories=[
            Category(category="Languages", keywords=[f"relevant-{i}" for i in range(200)]),
            Category(category="Tools", keywords=[f"hard-{i}" for i in range(200)]),
            Category(category="Concepts", keywords=[f"soft-{i}" for i in range(200)]),
        ],
    )
    sections = _sections(skills=bulk_skills)

    cv, flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)

    assert estimate_page_count(cv) > MAX_PAGES  # volume is no longer flattened to ~3 lines
    assert any(f.kind is FlagKind.ONE_PAGE_PRESSURE for f in flags)


def test_section_order_includes_populated_certificates_and_languages() -> None:
    sections = _sections(
        certificates=[Certificate(title="AWS SA", issuer="Amazon", year=2024)],
        languages=[Language(language="English", level="Native")],
    )

    cv, _flags = assemble_and_gate(sections, CandidateLevel.SENIOR_IC)

    assert "certificates" in cv.section_order
    assert "languages" in cv.section_order


def test_estimate_page_count_counts_spoken_languages_once() -> None:
    langs = [Language(language="English", level="Fluent"), Language(language="French", level="B2")]
    exp = _exp("Acme", [_bullet(f"b{i}") for i in range(42)])  # 2 + 42 = 44 lines

    def _cv(extra: list[Category]) -> CVTemplate:
        return CVTemplate(
            personal_info=PersonalInfo(name="Ada", location="London", email="ada@x.io"),
            summary=Summary(text="x"),  # 1 line
            skills=Skills(categories=[Category(category="Tools", keywords=["Python"])] + extra),
            experiences=[exp],
            languages=langs,  # counted once: +2
        )

    base = _cv([])  # 2 + 1 + 1 + 2 + 44 = 50 → exactly one page
    spoken = Category(category=SPOKEN_LANGUAGES_CATEGORY, keywords=["English fluent", "French b2"])
    dup = _cv([spoken])

    assert estimate_page_count(base) == 1
    assert estimate_page_count(dup) == estimate_page_count(base)  # spoken category must not inflate
