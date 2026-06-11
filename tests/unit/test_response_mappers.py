"""Unit tests for the dataclass -> Pydantic response mappers in main.py.

The transport boundary mirrors the pipeline's `CVTemplate` / `AtsScore` / `SectionFlag`
dataclasses field-by-field. This is the drift guard: every
dataclass field MUST be present on its Pydantic mirror, and the mappers must copy values
faithfully (including the `action_verb` enum -> string coercion).
"""

from __future__ import annotations

import dataclasses

import main
from cv_generator import AtsScore, FlagKind, SectionFlag
from helprers import cv_template as ct

# `Link.URL_REGEX` is a compiled-pattern validation constant that dataclasses surfaces as a
# "field"; it is not part of the transport contract (CVLink is title+url only), so the
# drift guard ignores non-data helper constants like it.
# `Skills.provenance` is validation-time evidence (two-tier model) deliberately kept OFF the
# transport contract — tier/anchor are never rendered, copied, or scored — so it is excluded
# from the drift guard exactly like URL_REGEX (a non-transport field, not a mapper omission).
_IGNORED_FIELDS = {"URL_REGEX", "provenance"}


def _dataclass_field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)} - _IGNORED_FIELDS


# --- drift guard (every dataclass field mirrored) ------------------

_MIRROR_PAIRS = [
    (ct.CVTemplate, main.CVSchema),
    (ct.PersonalInfo, main.CVPersonalInfo),
    (ct.Summary, main.CVSummary),
    (ct.Skills, main.CVSkills),
    (ct.Category, main.CVCategory),
    (ct.BulletPoint, main.CVBullet),
    (ct.Experience, main.CVExperience),
    (ct.Education, main.CVEducation),
    (ct.Project, main.CVProject),
    (ct.Certificate, main.CVCertificate),
    (ct.Language, main.CVLanguage),
    (ct.Link, main.CVLink),
    (AtsScore, main.AtsScore),
    (SectionFlag, main.Flag),
]


def test_pydantic_mirror_preserves_every_dataclass_field() -> None:
    for dataclass_cls, pydantic_cls in _MIRROR_PAIRS:
        dc_fields = _dataclass_field_names(dataclass_cls)
        py_fields = set(pydantic_cls.model_fields.keys())
        missing = dc_fields - py_fields
        assert not missing, (
            f"{pydantic_cls.__name__} is missing {missing} from {dataclass_cls.__name__}"
        )


def test_pydantic_mirror_preserves_required_ness() -> None:
    """A field that is required upstream (no default) MUST be required on the mirror.

    The name-only guard above let `CVCertificate.year` drift from required `int` to optional;
    this catches that class of drift — a required dataclass field silently defaulting on the
    mirror, so a mapper bug would let it fall back to the default instead of failing.
    """
    for dataclass_cls, pydantic_cls in _MIRROR_PAIRS:
        py_fields = pydantic_cls.model_fields
        for dc_field in dataclasses.fields(dataclass_cls):
            if dc_field.name in _IGNORED_FIELDS or dc_field.name not in py_fields:
                continue
            dc_required = (
                dc_field.default is dataclasses.MISSING
                and dc_field.default_factory is dataclasses.MISSING
            )
            if dc_required:
                assert py_fields[dc_field.name].is_required(), (
                    f"{pydantic_cls.__name__}.{dc_field.name} is optional but "
                    f"{dataclass_cls.__name__}.{dc_field.name} is required upstream"
                )


def test_map_cv_preserves_content_and_coerces_action_verb() -> None:
    cv = ct.CVTemplate(
        personal_info=ct.PersonalInfo(
            name="Jane Doe",
            location="Remote",
            email="jane@example.com",
            links=[ct.Link(title="LinkedIn", url="https://linkedin.com/in/jane")],
        ),
        summary=ct.Summary(text="Engineer.", relevant_skills=["Python"]),
        skills=ct.Skills(
            categories=[
                ct.Category(category="Languages", keywords=["Python"]),
                ct.Category(category="Tools", keywords=["FastAPI"]),
            ]
        ),
        experiences=[
            ct.Experience(
                role="Dev",
                company="Acme",
                company_description="Leading platform",
                start_date="2020-01",
                end_date="2022-01",
                bullets=[
                    ct.BulletPoint(
                        action_verb=ct.ActionVerb.DEVELOPED,
                        description="services",
                        skills=["Python"],
                        impact="+10%",
                    )
                ],
            )
        ],
        section_order=["contact", "summary", "skills", "experience"],
    )
    mapped = main._map_cv(cv)
    assert isinstance(mapped, main.CVSchema)
    assert mapped.personal_info.name == "Jane Doe"
    assert mapped.personal_info.links[0].url == "https://linkedin.com/in/jane"
    assert mapped.section_order == ["contact", "summary", "skills", "experience"]
    # action_verb is the ActionVerb enum upstream; the mirror is a plain string value.
    assert mapped.experiences[0].bullets[0].action_verb == "Developed"


def test_map_cv_admits_none_optional_fields_without_raising() -> None:
    """A real CV that omits location / company_description / cert year+issuer / language
    level / link title maps + serializes WITHOUT raising (reproduces the live stage=assemble
    500 that fired on personal_info.location=None, plus the next-to-fail fields).
    """
    cv = ct.CVTemplate(
        personal_info=ct.PersonalInfo(
            name="Jane Doe",
            email="jane@example.com",
            location=None,  # ROOT of the live 500
            links=[ct.Link(url="https://example.com/jane")],  # bare URL, no title
        ),
        summary=ct.Summary(text="Engineer."),
        skills=ct.Skills(),
        experiences=[
            ct.Experience(
                role="Dev",
                company="ITV Group",
                start_date="2020-01",
                end_date="2022-01",
                company_description=None,  # entries with no blurb
                bullets=[ct.BulletPoint(action_verb="Developed", description="services")],
            )
        ],
        certificates=[ct.Certificate(title="AWS SA", issuer=None, year=None)],
        languages=[ct.Language(language="English", level=None)],
        projects=[ct.Project(name="Engine Notes", description=None)],
    )
    mapped = main._map_cv(cv)
    # The full response envelope must also serialize (the 500 fired on GenerateResponse build).
    response = main.GenerateResponse(
        cv=mapped,
        cover_letter="",
        ats_score=main._map_ats(AtsScore(before_pct=0.0, after_pct=0.0)),
    )
    assert response.cv.personal_info.location is None
    assert response.cv.personal_info.links[0].title is None
    assert response.cv.experiences[0].company_description is None
    assert response.cv.certificates[0].issuer is None
    assert response.cv.certificates[0].year is None
    assert response.cv.languages[0].level is None
    assert response.cv.projects[0].description is None
    # Core identity is preserved verbatim (never dropped or defaulted).
    assert response.cv.personal_info.name == "Jane Doe"
    assert response.cv.experiences[0].company == "ITV Group"


def test_map_ats_preserves_before_after_and_coverage() -> None:
    ats = AtsScore(before_pct=30.0, after_pct=75.0, matched=["python"], missing=["go"])
    mapped = main._map_ats(ats)
    assert isinstance(mapped, main.AtsScore)
    assert (mapped.before_pct, mapped.after_pct) == (30.0, 75.0)
    assert mapped.coverage_pct == 75.0  # == after_pct (computed property mirrored)
    assert mapped.matched == ["python"]
    assert mapped.missing == ["go"]


def test_map_flag_preserves_section_kind_message() -> None:
    flag = SectionFlag(
        section="global", kind=FlagKind.UNMET_COVERAGE, message="Missing: Kubernetes"
    )
    mapped = main._map_flag(flag)
    assert isinstance(mapped, main.Flag)
    assert mapped.section == "global"
    assert mapped.kind == "unmet_coverage"
    assert mapped.message == "Missing: Kubernetes"
