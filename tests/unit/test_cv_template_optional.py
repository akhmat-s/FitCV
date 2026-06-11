"""FIX 8: the deprecated ``cv_template.to_html`` / ``validate`` paths must be None-safe for the
now-Optional fields (``level``, ``start_date``/``end_date``, ``company_description``,
``location``, ``link.title``).

These methods are NOT on the live ``/generate`` path (no PDF in the MVP — the live validator is
``cv_generator._validate_languages``), but they were newly broken when those fields became
Optional and still deref them (``len(None)``, ``None.lower()``, ``html.escape(None)``). Field-
agnostic by construction — no profession lexicon appears in any assertion.
"""

from __future__ import annotations

from helprers.cv_template import (
    CVTemplate,
    Experience,
    Language,
    Link,
    PersonalInfo,
    Skills,
    Summary,
)


def _cv(**overrides: object) -> CVTemplate:
    base: dict = dict(
        personal_info=PersonalInfo(name="Ada", email="ada@x.io"),  # location omitted → None
        summary=Summary(text="line one\nline two\nline three"),  # 3 lines → no summary error
        skills=Skills(),
        languages=[Language(language="English")],  # level omitted → None
        experiences=[Experience(role="Engineer", company="Acme")],  # dates/description → None
    )
    base.update(overrides)
    return CVTemplate(**base)


def test_to_html_is_none_safe_for_optional_fields() -> None:
    # location=None, link.title=None, company_description=None, start/end_date=None, level=None.
    cv = _cv(
        personal_info=PersonalInfo(
            name="Ada",
            email="ada@x.io",
            links=[Link(url="https://example.com")],  # title omitted → None
        ),
    )
    html = cv.to_html()  # must not raise AttributeError/TypeError on any None field
    assert isinstance(html, str) and html
    assert "None" not in html  # no literal "None" leaks from a missing field


def test_validate_is_none_safe_and_no_spurious_level_error() -> None:
    # company_description=None previously crashed validate() at len(None); a missing language level
    # previously raised a spurious "invalid level" error. After FIX 8 the CV is otherwise valid, so
    # validate() returns None (no raise) — proving both the crash and the spurious error are gone.
    cv = _cv()
    assert cv.validate() is None
