"""Unit tests for the Streamlit client's render helpers.

Each helper takes an injected `st`-like object; a `MagicMock` records the widget calls so we
can assert what was rendered without a Streamlit runtime. Covers the structured CV, ATS
before->after, cover letter, honest-gap flags, accessibility labels, and the absence of
download/export or in-UI edit affordances.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app


def _sample_cv() -> dict:
    return {
        "section_order": ["contact", "summary", "skills", "experience", "education"],
        "personal_info": {
            "name": "Jane Doe",
            "location": "Remote",
            "email": "jane@example.com",
            "phone": None,
            "links": [{"title": "LinkedIn", "url": "https://linkedin.com/in/jane"}],
        },
        "summary": {"text": "Backend engineer.", "relevant_skills": ["Python"]},
        "skills": {
            "categories": [
                {"category": "Languages", "keywords": ["Python"]},
                {"category": "Tools", "keywords": ["FastAPI"]},
            ]
        },
        "experiences": [
            {
                "role": "Backend Engineer",
                "company": "Acme",
                "company_description": "Leading platform",
                "start_date": "2020-01",
                "end_date": "2022-01",
                "location": "Remote",
                "bullets": [
                    {
                        "action_verb": "Developed",
                        "description": "payment services",
                        "skills": ["Python"],
                        "impact": "+10%",
                        "benefit": None,
                    }
                ],
            }
        ],
        "education": [
            {
                "institution": "MIT",
                "degree": "BSc",
                "start_year": None,
                "end_year": 2019,
                "gpa": None,
            }
        ],
        "projects": [],
        "certificates": [],
        "languages": [],
    }


def _ats(before: float = 40.0, after: float = 80.0) -> dict:
    return {
        "before_pct": before,
        "after_pct": after,
        "coverage_pct": after,
        "matched": ["python", "fastapi"],
        "missing": ["kubernetes"],
    }


def _all_text(mock_method: MagicMock) -> str:
    """Concatenate every positional/kw string argument across a mock method's calls."""
    chunks: list[str] = []
    for call in mock_method.call_args_list:
        chunks.extend(str(a) for a in call.args)
        chunks.extend(str(v) for v in call.kwargs.values())
    return " ".join(chunks)


# --- TailoredCvView renders one block per section_order ------


def test_tailored_cv_renders_one_block_per_section_copy_friendly() -> None:
    st = MagicMock()
    cv = _sample_cv()
    app.render_tailored_cv(st, cv)
    assert st.subheader.call_count == len(cv["section_order"])
    # Copy-friendly: each section body goes into a copyable st.code block.
    assert st.code.call_count == len(cv["section_order"])


def test_tailored_cv_renders_certificates_content_not_placeholder() -> None:
    # The pipeline appends "certificates" to section_order when present
    # (cv_generator._section_order). The renderer must show its real content, NOT the "—"
    # placeholder it emits for a section with no text builder.
    st = MagicMock()
    cv = _sample_cv()
    cv["section_order"] = ["certificates"]
    cv["certificates"] = [{"title": "AWS SAA", "issuer": "Amazon", "year": 2023, "link": None}]
    app.render_tailored_cv(st, cv)
    rendered = _all_text(st.code)
    assert "AWS SAA" in rendered
    assert "Amazon" in rendered
    # No section block was rendered as the bare "—" placeholder (its content was built).
    section_bodies = [call.args[0] for call in st.code.call_args_list]
    assert "—" not in section_bodies


def test_tailored_cv_renders_no_standalone_languages_section() -> None:
    # The dedicated standalone "Languages" section is dropped: even though the backend still lists
    # "languages" in section_order, the renderer has no builder for it, so no standalone Languages
    # heading appears. Spoken languages render exactly once — inside the Skills block below.
    st = MagicMock()
    cv = _sample_cv()
    cv["section_order"] = ["skills", "languages"]
    cv["skills"] = {
        "categories": [
            {"category": "Languages", "keywords": ["Python"]},
            {"category": "Spoken Languages", "keywords": ["English C1"]},
        ]
    }
    cv["languages"] = [{"language": "English", "level": "C1"}]
    app.render_tailored_cv(st, cv)
    headings = [call.args[0] for call in st.subheader.call_args_list]
    assert "Languages" not in headings  # no standalone Languages section heading
    assert "Skills" in headings
    # Spoken languages still render — once — inside the Skills block.
    rendered = _all_text(st.code)
    assert "Spoken Languages: English C1" in rendered


# --- AtsCoverageMetric shows after_pct + signed delta --------


def test_ats_metric_shows_after_value_and_signed_delta() -> None:
    st = MagicMock()
    app.render_ats_panel(st, _ats(before=40.0, after=80.0))
    st.metric.assert_called_once()
    kwargs = st.metric.call_args.kwargs
    assert "80" in str(kwargs["value"])
    assert str(kwargs["delta"]).startswith("+")  # +40.0%


def test_ats_metric_shows_negative_delta_without_clamp() -> None:
    # when after_pct < before_pct, the delta is a signed negative, never clamped to 0.
    st = MagicMock()
    app.render_ats_panel(st, _ats(before=80.0, after=40.0))
    assert str(st.metric.call_args.kwargs["delta"]).startswith("-")


# --- KeywordCounts renders matched/missing counts as text ----


def test_keyword_counts_render_matched_and_missing_counts() -> None:
    st = MagicMock()
    ats = _ats()
    app.render_ats_panel(st, ats)
    text = _all_text(st.markdown)
    assert str(len(ats["matched"])) in text
    assert str(len(ats["missing"])) in text


# --- CoverLetterExpander renders plain-text cover letter ------


def test_cover_letter_expander_renders_copy_friendly_text() -> None:
    st = MagicMock()
    app.render_cover_letter(st, "Dear team,\n\nI am a great fit.")
    st.expander.assert_called_once()
    assert "Cover letter" in str(st.expander.call_args.args[0])
    assert "I am a great fit." in _all_text(st.code)


# --- CvUploader and JdTextArea each have a visible label ------


def test_inputs_have_visible_labels() -> None:
    st = MagicMock()
    app.render_input_form(st, inputs_off=False, button_off=True)
    uploader_label = st.file_uploader.call_args.args[0]
    textarea_label = st.text_area.call_args.args[0]
    assert isinstance(uploader_label, str) and uploader_label.strip()
    assert isinstance(textarea_label, str) and textarea_label.strip()
    assert st.file_uploader.call_args.kwargs.get("label_visibility", "visible") != "collapsed"
    assert st.text_area.call_args.kwargs.get("label_visibility", "visible") != "collapsed"


# --- FlagsPanel visibility rule + lists flags/missing ---------


def test_flags_panel_visibility_rule() -> None:
    # Visibility gates on flags alone — omitted keywords now ride a dedicated backend flag.
    assert app.should_show_flags([]) is False
    assert app.should_show_flags([{"message": "x"}]) is True


def test_flags_panel_lists_each_message_as_its_own_item_with_heading() -> None:
    st = MagicMock()
    # the omitted keywords arrive as a single backend "Missing (no CV evidence …)" flag
    flags = [
        {"section": "global", "kind": "did_not_converge", "message": "Coverage 40% …"},
        {
            "section": "global",
            "kind": "unmet_coverage",
            "message": "Missing (no CV evidence — omitted, not fabricated): Kubernetes",
        },
    ]
    app.render_flags(st, flags)
    # A scannable panel: a clear heading, then each flag as its OWN item — never one blob.
    headings = [call.args[0] for call in st.subheader.call_args_list]
    assert "Recommendations" in headings
    items = [call.args[0] for call in st.warning.call_args_list]
    items += [call.args[0] for call in st.info.call_args_list]
    assert len(items) == len(flags)  # one rendered item per flag, not a concatenated string
    text = _all_text(st.warning) + " " + _all_text(st.info)
    assert "Coverage 40%" in text
    assert "Kubernetes" in text


def test_flags_panel_splits_warnings_and_notes_by_kind() -> None:
    # `kind` is present (backend SectionFlag.kind survives into the response), so the panel
    # groups actionable issues into a warning treatment and informational flags into a note one.
    st = MagicMock()
    flags = [
        {"section": "global", "kind": "unmet_coverage", "message": "Coverage shortfall."},
        {"section": "global", "kind": "cover_letter_no_requirements", "message": "No must-haves."},
    ]
    app.render_flags(st, flags)
    assert "Coverage shortfall." in [call.args[0] for call in st.warning.call_args_list]
    assert "No must-haves." in [call.args[0] for call in st.info.call_args_list]


def test_flags_panel_renders_message_verbatim() -> None:
    # Messages are rendered exactly as the backend emits them — no rewording, no prefix stripping.
    st = MagicMock()
    verbatim = "Missing (no CV evidence — omitted, not fabricated): Kubernetes"
    app.render_flags(st, [{"section": "global", "kind": "unmet_coverage", "message": verbatim}])
    assert st.warning.call_args.args[0] == verbatim


def test_flags_panel_hidden_when_no_gaps() -> None:
    st = MagicMock()
    app.render_flags(st, [])
    st.warning.assert_not_called()
    st.info.assert_not_called()
    st.subheader.assert_not_called()


# --- no download/export and no in-UI edit widgets -------------


def test_result_area_has_no_download_or_edit_widgets() -> None:
    st = MagicMock()
    result = {"cv": _sample_cv(), "cover_letter": "Dear team", "ats_score": _ats(), "flags": []}
    app.render_result(st, result)
    st.download_button.assert_not_called()
    st.text_input.assert_not_called()
    st.text_area.assert_not_called()  # JD textarea lives in the input form, never the result
    st.data_editor.assert_not_called()


def test_error_banner_has_no_download_or_edit_widgets() -> None:
    st = MagicMock()
    app.render_error(st, "Unsupported file type.")
    st.error.assert_called_once()
    st.download_button.assert_not_called()
    st.data_editor.assert_not_called()


# --- the error banner shows plain language, never the internal pipeline stage ------


def test_error_banner_does_not_show_internal_stage() -> None:
    st = MagicMock()
    app.render_error(st, "Could not reach the server. Is the backend running?")
    shown = " ".join(str(a) for a in st.error.call_args.args)
    assert "stage" not in shown.lower()


# --- an empty-but-listed section is skipped, not rendered as a "—" placeholder --


def test_empty_section_in_section_order_is_skipped_not_placeholder() -> None:
    # _section_order always includes projects/experience/education even when empty
    # (cv_generator._section_order). An empty section must render NO heading and no "—".
    st = MagicMock()
    cv = _sample_cv()
    cv["section_order"] = ["contact", "summary", "projects", "experience"]
    cv["projects"] = []
    cv["experiences"] = []
    app.render_tailored_cv(st, cv)
    headings = [call.args[0] for call in st.subheader.call_args_list]
    assert "Projects" not in headings
    assert "Experience" not in headings
    assert "Contact" in headings and "Summary" in headings
    assert "—" not in [call.args[0] for call in st.code.call_args_list]


# --- the copied text renders every field the ATS scorer counts (score == copied text) ----


def test_rendered_cv_includes_every_ats_scored_field() -> None:
    # cv_generator._cv_to_text scores company_description, bullet.benefit, summary.text and
    # project.skills — they MUST appear in the copy-friendly text, or the displayed coverage
    # overstates the resume the user actually submits. summary.relevant_skills and
    # bullet.skills are NOT scored (structured tags, not rendered prose), so neither
    # may leak into the rendered copy.
    st = MagicMock()
    cv = _sample_cv()
    cv["section_order"] = ["summary", "experience", "projects"]
    cv["summary"] = {"text": "Engineer fluent in KW_SUMMARY.", "relevant_skills": ["KW_NOTSCORED"]}
    cv["experiences"] = [
        {
            "role": "Eng",
            "company": "Acme",
            "company_description": "KW_COMPANYDESC fintech platform",
            "start_date": "2020",
            "end_date": "2022",
            "location": None,
            "bullets": [
                {
                    "action_verb": "Built",
                    "description": "services",
                    "skills": ["KW_BULLETSKILL"],
                    "impact": "+10%",
                    "benefit": "KW_BENEFIT result",
                }
            ],
        }
    ]
    cv["projects"] = [
        {"name": "Proj", "description": "desc", "skills": ["KW_PROJSKILL"], "link": None}
    ]
    app.render_tailored_cv(st, cv)
    rendered = _all_text(st.code)
    for token in ("KW_SUMMARY", "KW_COMPANYDESC", "KW_BENEFIT", "KW_PROJSKILL"):
        assert token in rendered, f"{token} is scored by ATS but missing from the copied text"
    # relevant_skills is no longer rendered (or scored) — no duplicate "Relevant skills:" line.
    assert "KW_NOTSCORED" not in rendered
    assert "Relevant skills:" not in rendered
    # bullet.skills are structured tags — no longer scored, so not rendered (render == score).
    assert "KW_BULLETSKILL" not in rendered


def test_experience_text_omits_parens_when_both_dates_absent() -> None:
    # The Optional refactor: with null start/end dates the header drops the date parentheses
    # entirely and never renders the literal string "None".
    cv = {
        "experiences": [
            {
                "role": "Head Chef",
                "company": "Le Bernardin",
                "company_description": "",
                "start_date": None,
                "end_date": None,
                "bullets": [],
            }
        ]
    }

    text = app._experience_text(cv)

    assert "Head Chef" in text and "Le Bernardin" in text
    assert "None" not in text  # a null date never leaks as the literal "None"
    assert "(" not in text and ")" not in text  # no empty date parentheses


# --- Skills renders as plain canonical category lines, no markdown --------------


def test_skills_renders_plain_category_lines_in_order() -> None:
    cv = {
        "skills": {
            "categories": [
                {"category": "Languages", "keywords": ["JavaScript", "TypeScript"]},
                {"category": "Tools & Platforms", "keywords": ["Webpack", "Jest"]},
                {"category": "Spoken Languages", "keywords": ["English fluent"]},
            ]
        }
    }
    text = app._skills_text(cv)
    # Plain text only — the `**header**` markdown leak is gone; `st.code` shows/copies verbatim.
    assert text.splitlines() == [
        "Languages: JavaScript, TypeScript",
        "Tools & Platforms: Webpack, Jest",
        "Spoken Languages: English fluent",
    ]
    # the fixed Relevant/Hard/Soft bucket labels are gone
    for legacy in ("Relevant:", "Hard:", "Soft:"):
        assert legacy not in text


def test_skills_render_carries_no_markdown_in_rendered_or_copied_text() -> None:
    # Regression re-pin (plain-text invariant): the rendered AND copied Skills string — the
    # single source `st.code` both displays and hands to the clipboard — contains ZERO markdown.
    cv = {
        "skills": {
            "categories": [
                {"category": "Testing & QA", "keywords": ["SDET", "Test Automation"]},
                {"category": "Practices & Concepts", "keywords": ["Systems Thinking"]},
            ]
        }
    }
    text = app._skills_text(cv)
    for control in ("*", "_", "`", "#"):
        assert control not in text, f"markdown control {control!r} leaked into Skills text"


def test_skills_render_skips_empty_categories() -> None:
    cv = {
        "skills": {
            "categories": [
                {"category": "Languages", "keywords": []},
                {"category": "Tools & Platforms", "keywords": ["Docker"]},
            ]
        }
    }
    assert app._skills_text(cv) == "Tools & Platforms: Docker"


def test_skills_render_ungrouped_emits_bare_keyword_line_no_prefix() -> None:
    # De-bias: an empty (ungrouped) emergent header — the domain-universal fallback — renders the
    # keywords as a bare line with NO invented "Header:" prefix and no stray leading colon.
    cv = {
        "skills": {
            "categories": [
                {"category": "", "keywords": ["Python", "FastAPI"]},
                {"category": "Clinical Skills", "keywords": ["Triage"]},
            ]
        }
    }
    assert app._skills_text(cv).splitlines() == [
        "Python, FastAPI",
        "Clinical Skills: Triage",
    ]


def test_skills_render_shows_no_mobile_noise_for_sdet_fixture() -> None:
    # Mirrors the generation fixture: the rendered Skills shows canonical domain headers and the
    # evidenced AI/SDET keywords, with no mobile noise (the generator already excluded it).
    cv = {
        "skills": {
            "categories": [
                {"category": "Languages", "keywords": ["Python"]},
                {"category": "Testing & QA", "keywords": ["Selenium", "Test Automation"]},
                {"category": "Practices & Concepts", "keywords": ["CI/CD"]},
            ]
        }
    }
    text = app._skills_text(cv)
    assert "Testing & QA: Selenium, Test Automation" in text
    for noise in ("Swift", "Kotlin", "Flutter", "Mobile Automation"):
        assert noise not in text


# --- a near-zero negative delta renders as 0, not a "-0.0%" red down-arrow ------


def test_ats_metric_near_zero_negative_delta_is_not_negative() -> None:
    st = MagicMock()
    app.render_ats_panel(st, _ats(before=80.04, after=80.0))  # round(-0.04, 1) == -0.0
    delta = str(st.metric.call_args.kwargs["delta"])
    assert not delta.startswith("-")
    assert delta == "+0.0%"


# --- honest zero-keyword panel (no false 100% when nothing is scored) ----


def test_ats_panel_zero_keywords_shows_message_not_percentage() -> None:
    # When nothing was scored (matched empty AND missing empty), the internal coverage is a
    # vacuous 100% — never render it as a score. Show a plain-language line instead.
    st = MagicMock()
    ats = _ats(before=100.0, after=100.0)
    ats["matched"] = []
    ats["missing"] = []
    app.render_ats_panel(st, ats)

    st.metric.assert_not_called()  # no percentage metric / green delta
    text = _all_text(st.markdown)
    assert "100%" not in text
    assert "no keyword" in text.lower()


def test_ats_panel_with_keywords_renders_metric_normally() -> None:
    # The normal path is unchanged when there ARE keywords to score.
    st = MagicMock()
    app.render_ats_panel(st, _ats(before=40.0, after=80.0))
    st.metric.assert_called_once()
    assert "80" in str(st.metric.call_args.kwargs["value"])
