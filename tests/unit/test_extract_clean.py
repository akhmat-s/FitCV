"""Unit tests for char-cleaning the extract inputs before extraction.

Parsed CV text and the pasted JD are normalized (homoglyph→Latin, invisibles
stripped) via ``TextPreprocessing.clean`` *before* the function-calling extraction,
so any downstream keyword matching is computed on normalized text — ``match(normalize(text))``.
The OpenRouter client is mocked; the text actually handed to ``call_tool`` is captured
and asserted, which is the verifiable condition.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from extract import build_extract
from helprers.text_preprocessing import TextPreprocessing
from schemas import ExtractResult

_CV_TOOL_JSON = {
    "personal_info": {
        "name": "Ada Lovelace",
        "location": "London, UK",
        "email": "ada@example.com",
    },
}

_JD_TOOL_JSON = {
    "role_title": "Senior Backend Engineer",
    "company": "Globex",
    "keywords": ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
    "requirements_must": [],
    "requirements_nice": [],
    "keyword_plan": {},
    "candidate_level": "senior_ic",
}

# Cyrillic homoglyphs that look like Latin letters: с=U+0441, о=U+043E, р=U+0440, е=U+0435.
# "sсоре" reads as "scope" but the middle four letters are Cyrillic look-alikes.
_HOMOGLYPH_WORD = "sсоре"  # s + с + о + р + е  → "scope"
_LATIN_WORD = "scope"
# A zero-width space (U+200B) embedded mid-text; an invisible the cleaner must strip.
_INVISIBLE = "​"


def _cv_text_passed_to(mock_llm: MagicMock) -> str:
    """Return the CV ``user_prompt`` text handed to the first ``call_tool`` call."""
    # call_tool(system_prompt, user_prompt, tool_schema); the CV-facts call is first.
    cv_call = mock_llm.call_tool.call_args_list[0]
    return cv_call.args[1]


def _jd_text_passed_to(mock_llm: MagicMock) -> str:
    """Return the JD ``user_prompt`` text handed to the second ``call_tool`` call."""
    jd_call = mock_llm.call_tool.call_args_list[1]
    return jd_call.args[1]


async def test_cv_text_is_normalized_before_extraction(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    dirty_cv = f"Skilled in {_HOMOGLYPH_WORD}{_INVISIBLE} design"
    cv_bytes = dirty_cv.encode("utf-8")

    result = await build_extract(cv_bytes, "cv.txt", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ExtractResult)
    cv_text = _cv_text_passed_to(mock_llm)
    # Latin form present; Cyrillic homoglyphs and the zero-width invisible are gone.
    assert _LATIN_WORD in cv_text
    assert "с" not in cv_text and "о" not in cv_text and "р" not in cv_text
    assert _INVISIBLE not in cv_text


async def test_keyword_matching_runs_on_normalized_text(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    dirty_jd = f"We need {_HOMOGLYPH_WORD} skills and {_INVISIBLE}testing"
    cv_bytes = b"Ada Lovelace Analyst"

    await build_extract(cv_bytes, "cv.txt", dirty_jd, llm=mock_llm)

    jd_text = _jd_text_passed_to(mock_llm)
    # The JD that drives keyword analysis is the normalized text, not the raw input.
    assert _LATIN_WORD in jd_text
    assert _HOMOGLYPH_WORD not in jd_text
    assert _INVISIBLE not in jd_text


def test_normalization_makes_homoglyph_match_latin() -> None:
    # The match(normalize(text)) guarantee: a homoglyph keyword only equals its Latin
    # form after normalization, so matching must be computed on normalized text.
    assert _HOMOGLYPH_WORD != _LATIN_WORD
    assert TextPreprocessing.clean(_HOMOGLYPH_WORD) == _LATIN_WORD


_CYRILLIC_NAME = "Сергей Иванов"


def test_normalize_input_preserves_letter_spaced_header_and_newlines() -> None:
    text = "S K I L L S\nSenior Backend Engineer\nLed payment team"

    result = TextPreprocessing.normalize_input(text)

    # Words and newlines are intact — no whole-document whitespace collapse.
    assert "Senior Backend Engineer" in result
    assert "Led payment team" in result
    assert result.count("\n") == text.count("\n")


def test_normalize_input_preserves_legitimate_cyrillic_name() -> None:
    # Truth-preservation: an all-Cyrillic name is NOT Latinized.
    assert TextPreprocessing.normalize_input(_CYRILLIC_NAME) == _CYRILLIC_NAME


def test_normalize_input_strips_invisible_characters() -> None:
    result = TextPreprocessing.normalize_input(f"Skilled{_INVISIBLE} engineer")

    assert _INVISIBLE not in result
    assert "Skilled engineer" in result


def test_normalize_input_maps_homoglyph_spoof_inside_mixed_word() -> None:
    # A Latin word spoofed with Cyrillic look-alikes IS normalized to Latin.
    assert TextPreprocessing.normalize_input(_HOMOGLYPH_WORD) == _LATIN_WORD


async def test_build_extract_does_not_collapse_letter_spaced_cv(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    cv = "S K I L L S\nSenior Backend Engineer\nLed payment team at Globex"

    await build_extract(cv.encode("utf-8"), "cv.txt", "Senior Python engineer", llm=mock_llm)

    cv_text = _cv_text_passed_to(mock_llm)
    # The destructive whitespace-collapse heuristic must NOT have fired on real input.
    assert "Senior Backend Engineer" in cv_text
    assert "Led payment team at Globex" in cv_text
