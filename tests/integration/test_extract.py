"""Integration tests for the extract pass (extract.py).

The OpenRouter client is injected via the conftest ``mock_llm`` seam — its
``call_tool`` returns the tool-call dict the model would have produced, so the
extract functions are exercised end-to-end without any network access.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from unittest.mock import MagicMock

import docx  # python-docx
import fitz  # PyMuPDF
import httpx
import openai
import pytest

import extract
from extract import analyze_jd, build_extract, extract_cv_facts
from helprers.llm_model import ProviderResponseError
from schemas import (
    MAX_UPLOAD_MB,
    MIN_KEYWORDS,
    CandidateLevel,
    CVFacts,
    ErrorResponse,
    ErrorStage,
    ExtractResult,
    JdAnalysis,
    TargetSection,
)

_CV_TOOL_JSON = {
    "personal_info": {
        "name": "Ada Lovelace",
        "location": "London, UK",
        "email": "ada@example.com",
        "links": [{"title": "LinkedIn", "url": "https://linkedin.com/in/ada"}],
    },
    "experiences": [
        {
            "role": "Analyst",
            "company": "Analytical Engines",
            "start_date": "1840",
            "end_date": "Present",
            "bullets": ["Wrote the first algorithm"],
        }
    ],
    "languages": [{"language": "English", "level": "Native"}],
}

_JD_TOOL_JSON = {
    "role_title": "Senior Backend Engineer",
    "company": "Globex",
    "keywords": ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
    "requirements_must": ["5y Python"],
    "requirements_nice": ["GraphQL"],
    "keyword_plan": {"Python": "skills", "FastAPI": "experience"},
    "candidate_level": "senior_ic",
}


def _make_pdf_bytes(text: str) -> bytes:
    """Build a single-page PDF carrying ``text`` entirely in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _make_docx_bytes(text: str) -> bytes:
    """Build a one-paragraph DOCX carrying ``text`` entirely in memory."""
    document = docx.Document()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _make_image_only_pdf_bytes() -> bytes:
    """Build a valid PDF with a blank page (parses fine, yields no extractable text).

    Stands in for an image-only PDF: the document is well-formed so the parser does
    not error, but ``get_text`` returns nothing — exercising the no-text-after-parse
    guard rather than the corrupt-file guard.
    """
    doc = fitz.open()
    doc.new_page()
    data: bytes = doc.tobytes()
    doc.close()
    return data


async def test_extract_cv_facts_returns_validated_cv_facts(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.return_value = _CV_TOOL_JSON

    facts = await extract_cv_facts("Ada Lovelace - Analyst at Analytical Engines", llm=mock_llm)

    assert isinstance(facts, CVFacts)
    assert facts.personal_info.name == "Ada Lovelace"
    assert facts.experiences[0].company == "Analytical Engines"
    assert facts.languages[0].level == "Native"
    assert mock_llm.call_tool.call_count == 1


async def test_analyze_jd_returns_validated_jd_analysis(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.return_value = _JD_TOOL_JSON

    jd = await analyze_jd("We need a senior Python backend engineer", llm=mock_llm)

    assert isinstance(jd, JdAnalysis)
    assert jd.role_title == "Senior Backend Engineer"
    assert jd.candidate_level is CandidateLevel.SENIOR_IC
    assert jd.keyword_plan["FastAPI"] is TargetSection.EXPERIENCE
    assert len(jd.keywords) == 5
    assert mock_llm.call_tool.call_count == 1


async def test_build_extract_returns_extract_result(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst Analytical Engines")

    result = await build_extract(
        pdf_bytes,
        "cv.pdf",
        "Senior Python backend engineer at Globex",
        llm=mock_llm,
    )

    assert isinstance(result, ExtractResult)
    assert result.facts.personal_info.name == "Ada Lovelace"
    assert result.jd.role_title == "Senior Backend Engineer"
    assert result.flags == []


async def test_build_extract_makes_two_tool_calls(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst")

    await build_extract(pdf_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    # One CV-facts call + one JD-analysis call.
    assert mock_llm.call_tool.call_count == 2


async def test_build_extract_same_facts_across_formats(mock_llm: MagicMock) -> None:
    cv_text = "Ada Lovelace Analyst Analytical Engines"
    jd_text = "Senior Python backend engineer at Globex"
    sources = {
        "cv.pdf": _make_pdf_bytes(cv_text),
        "cv.docx": _make_docx_bytes(cv_text),
        "cv.txt": cv_text.encode("utf-8"),
    }

    facts_per_format = {}
    for filename, cv_bytes in sources.items():
        mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
        result = await build_extract(cv_bytes, filename, jd_text, llm=mock_llm)
        assert isinstance(result, ExtractResult)
        facts_per_format[filename] = result.facts

    pdf_facts = facts_per_format["cv.pdf"]
    assert facts_per_format["cv.docx"] == pdf_facts
    assert facts_per_format["cv.txt"] == pdf_facts


_JD_FULL_PLAN_JSON = {
    "role_title": "Senior Backend Engineer",
    "company": "Globex",
    "keywords": ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
    "requirements_must": ["5y Python"],
    "requirements_nice": ["GraphQL"],
    "keyword_plan": {
        "Python": "skills",
        "FastAPI": "experience",
        "Pydantic": "skills",
        "PostgreSQL": "skills",
        "Docker": "projects",
    },
    "candidate_level": "senior_ic",
}


async def test_analyze_jd_maps_every_keyword_to_target_section(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.return_value = _JD_FULL_PLAN_JSON

    jd = await analyze_jd("Senior Python backend engineer at Globex", llm=mock_llm)

    assert len(jd.keywords) >= MIN_KEYWORDS
    # Every keyword has a plan entry and each value is a TargetSection member.
    assert set(jd.keyword_plan) == set(jd.keywords)
    assert all(isinstance(section, TargetSection) for section in jd.keyword_plan.values())
    assert isinstance(jd.candidate_level, CandidateLevel)
    assert jd.candidate_level is CandidateLevel.SENIOR_IC


_JD_FEW_KEYWORDS_JSON = {
    "role_title": "Junior Engineer",
    "company": "Initech",
    "keywords": ["Python", "SQL"],
    "requirements_must": [],
    "requirements_nice": [],
    "keyword_plan": {"Python": "skills", "SQL": "skills"},
    "candidate_level": "entry",
}


async def test_build_extract_appends_keyword_gap_flag(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_FEW_KEYWORDS_JSON]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst")

    result = await build_extract(pdf_bytes, "cv.pdf", "Junior engineer at Initech", llm=mock_llm)

    # Still a full ExtractResult — the gap is non-fatal.
    assert isinstance(result, ExtractResult)
    assert len(result.flags) == 1
    assert "keyword-gap" in result.flags[0]
    # Never fabricates keywords to reach the threshold.
    assert len(result.jd.keywords) == 2
    assert result.jd.keywords == ["Python", "SQL"]


async def test_build_extract_no_flag_when_keywords_meet_threshold(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst")

    result = await build_extract(pdf_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ExtractResult)
    assert len(result.jd.keywords) >= MIN_KEYWORDS
    assert result.flags == []


_PROVIDER_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


async def test_build_extract_network_failure_returns_retryable_extract_error(
    mock_llm: MagicMock,
) -> None:
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    original_cv_bytes = bytes(cv_bytes)
    job_description = "Senior Python engineer"
    mock_llm.call_tool.side_effect = openai.APIConnectionError(request=_PROVIDER_REQUEST)

    result = await build_extract(cv_bytes, "cv.pdf", job_description, llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    assert "reach" in result.error.lower() or "connection" in result.error.lower()
    # Inputs are left intact (stateless, non-destructive).
    assert cv_bytes == original_cv_bytes
    assert job_description == "Senior Python engineer"
    # Retryable: a subsequent healthy call on the same inputs succeeds.
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    retry = await build_extract(cv_bytes, "cv.pdf", job_description, llm=mock_llm)
    assert isinstance(retry, ExtractResult)


async def test_build_extract_timeout_returns_extract_error(mock_llm: MagicMock) -> None:
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    mock_llm.call_tool.side_effect = openai.APITimeoutError(request=_PROVIDER_REQUEST)

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    assert "too long" in result.error.lower() or "timed out" in result.error.lower()


async def test_build_extract_auth_error_is_redacted_extract_error(mock_llm: MagicMock) -> None:
    api_key = "sk-or-test-dummy"  # matches conftest _no_live_credentials
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    auth_error = openai.AuthenticationError(
        message=f"401 Unauthorized: Bearer {api_key} is invalid",
        response=httpx.Response(401, request=_PROVIDER_REQUEST),
        body=None,
    )
    mock_llm.call_tool.side_effect = auth_error

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    # permission_denied framing — distinct from network/timeout.
    assert "configuration" in result.error.lower() or "rejected" in result.error.lower()
    # The API key never appears in the surfaced error.
    assert api_key not in result.error


async def test_build_extract_auth_error_never_leaks_a_nonenv_api_key(mock_llm: MagicMock) -> None:
    # A key passed to the client constructor (not the env var) must still never surface,
    # even if the provider echoes it verbatim in the exception string.
    leaked_key = "sk-or-constructor-only-SECRET"
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    auth_error = openai.AuthenticationError(
        message=f"401 Unauthorized: Bearer {leaked_key} is invalid",
        response=httpx.Response(401, request=_PROVIDER_REQUEST),
        body=None,
    )
    mock_llm.call_tool.side_effect = auth_error

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    assert leaked_key not in result.error


async def test_build_extract_rate_limit_returns_retryable_extract_error(
    mock_llm: MagicMock,
) -> None:
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    rate_limited = openai.RateLimitError(
        message="429 Too Many Requests",
        response=httpx.Response(429, request=_PROVIDER_REQUEST),
        body=None,
    )
    mock_llm.call_tool.side_effect = rate_limited

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    # A 429 gets a distinct, actionable message — not the "check your connection" copy.
    assert "busy" in result.error.lower() or "rate" in result.error.lower()


async def test_build_extract_server_error_returns_retryable_extract_error(
    mock_llm: MagicMock,
) -> None:
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    server_error = openai.InternalServerError(
        message="503 Service Unavailable",
        response=httpx.Response(503, request=_PROVIDER_REQUEST),
        body=None,
    )
    mock_llm.call_tool.side_effect = server_error

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    # A 5xx is a service-side error, not a local connection problem.
    assert "service" in result.error.lower() and "connection" not in result.error.lower()


async def test_build_extract_schema_violating_response_returns_extract_error(
    mock_llm: MagicMock,
) -> None:
    # The model returns a tool call that omits required CVFacts fields. The resulting
    # pydantic ValidationError must become a stage=extract ErrorResponse, not a crash.
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    mock_llm.call_tool.side_effect = [{"personal_info": {}}, _JD_TOOL_JSON]

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT


async def test_build_extract_malformed_provider_response_returns_extract_error(
    mock_llm: MagicMock,
) -> None:
    # call_tool raises ProviderResponseError when the provider returns no usable tool
    # call; build_extract must translate it to a stage=extract ErrorResponse.
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    mock_llm.call_tool.side_effect = ProviderResponseError("no tool call returned")

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT


async def test_build_extract_analyzes_jd_exactly_once(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, _JD_TOOL_JSON]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst Analytical Engines")

    spy = MagicMock(wraps=extract.analyze_jd)
    monkeypatch.setattr(extract, "analyze_jd", spy)

    result = await build_extract(
        pdf_bytes,
        "cv.pdf",
        "Senior Python backend engineer at Globex",
        llm=mock_llm,
    )

    # A single shared ExtractResult, the JD analyzed exactly once.
    assert isinstance(result, ExtractResult)
    assert spy.call_count == 1
    # The one shared analysis is the object carried on the result (no re-parse).
    assert result.jd.role_title == "Senior Backend Engineer"


async def test_build_extract_corrupt_file_halts_at_parse(mock_llm: MagicMock) -> None:
    # Well-formed PDF header followed by garbage — fitz cannot open the stream.
    corrupt_bytes = b"%PDF-1.4\n" + b"not a real pdf, just noise " * 8
    mock_llm.call_tool.side_effect = AssertionError("provider must not be called on parse error")

    result = await build_extract(corrupt_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    # Actionable: tells the user what to do.
    assert "upload" in result.error.lower() or "read" in result.error.lower()
    # Halts before extract — the JD is never analyzed.
    assert mock_llm.call_tool.call_count == 0


async def test_build_extract_empty_file_is_bad_file_parse_error(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = AssertionError("provider must not be called on parse error")

    result = await build_extract(b"", "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    assert "empty" in result.error.lower()
    assert mock_llm.call_tool.call_count == 0


async def test_build_extract_oversized_file_rejected_before_parse(mock_llm: MagicMock) -> None:
    mock_llm.call_tool.side_effect = AssertionError("provider must not be called on parse error")
    oversized = b"Ada Lovelace Analyst " + b"x" * (MAX_UPLOAD_MB * 1024 * 1024)

    result = await build_extract(oversized, "cv.txt", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    assert "large" in result.error.lower() or "too big" in result.error.lower()
    # Rejected before parse: the provider is never reached.
    assert mock_llm.call_tool.call_count == 0


async def test_build_extract_image_only_pdf_returns_no_text_parse_error(
    mock_llm: MagicMock,
) -> None:
    image_only_pdf = _make_image_only_pdf_bytes()
    mock_llm.call_tool.side_effect = AssertionError("provider must not be called on parse error")

    result = await build_extract(image_only_pdf, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    # Not empty facts — an actionable no-text parse error.
    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    assert "text" in result.error.lower()
    # No extraction attempted on empty text.
    assert mock_llm.call_tool.call_count == 0


async def test_build_extract_unset_key_parse_error_never_constructs_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # llm=None forces real LLMModel() construction; for an empty upload the parse guard
    # must return BEFORE any client is built, so no openai.OpenAIError is raised.
    result = await build_extract(b"", "cv.pdf", "Senior Python engineer")

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    assert "empty" in result.error.lower()


async def test_build_extract_unset_key_valid_input_returns_config_extract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst")

    # No llm injected: a real LLMModel() is built inside the try; with no key the openai
    # SDK raises openai.OpenAIError (base, not APIError), which must translate cleanly.
    result = await build_extract(pdf_bytes, "cv.pdf", "Senior Python engineer")

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    assert "configuration" in result.error.lower() or "rejected" in result.error.lower()


async def test_build_extract_corrupt_docx_halts_at_parse(mock_llm: MagicMock) -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("hello.txt", "not a docx")
    bad_docx = buffer.getvalue()
    mock_llm.call_tool.side_effect = AssertionError("provider must not be called on parse error")

    result = await build_extract(bad_docx, "cv.docx", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    assert "docx" in result.error.lower() or "corrupt" in result.error.lower()
    assert mock_llm.call_tool.call_count == 0


async def test_build_extract_bad_request_400_is_nonretryable_extract_error(
    mock_llm: MagicMock,
) -> None:
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    bad_request = openai.BadRequestError(
        message="400 context_length_exceeded",
        response=httpx.Response(400, request=_PROVIDER_REQUEST),
        body=None,
    )
    mock_llm.call_tool.side_effect = bad_request

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    assert "check your connection" not in result.error.lower()
    assert "try again" not in result.error.lower()


async def test_build_extract_not_found_404_is_nonretryable_extract_error(
    mock_llm: MagicMock,
) -> None:
    cv_bytes = _make_pdf_bytes("Ada Lovelace Analyst")
    not_found = openai.NotFoundError(
        message="404 model not found",
        response=httpx.Response(404, request=_PROVIDER_REQUEST),
        body=None,
    )
    mock_llm.call_tool.side_effect = not_found

    result = await build_extract(cv_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.EXTRACT
    assert "check your connection" not in result.error.lower()
    assert "try again" not in result.error.lower()


async def test_build_extract_duplicate_keywords_still_flag_gap(mock_llm: MagicMock) -> None:
    duped_jd = dict(_JD_TOOL_JSON, keywords=["Python", "python", "PYTHON"], keyword_plan={})
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, duped_jd]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst")

    result = await build_extract(pdf_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ExtractResult)
    # Case-insensitive dedup leaves one distinct keyword → below MIN_KEYWORDS → gap flag.
    assert result.jd.keywords == ["Python"]
    assert len(result.flags) == 1
    assert "keyword-gap" in result.flags[0]


async def test_build_extract_blank_keywords_dropped_and_flag_gap(mock_llm: MagicMock) -> None:
    blank_jd = dict(_JD_TOOL_JSON, keywords=["Python", "", "   ", ""], keyword_plan={})
    mock_llm.call_tool.side_effect = [_CV_TOOL_JSON, blank_jd]
    pdf_bytes = _make_pdf_bytes("Ada Lovelace Analyst")

    result = await build_extract(pdf_bytes, "cv.pdf", "Senior Python engineer", llm=mock_llm)

    assert isinstance(result, ExtractResult)
    assert result.jd.keywords == ["Python"]
    assert len(result.flags) == 1
    assert "keyword-gap" in result.flags[0]
