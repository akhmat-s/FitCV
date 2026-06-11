"""Shared extract pass: parse a CV + analyze a JD into one `ExtractResult`.

`build_extract(cv_bytes, filename, job_description) -> ExtractResult | ErrorResponse`
orchestrates parse -> normalize-input -> extract_cv_facts() + analyze_jd() -> assemble +
keyword-gap flag. This module is internal (no HTTP route).

The two function-calling tool schemas live in `tool_schemas.py` (hand-built
flat shape, no $defs/$ref) and are re-exported here for the extract calls.
"""

from __future__ import annotations

import asyncio
import logging
import os

import openai
from pydantic import ValidationError

from helprers.llm_model import LLMModel, ProviderResponseError
from helprers.pdf_parser import CorruptFileError, UnsupportedFormatError, extract_text
from helprers.prompts import Prompts
from helprers.text_preprocessing import TextPreprocessing
from schemas import (
    MAX_UPLOAD_MB,
    MIN_KEYWORDS,
    CVFacts,
    ErrorResponse,
    ErrorStage,
    ExtractResult,
    JdAnalysis,
)
from tool_schemas import analyze_jd_tool_schema, cv_facts_tool_schema

logger = logging.getLogger(__name__)

#: Placeholder used in messages where a leaked secret would otherwise appear.
_REDACTED = "***REDACTED***"

#: User-facing provider-error messages. None embed the raw
#: provider exception, so the API key can never leak through them regardless of which
#: key the client used; ``to_error_response`` redaction stays as defense-in-depth.
_NETWORK_ERROR = "Couldn't reach the analysis service. Check your connection and try again."
_TIMEOUT_ERROR = "Analysis took too long and was stopped. Try again."
_AUTH_ERROR = "The analysis service rejected the request (server configuration)."
_RATE_LIMIT_ERROR = "The analysis service is busy (rate limited). Wait a moment and try again."
_SERVER_ERROR = "The analysis service had an error. Try again shortly."
_BAD_RESPONSE_ERROR = "The analysis service returned an unreadable response. Try again."

#: Non-retryable client-side error (4xx): the request itself won't succeed on retry
#: (e.g. input too long, or a misconfigured model). Deliberately omits "try again".
_CLIENT_ERROR = (
    "The analysis service couldn't process this request. The input may be too long, "
    "or the service may be misconfigured."
)

#: User-facing parse-stage message for a zero-byte upload.
_EMPTY_FILE_ERROR = "This file is empty. Upload a text-based PDF, DOCX, or TXT."

#: User-facing parse-stage message for an oversized upload.
_OVERSIZED_FILE_ERROR = (
    f"This file is too large (max {MAX_UPLOAD_MB} MB). Upload a smaller PDF, DOCX, or TXT."
)

#: User-facing parse-stage message when a file parses but yields no text
#: (e.g. an image-only PDF): a no-text error, not empty facts.
_NO_TEXT_ERROR = (
    "Couldn't read any text from this file. Upload a text-based PDF, DOCX, or TXT "
    "— image-only PDFs can't be parsed."
)


def _redact_api_key(message: str) -> str:
    """Strip the OpenRouter API key from any error text before it is surfaced.

    The key never reaches the client. Provider libraries can echo the bearer
    token in exception strings, so we defensively replace the live key value if
    it appears in the message.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and api_key in message:
        message = message.replace(api_key, _REDACTED)
    return message


def to_error_response(stage: ErrorStage, message: str) -> ErrorResponse:
    """Build an actionable, key-redacted `ErrorResponse` for a halted stage."""
    return ErrorResponse(error=_redact_api_key(message), stage=stage)


async def extract_cv_facts(text: str, llm: LLMModel | None = None) -> CVFacts:
    """Extract structured `CVFacts` from char-cleaned CV text.

    Args:
        text: The CV text to analyze.
        llm: Injectable function-calling client; defaults to a real ``LLMModel``.
    """
    client = llm or LLMModel()
    raw = await client.call_tool(
        Prompts.EXTRACT_CV_FACTS_SYSTEM,
        Prompts.EXTRACT_CV_FACTS_USER.format(cv_text=text),
        cv_facts_tool_schema(),
    )
    return CVFacts.model_validate(raw)


async def analyze_jd(text: str, llm: LLMModel | None = None) -> JdAnalysis:
    """Analyze a job description into a validated `JdAnalysis`.

    Args:
        text: The raw job-description text.
        llm: Injectable function-calling client; defaults to a real ``LLMModel``.
    """
    client = llm or LLMModel()
    raw = await client.call_tool(
        Prompts.ANALYZE_JD_SYSTEM,
        Prompts.ANALYZE_JD_USER.format(jd_text=text),
        analyze_jd_tool_schema(),
    )
    return JdAnalysis.model_validate(raw)


def _dedup_keywords(keywords: list[str]) -> list[str]:
    """Drop blanks and case-insensitive duplicates, preserving first-seen order.

    The model can return repeated or empty keywords; counting them raw would
    defeat the honest-gap flag and inflate the ATS keyword count. We clean once,
    before BOTH the gap check and ``to_job_target()`` feeds ATS scoring.
    """
    seen: set[str] = set()
    deduped: list[str] = []
    for keyword in keywords:
        cleaned = keyword.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(cleaned)
    return deduped


def _keyword_gap_flags(jd: JdAnalysis) -> list[str]:
    """Return a non-fatal keyword-gap flag list when the JD yields too few keywords.

    When the model honestly returns fewer than ``MIN_KEYWORDS`` keywords, surface
    the shortfall as a non-fatal flag instead of fabricating keywords to reach the
    threshold. ``>= MIN_KEYWORDS`` yields no flag. Operates on the deduped keyword
    list (see ``_dedup_keywords``).
    """
    keyword_count = len(jd.keywords)
    if keyword_count >= MIN_KEYWORDS:
        return []
    return [
        f"keyword-gap: only {keyword_count} job-description keyword(s) found "
        f"(target {MIN_KEYWORDS}); none were fabricated."
    ]


def _provider_error(exc: Exception) -> ErrorResponse:
    """Translate an extract-stage failure into an actionable ErrorResponse.

    Covers OpenRouter provider exceptions, the base ``openai.OpenAIError`` (e.g.
    missing credentials at client construction), and malformed/schema-violating
    responses (ProviderResponseError / ValidationError) so a bad reply becomes an
    actionable error, not a crash. Ordering is most-specific first because the openai
    hierarchy nests: APITimeoutError ⊂ APIConnectionError, and RateLimitError /
    AuthenticationError / PermissionDeniedError ⊂ APIStatusError.

    Retryability is honest: network / timeout / rate-limit / 5xx say "try again";
    a 4xx client error (bad request, not-found model, etc.) does NOT — it won't succeed
    on retry. An unclassified APIError falls through to the server message, not the
    network one. User-facing copy never embeds the raw exception, so the API key cannot
    leak (``to_error_response`` redaction is defense-in-depth). The exception *type* is
    logged (redacted — no message) for observability.
    """
    logger.warning("Extract provider failure: %s", type(exc).__name__)
    if isinstance(exc, openai.APITimeoutError):
        return to_error_response(ErrorStage.EXTRACT, _TIMEOUT_ERROR)
    if isinstance(exc, openai.APIConnectionError):
        return to_error_response(ErrorStage.EXTRACT, _NETWORK_ERROR)
    if isinstance(exc, openai.RateLimitError):
        return to_error_response(ErrorStage.EXTRACT, _RATE_LIMIT_ERROR)
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return to_error_response(ErrorStage.EXTRACT, _AUTH_ERROR)
    if isinstance(exc, openai.APIStatusError):
        if 400 <= exc.status_code < 500:
            return to_error_response(ErrorStage.EXTRACT, _CLIENT_ERROR)
        return to_error_response(ErrorStage.EXTRACT, _SERVER_ERROR)
    if isinstance(exc, (ProviderResponseError, ValidationError)):
        return to_error_response(ErrorStage.EXTRACT, _BAD_RESPONSE_ERROR)
    if isinstance(exc, openai.APIError):
        # An API-level error we didn't specifically classify (e.g.
        # APIResponseValidationError) — a service-side problem, not a local network one.
        return to_error_response(ErrorStage.EXTRACT, _SERVER_ERROR)
    if isinstance(exc, openai.OpenAIError):
        # Base SDK/config error not tied to an HTTP response — e.g. missing credentials
        # raised at client construction. Surface as a server-configuration failure.
        return to_error_response(ErrorStage.EXTRACT, _AUTH_ERROR)
    return to_error_response(ErrorStage.EXTRACT, _SERVER_ERROR)


async def _parse_cv_text(cv_bytes: bytes, filename: str) -> str | ErrorResponse:
    """Parse + char-clean the CV bytes to text, or return a parse-stage ErrorResponse.

    Bad-file paths halt at ``stage=parse`` before any provider call: an empty
    (zero-byte) upload, an oversized upload (> ``MAX_UPLOAD_MB``), a corrupt/unreadable
    file, and a file that parses but yields no text (image-only PDF). The emptiness and
    size guards run *before* ``extract_text`` so such files are never parsed; a corrupt
    accepted-format file surfaces as ``CorruptFileError`` from the parser. The no-text
    guard runs *after* input normalization so an invisibles-only document is treated as
    no-text, not empty facts. Returns the normalized CV text on success.

    Normalize with ``normalize_input`` (invisible-strip + targeted homoglyph mapping
    ONLY), NOT the full ``clean()`` pipeline, whose whitespace-collapse and AI-tell
    heuristics are OUTPUT-only and fuse real resumes into garbage.

    The empty/oversize guards stay synchronous (pure size checks), but ``extract_text``
    (PyMuPDF / python-docx) is CPU/IO-bound and blocking, so it runs via
    ``asyncio.to_thread`` to keep the event loop free. This is the ONLY threadpooled step —
    the downstream normalize/dedup/flag work is fast pure-CPU and runs inline.
    """
    if not cv_bytes:
        return to_error_response(ErrorStage.PARSE, _EMPTY_FILE_ERROR)
    if len(cv_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        return to_error_response(ErrorStage.PARSE, _OVERSIZED_FILE_ERROR)

    try:
        raw_text = await asyncio.to_thread(extract_text, cv_bytes, filename)
    except (UnsupportedFormatError, CorruptFileError) as exc:
        return to_error_response(ErrorStage.PARSE, str(exc))

    cv_text = TextPreprocessing.normalize_input(raw_text)
    if not cv_text.strip():
        return to_error_response(ErrorStage.PARSE, _NO_TEXT_ERROR)
    return cv_text


async def build_extract(
    cv_bytes: bytes,
    filename: str,
    job_description: str,
    llm: LLMModel | None = None,
) -> ExtractResult | ErrorResponse:
    """Build the single shared `ExtractResult` for one CV + one JD.

    Orchestrates: parse CV bytes → char-clean → extract facts + analyze JD →
    assemble one `ExtractResult`. The JD is analyzed exactly once and never re-parsed
    downstream. Bad-file uploads halt at ``stage=parse``; a non-fatal keyword
    gap is appended to ``flags``.

    Both the CV text and the JD are normalized with ``normalize_input`` before extraction
    so keyword matching runs on normalized text — ``match(normalize(text))``. The
    empty/corrupt parse guards via ``_parse_cv_text`` halt before the provider call
    (analyze_jd not reached). The single shared analysis is computed once and reused for
    both the gap-flag check and the result.

    The provider client is constructed AFTER the parse guards and inside the try, so a
    rejected upload never builds a client and a missing-credentials OpenAIError at
    construction becomes a clean stage=extract ErrorResponse, not a 500.

    Args:
        cv_bytes: The uploaded CV file bytes.
        filename: Original filename, used to route parsing on the extension.
        job_description: The pasted job-description text.
        llm: Injectable function-calling client; defaults to a real ``LLMModel``.

    Returns:
        The assembled ``ExtractResult`` on success, or an ``ErrorResponse`` when the
        upload is rejected at ``stage=parse`` (empty/corrupt/unsupported) or the
        provider call fails at ``stage=extract`` (network/timeout/auth/config).
    """
    cv_text = await _parse_cv_text(cv_bytes, filename)
    if isinstance(cv_text, ErrorResponse):
        return cv_text

    # Normalize the JD too so keyword analysis runs on normalized text.
    jd_text = TextPreprocessing.normalize_input(job_description)
    try:
        client = llm or LLMModel()
        # The CV-facts and JD-analysis calls are independent (neither feeds the other),
        # so run them concurrently. ``extract_cv_facts`` stays the FIRST gather arg, so it reaches
        # its await (and issues its provider call) first — identical to the sequential path — and
        # ``gather`` returns results in argument order, so ``facts``/``jd`` map back unchanged. Only
        # two calls (< MAX_CONCURRENT_LLM_CALLS), so no semaphore is needed here.
        facts, jd = await asyncio.gather(
            extract_cv_facts(cv_text, llm=client),
            analyze_jd(jd_text, llm=client),
        )
    except (openai.OpenAIError, ProviderResponseError, ValidationError) as exc:
        return _provider_error(exc)

    # Dedup + drop blanks before the gap check AND before to_job_target() scoring.
    jd.keywords = _dedup_keywords(jd.keywords)
    return ExtractResult(facts=facts, jd=jd, flags=_keyword_gap_flags(jd))
