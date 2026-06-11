"""Contract tests for the `POST /generate` endpoint.

The endpoint orchestrates the existing pipeline (`build_extract` -> `generate_tailored_cv`
+ `generate_cover_letter`) into a single JSON envelope. These tests patch the three pipeline
entry points so no LLM is called and exercise the transport boundary only:
the 200 envelope shape, sanitized error mapping, flags-ride-200, and statelessness /
single-extract reuse.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main
from cv_generator import AtsScore, CoverLetterResult, FlagKind, SectionFlag, TailoredResult
from helprers.cv_template import (
    ActionVerb,
    BulletPoint,
    Category,
    CVTemplate,
    Education,
    Experience,
    Link,
    PersonalInfo,
    Skills,
    Summary,
)
from schemas import ErrorResponse, ErrorStage


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def _sample_cv() -> CVTemplate:
    return CVTemplate(
        personal_info=PersonalInfo(
            name="Jane Doe",
            location="Remote",
            email="jane@example.com",
            links=[Link(title="LinkedIn", url="https://linkedin.com/in/jane")],
        ),
        summary=Summary(text="Engineer with backend focus.", relevant_skills=["Python"]),
        skills=Skills(
            categories=[
                Category(category="Languages", keywords=["Python"]),
                Category(category="Tools", keywords=["FastAPI"]),
                Category(category="Soft skills", keywords=["Communication"]),
            ],
        ),
        experiences=[
            Experience(
                role="Backend Engineer",
                company="Acme",
                company_description="Leading logistics platform",
                start_date="2020-01",
                end_date="2022-01",
                location="Remote",
                bullets=[
                    BulletPoint(
                        action_verb=ActionVerb.DEVELOPED,
                        description="payment services",
                        skills=["Python"],
                        impact="+10% throughput",
                    )
                ],
            )
        ],
        education=[Education(institution="MIT", degree="BSc CS", end_year=2019)],
        section_order=["contact", "summary", "skills", "experience", "education"],
    )


def _sample_ats() -> AtsScore:
    return AtsScore(
        before_pct=40.0,
        after_pct=80.0,
        matched=["python", "fastapi"],
        missing=["kubernetes"],
    )


def _multipart(jd: str = "We need a Python backend engineer.") -> dict:
    return {
        "files": {"resume": ("cv.txt", b"Jane Doe resume text", "text/plain")},
        "data": {"job_description": jd},
    }


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extract: object | None = None,
    tailored: object | None = None,
    cover: object | None = None,
) -> dict[str, AsyncMock]:
    """Patch the three pipeline entry points referenced by main; return the mocks.

    The pipeline entry points are coroutines now, so the patches are ``AsyncMock`` — the async
    route ``await``s them and the same ``return_value`` / ``call_args`` / ``assert_called*``
    assertions hold.
    """
    extract = extract if extract is not None else object()
    tailored = (
        tailored
        if tailored is not None
        else TailoredResult(cv=_sample_cv(), ats_score=_sample_ats(), flags=[])
    )
    cover = cover if cover is not None else CoverLetterResult(cover_letter="Dear team,\n\n...")
    mocks = {
        "build_extract": AsyncMock(return_value=extract),
        "generate_tailored_cv": AsyncMock(return_value=tailored),
        "generate_cover_letter": AsyncMock(return_value=cover),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(main, name, mock)
    return mocks


def test_generate_accepts_multipart_and_returns_json(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /generate accepts multipart/form-data and returns application/json."""
    _patch_pipeline(monkeypatch)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


def test_generate_200_body_has_envelope_keys(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """200 body contains cv, cover_letter, ats_score, flags."""
    _patch_pipeline(monkeypatch)
    body = client.post("/generate", **_multipart()).json()
    assert set(["cv", "cover_letter", "ats_score", "flags"]).issubset(body.keys())
    assert body["cover_letter"].startswith("Dear team,")
    assert body["ats_score"]["after_pct"] == 80.0
    assert body["cv"]["section_order"] == [
        "contact",
        "summary",
        "skills",
        "experience",
        "education",
    ]


def test_unsupported_file_maps_to_parse_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bad/unsupported file -> non-2xx {error, stage: parse}."""
    err = ErrorResponse(
        error="Unsupported file type. Upload a PDF, DOCX, or TXT.", stage=ErrorStage.PARSE
    )
    _patch_pipeline(monkeypatch, extract=err)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 400
    body = resp.json()
    assert body["stage"] == "parse"
    assert body["error"]
    assert set(body.keys()) == {"error", "stage"}


def test_empty_job_description_maps_to_parse_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """empty/whitespace-only JD -> {stage: parse}, before any provider call."""
    mocks = _patch_pipeline(monkeypatch)
    resp = client.post("/generate", **_multipart(jd="   \n  "))
    assert resp.status_code == 400
    assert resp.json()["stage"] == "parse"
    mocks["build_extract"].assert_not_called()


def test_provider_failure_is_sanitized(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The redaction boundary replaces a leaked API key with ***.

    Feeds a message that actually CONTAINS the key (an extract-stage message passes through
    unchanged, so `_sanitize` is what must redact it) — the test fails if `_sanitize` no-ops.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETKEY-123")
    err = ErrorResponse(
        error="Provider rejected token sk-or-SECRETKEY-123 (HTTP 401).", stage=ErrorStage.EXTRACT
    )
    _patch_pipeline(monkeypatch, extract=err)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 502
    raw = resp.text
    assert "sk-or-SECRETKEY-123" not in raw
    assert "***" in resp.json()["error"]
    assert "Traceback" not in raw


def test_generator_error_uses_stable_message_not_raw_provider_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generate-stage failure ships a stable message, never the raw str(exc).

    The pipeline's generate envelope carries the raw provider error (only key-redacted); the
    endpoint must replace it with a stable, actionable message and never forward provider
    JSON / status fragments (or any residual key) to the client.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETKEY-123")
    raw = 'Error code: 429 - {"error":{"message":"rate limited","key":"sk-or-SECRETKEY-123"}}'
    _patch_pipeline(monkeypatch, tailored={"error": raw, "stage": "generate"})
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 502
    body = resp.json()
    assert set(body.keys()) == {"error", "stage"}
    assert body["stage"] == "generate"
    assert body["error"] == main._STABLE_MESSAGE_BY_STAGE["generate"]
    # Neither the raw provider fragment nor the key survives.
    assert "429" not in resp.text
    assert "sk-or-SECRETKEY-123" not in resp.text


def test_unknown_stage_falls_back_sanitized_not_raw_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An out-of-enum stage falls back to a sanitized envelope, not an unhandled 500.

    `_error` must not let `ErrorStage(stage)` raise inside the handler (which would bypass
    `_sanitize` and emit a raw 500 leaking the message/key).
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETKEY-123")
    raw = "boom from sk-or-SECRETKEY-123"
    _patch_pipeline(monkeypatch, tailored={"error": raw, "stage": "bananas"})
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 502  # fallback (extract), never an unhandled 500
    body = resp.json()
    assert set(body.keys()) == {"error", "stage"}
    assert body["stage"] in {e.value for e in ErrorStage}
    assert "sk-or-SECRETKEY-123" not in resp.text


def test_unexpected_exception_returns_sanitized_envelope_not_raw_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected exception (a bug, not a handled {error, stage}) must stay inside the
    contract: the app-level catch-all routes it through `_error`/`_sanitize` to a sanitized
    generic {error, stage} envelope — never a raw Starlette 500 that could leak the traceback/key.
    An unanticipated exception is attributed to the upstream `_FALLBACK_STAGE` (502-class), not the
    internal `assemble` stage, and carries the generic fallback message (never the raw `str(exc)`).
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETKEY-123")

    def _boom(extract: object) -> object:
        # An unhandled exception TYPE (not ProviderResponseError nor a dict envelope) on the
        # live path; the endpoint does not catch it, so only the app-level handler can contain it.
        raise RuntimeError("unexpected boom leaking sk-or-SECRETKEY-123")

    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(main, "generate_tailored_cv", _boom)
    # raise_server_exceptions=False so the registered Exception handler's RESPONSE is observed
    # (the default TestClient re-raises after the handler runs, masking the envelope).
    local_client = TestClient(main.app, raise_server_exceptions=False)
    resp = local_client.post("/generate", **_multipart())

    assert resp.status_code == 502  # upstream fallback, not the internal-failure 500
    body = resp.json()
    assert set(body.keys()) == {"error", "stage"}
    assert body["stage"] == main._FALLBACK_STAGE.value  # upstream fallback, not "assemble"
    assert "sk-or-SECRETKEY-123" not in resp.text  # key never leaks into the body
    assert "boom" not in resp.text  # raw exception text never forwarded


def test_unreadable_upload_maps_to_parse_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read failure stays inside the {error, stage:parse} envelope, not a 500."""
    _patch_pipeline(monkeypatch)

    def _boom(resume: object) -> bytes:
        raise OSError("spooled upload is gone")

    # The route reads the upload via the async `_read_upload` seam; a sync `_boom` raising OSError
    # surfaces at the call site (before the await), and the handler maps it to {error, stage:parse}.
    monkeypatch.setattr(main, "_read_upload", _boom)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 400
    body = resp.json()
    assert body["stage"] == "parse"
    assert set(body.keys()) == {"error", "stage"}


def test_flags_ride_200_and_stay_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 with flags stays success (not error); flags ride the body."""
    flag = SectionFlag(
        section="global", kind=FlagKind.UNMET_COVERAGE, message="Missing keyword: Kubernetes"
    )
    tailored = TailoredResult(cv=_sample_cv(), ats_score=_sample_ats(), flags=[flag])
    cover = CoverLetterResult(
        cover_letter="Dear team,\n\n...",
        flags=[
            SectionFlag(
                section="cover_letter",
                kind=FlagKind.COVER_LETTER_GAP,
                message="No must-haves",
            )
        ],
    )
    _patch_pipeline(monkeypatch, tailored=tailored, cover=cover)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 200
    body = resp.json()
    messages = [f["message"] for f in body["flags"]]
    assert "Missing keyword: Kubernetes" in messages
    assert "No must-haves" in messages


def test_request_persists_nothing_and_reuses_single_extract(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No upload written to disk; build_extract invoked once and reused."""
    sentinel_extract = object()
    mocks = _patch_pipeline(monkeypatch, extract=sentinel_extract)

    import builtins

    real_open = builtins.open
    write_opens: list[str] = []

    def _spy_open(file, mode="r", *args, **kwargs):  # noqa: ANN001
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            write_opens.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _spy_open)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 200

    mocks["build_extract"].assert_called_once()
    # Both generators receive the exact same ExtractResult instance (single shared extract).
    assert mocks["generate_tailored_cv"].call_args.args[0] is sentinel_extract
    assert mocks["generate_cover_letter"].call_args.args[0] is sentinel_extract
    assert write_opens == []


# Async migration: the handler is a coroutine so its I/O is non-blocking


def test_generate_handler_is_async_for_nonblocking_io() -> None:
    """The route is now a coroutine: its LLM calls are true ``await``s against ``AsyncOpenAI`` and
    the blocking CV parse runs via ``asyncio.to_thread``, so no blocking work sits on the event
    loop (GET /health stays responsive while a /generate is in flight). This replaces the former
    sync-``def`` + threadpool-offload design."""
    assert inspect.iscoroutinefunction(main.generate)


# A cover-letter failure degrades gracefully and keeps the finished CV


def test_cover_letter_failure_degrades_and_keeps_cv(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cover-letter generator failure returns 200 with the CV + ATS score + a
    cover_letter_gap flag (graceful degradation), NOT a 502 that discards the CV, and
    never the CV-generation failure message (the CV succeeded)."""
    _patch_pipeline(monkeypatch, cover={"error": "rate limited", "stage": "generate"})
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 200
    body = resp.json()
    assert body["cv"]["section_order"]  # the finished CV survived
    assert body["ats_score"]["after_pct"] == 80.0
    assert body["cover_letter"] == ""
    assert "cover_letter_gap" in [flag["kind"] for flag in body["flags"]]
    assert "problem generating your CV" not in resp.text


# A non-integer year maps to the {error, stage} envelope, not a raw 500


def test_non_integer_year_maps_to_envelope_not_raw_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model-emitted non-integer year raises a mapping ValidationError that stays inside
    the {error, stage} envelope instead of escaping as a raw 500."""
    cv = _sample_cv()
    cv.education = [Education(institution="MIT", degree="BSc CS", end_year="Present")]
    tailored = TailoredResult(cv=cv, ats_score=_sample_ats(), flags=[])
    _patch_pipeline(monkeypatch, tailored=tailored)
    resp = client.post("/generate", **_multipart())
    assert resp.status_code == 500
    body = resp.json()
    assert set(body.keys()) == {"error", "stage"}
    assert body["stage"] == "assemble"


# A missing multipart field maps to {error, stage:parse}, not FastAPI's 422 {detail}


def test_missing_field_maps_to_parse_envelope_not_422(client: TestClient) -> None:
    """A request missing a required field returns the documented {error, stage:parse}
    envelope, not FastAPI's default 422 {detail}."""
    resp = client.post("/generate", files={"resume": ("cv.txt", b"x", "text/plain")})
    assert resp.status_code == 400
    body = resp.json()
    assert set(body.keys()) == {"error", "stage"}
    assert body["stage"] == "parse"


# The api key is redacted in the LOG line, not only the HTTP body


def test_error_log_line_is_key_redacted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A key that slips into a raw pipeline message is redacted before it reaches the logs
    (api_key redacted in ALL log output), not only the response body."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETKEY-123")
    with caplog.at_level(logging.WARNING):
        main._error("generate", "boom token sk-or-SECRETKEY-123 leaked")
    assert "sk-or-SECRETKEY-123" not in caplog.text
    assert "***REDACTED***" in caplog.text


# The backend loads .env at import so OPENROUTER_API_KEY reaches the client


def test_backend_loads_dotenv_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the backend calls load_dotenv(), so a .env-configured OPENROUTER_API_KEY
    reaches the lazily-constructed OpenRouter client (else every /generate 502s)."""
    import dotenv

    calls: list[int] = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append(1))
    importlib.reload(main)
    try:
        assert calls, "main must call load_dotenv() at import"
    finally:
        importlib.reload(main)
