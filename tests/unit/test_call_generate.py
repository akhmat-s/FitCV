"""Unit tests for the Streamlit client's HTTP transport adapter.

`call_generate` is the UI's whole "never crashes when the backend is down" promise: it must
turn transport failures into an `error`-state payload (not an exception), distinguish a read
timeout from a connectivity failure (timeout vs network_failure), and survive a non-JSON
response. These run without a live server by monkeypatching `requests.post`.
"""

from __future__ import annotations

import importlib
import os

import requests

import app


class _FakeUpload:
    """Minimal stand-in for a Streamlit UploadedFile (only `.name` / `.getvalue()` are used)."""

    name = "cv.txt"

    def getvalue(self) -> bytes:
        return b"Jane Doe resume text"


def test_connectivity_failure_returns_unreachable_payload(monkeypatch) -> None:  # noqa: ANN001
    def _raise(*args: object, **kwargs: object) -> object:
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(app.requests, "post", _raise)
    status, body = app.call_generate(_FakeUpload(), "JD text")
    assert status == 0
    assert "reach the server" in body["error"]


def test_connect_timeout_returns_unreachable_not_timeout_payload(monkeypatch) -> None:  # noqa: ANN001
    # C9: ConnectTimeout subclasses BOTH ConnectionError and Timeout — an unreachable (black-holed)
    # host must surface the connectivity message, NOT the read-timeout "took too long" message.
    def _raise(*args: object, **kwargs: object) -> object:
        raise requests.exceptions.ConnectTimeout("connect timed out")

    monkeypatch.setattr(app.requests, "post", _raise)
    status, body = app.call_generate(_FakeUpload(), "JD text")
    assert status == 0
    assert "reach the server" in body["error"]
    assert "too long" not in body["error"]


def test_timeout_returns_distinct_timeout_payload(monkeypatch) -> None:  # noqa: ANN001
    def _raise(*args: object, **kwargs: object) -> object:
        # ReadTimeout ⊂ Timeout ⊂ RequestException — the read-timeout branch must win here.
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(app.requests, "post", _raise)
    status, body = app.call_generate(_FakeUpload(), "JD text")
    assert status == 0
    assert "too long" in body["error"]
    # Distinct from the connectivity message — not collapsed into "is the backend running?".
    assert "reach the server" not in body["error"]


def test_non_json_response_is_handled_without_crashing(monkeypatch) -> None:  # noqa: ANN001
    class _NonJsonResponse:
        status_code = 502

        def json(self) -> object:
            raise ValueError("no json")

    monkeypatch.setattr(app.requests, "post", lambda *a, **k: _NonJsonResponse())
    status, body = app.call_generate(_FakeUpload(), "JD text")
    assert status == 502
    assert body["error"] == "Unexpected server response."


def test_api_base_url_uses_dotenv_loaded_at_import(monkeypatch) -> None:  # noqa: ANN001
    # C2: load_dotenv() must run at import BEFORE the API_BASE_URL constant is read (a
    # load_dotenv() inside main() runs too late — the constant is already frozen). Spy on
    # dotenv.load_dotenv to set the var; if the call happens before the read, the constant
    # reflects it. If load_dotenv were still inside main(), the import-time read sees no var
    # and falls back to the localhost default — so this fails on the unfixed code.
    import dotenv

    monkeypatch.delenv("API_BASE_URL", raising=False)

    def _spy_load(*args: object, **kwargs: object) -> bool:
        os.environ["API_BASE_URL"] = "http://from-dotenv:9000"
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", _spy_load)
    importlib.reload(app)
    try:
        assert app.API_BASE_URL == "http://from-dotenv:9000"
    finally:
        monkeypatch.undo()
        os.environ.pop("API_BASE_URL", None)
        importlib.reload(app)


def test_success_returns_status_and_json_body(monkeypatch) -> None:  # noqa: ANN001
    class _OkResponse:
        status_code = 200

        def json(self) -> dict:
            return {"cv": {}, "cover_letter": "Dear team", "ats_score": {}, "flags": []}

    monkeypatch.setattr(app.requests, "post", lambda *a, **k: _OkResponse())
    status, body = app.call_generate(_FakeUpload(), "JD text")
    assert status == 200
    assert body["cover_letter"] == "Dear team"
