"""Unit tests for the Streamlit client's four-state machine.

The UI logic is kept thin and pure: a dict standing in for
`st.session_state`, plus pure predicates for input/button enablement. No Streamlit runtime
is exercised here — rendering is covered in test_render.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app


def _new_session() -> dict:
    state: dict = {}
    app.init_session(state)
    return state


# --- empty -> loading on Generate click ----------------------


def test_empty_transitions_to_loading_on_generate() -> None:
    state = _new_session()
    assert state["ui_state"] == app.UiState.EMPTY
    # Both inputs present => the button is enabled, so a click is possible.
    assert (
        app.is_generate_disabled(has_file=True, jd_text="JD text", ui_state=state["ui_state"])
        is False
    )
    app.begin_loading(state)
    assert state["ui_state"] == app.UiState.LOADING


# --- loading -> success on 200 renders ResultArea ------------


def test_loading_transitions_to_success_on_200() -> None:
    state = {"ui_state": app.UiState.LOADING, "result": None, "error": None}
    body = {"cv": {}, "cover_letter": "Dear team", "ats_score": {}, "flags": []}
    app.on_success(state, body)
    assert state["ui_state"] == app.UiState.SUCCESS
    assert state["result"]["cover_letter"] == "Dear team"


# --- re-submission blocked while loading ---------------------


def test_button_and_inputs_disabled_while_loading() -> None:
    loading = app.UiState.LOADING
    assert app.is_generate_disabled(has_file=True, jd_text="JD", ui_state=loading) is True
    assert app.inputs_disabled(app.UiState.LOADING) is True


# --- success -> loading on re-run clears prior result --------


def test_rerun_from_success_clears_result_first() -> None:
    state = {"ui_state": app.UiState.SUCCESS, "result": {"cover_letter": "old"}, "error": None}
    app.begin_loading(state)
    assert state["ui_state"] == app.UiState.LOADING
    assert state["result"] is None


# --- GenerateButton disabled until file AND non-empty JD -----


def test_generate_disabled_until_file_and_jd_present() -> None:
    s = app.UiState.EMPTY
    assert app.is_generate_disabled(has_file=False, jd_text="JD", ui_state=s) is True
    assert app.is_generate_disabled(has_file=True, jd_text="", ui_state=s) is True
    assert app.is_generate_disabled(has_file=True, jd_text="   \n ", ui_state=s) is True
    assert app.is_generate_disabled(has_file=False, jd_text="", ui_state=s) is True
    assert app.is_generate_disabled(has_file=True, jd_text="JD", ui_state=s) is False


# --- loading -> error on non-2xx re-enables inputs -----------


def test_loading_transitions_to_error_and_reenables_inputs() -> None:
    state = {"ui_state": app.UiState.LOADING, "result": None, "error": None}
    app.on_error(state, {"error": "Unsupported file type.", "stage": "parse"})
    assert state["ui_state"] == app.UiState.ERROR
    assert state["error"]["stage"] == "parse"
    assert app.inputs_disabled(app.UiState.ERROR) is False
    err = app.UiState.ERROR
    assert app.is_generate_disabled(has_file=True, jd_text="JD", ui_state=err) is False


# --- a 200 must carry the full result envelope to count as success -----------------------


def test_is_success_response_requires_full_envelope() -> None:
    full = {"cv": {}, "ats_score": {}, "cover_letter": "x", "flags": []}
    assert app.is_success_response(200, full) is True
    # A 200 whose body is the transport error dict (e.g. non-JSON body) is NOT success.
    assert app.is_success_response(200, {"error": "Unexpected server response."}) is False
    # A 200 with a wrong-shape body is NOT success (would KeyError render_result otherwise).
    assert app.is_success_response(200, {}) is False
    assert app.is_success_response(502, full) is False


# --- a click defers the request to a `loading` rerun (no same-run call) ------


def test_click_defers_request_to_loading_rerun(monkeypatch) -> None:  # noqa: ANN001
    """A Generate click enters `loading` and reruns BEFORE calling the backend, so the request
    runs on a pass where the inputs/button are disabled — a same-run call would never render the
    loading lock, leaving a double-submit possible."""
    fake_st = MagicMock()
    fake_st.session_state = {
        "ui_state": app.UiState.EMPTY,
        "result": None,
        "error": None,
        "_cv_file": object(),
        "_jd_text": "JD text",
    }
    calls = {"generate": 0}

    def _spy_form(st_obj: object, *, inputs_off: bool, button_off: bool) -> tuple:
        return (object(), "JD text", True)  # a file + JD present, button clicked

    def _spy_generate(cv_file: object, jd: str) -> tuple:
        calls["generate"] += 1
        return (200, {})

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_input_form", _spy_form)
    monkeypatch.setattr(app, "call_generate", _spy_generate)
    app.main()
    assert calls["generate"] == 0  # deferred to the loading rerun, not called in the click run
    assert fake_st.session_state["ui_state"] == app.UiState.LOADING
    fake_st.rerun.assert_called_once()


def test_loading_pass_runs_request_with_inputs_disabled(monkeypatch) -> None:  # noqa: ANN001
    """On the `loading` rerun the form is rendered with inputs AND button disabled (the
    lock is reachable at render, not test-only) and the deferred request runs."""
    fake_st = MagicMock()
    fake_st.session_state = {
        "ui_state": app.UiState.LOADING,
        "result": None,
        "error": None,
        "_cv_file": object(),
        "_jd_text": "JD text",
    }
    captured: dict = {}

    def _spy_form(st_obj: object, *, inputs_off: bool, button_off: bool) -> tuple:
        captured["inputs_off"] = inputs_off
        captured["button_off"] = button_off
        return (None, "", False)

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_input_form", _spy_form)
    monkeypatch.setattr(app, "call_generate", lambda f, j: (200, {"error": "x"}))
    app.main()
    assert captured["inputs_off"] is True
    assert captured["button_off"] is True
