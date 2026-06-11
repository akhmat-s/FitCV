"""Stateless Streamlit client for the FitCV tailoring pipeline.

The UI talks to the FastAPI backend over the HTTP boundary: it posts the CV file +
JD as `multipart/form-data` to `POST /generate` and renders four states (`empty`, `loading`,
`success`, `error`) held in `st.session_state`. No persistence, no accounts, no disk writes —
the upload bytes are forwarded straight to the backend and never saved.

The module is split into pure logic (the state machine + enablement predicates) and render
helpers that take an injected `st`-like object, so both are unit-testable without a Streamlit
runtime (tests/unit/test_app_state.py, tests/unit/test_render.py).
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env at import, BEFORE API_BASE_URL is read below, so a deployment can point the UI
# at a non-local backend via .env. A load_dotenv() inside main() runs too late — the module
# constant is already frozen at import. No secrets here; the OpenRouter key stays server-side.
load_dotenv()

logger = logging.getLogger(__name__)

# --- UI constants -------------------------------------------------------------
# The Streamlit client talks to the FastAPI backend over the HTTP boundary;
# these constants configure that single call. API_BASE_URL is read from the env so a
# deployment can point the UI at a non-local backend; the OpenRouter key stays
# server-side (read by the FastAPI process) and is never referenced here.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Endpoint path the client POSTs the multipart request to.
GENERATE_PATH = "/generate"

# requests.post read timeout, in seconds. `/generate` is a multi-call pipeline
# (extract + per-section generation with capped regeneration + cover letter), so its wall
# time is a multiple of the per-call EXTRACT_TIMEOUT_S, not one call plus a buffer. This is a
# deliberately generous client ceiling chosen to exceed the expected pipeline runtime — not a
# guarantee the client outlasts the server. If the pipeline still overruns it, the timeout
# branch in `call_generate` surfaces the "took too long" message instead of the client hanging.
REQUEST_TIMEOUT_S = 300

# Upload formats the file uploader accepts (mirror of
# schemas.ACCEPTED_FORMATS). Size/format are re-guarded server-side (stage=parse).
ACCEPTED_FORMATS = ["pdf", "docx", "txt"]


# --- State machine ------------------------------------------------------------


class UiState(StrEnum):
    """The four UI states held in `st.session_state`."""

    EMPTY = "empty"
    LOADING = "loading"
    SUCCESS = "success"
    ERROR = "error"


# Human-readable section headings for the tailored CV.
_SECTION_TITLES: dict[str, str] = {
    "contact": "Contact",
    "summary": "Summary",
    "skills": "Skills",
    "experience": "Experience",
    "education": "Education",
    "projects": "Projects",
    "certificates": "Certificates",
}


def init_session(state: dict) -> None:
    """Seed the session with the initial `empty` state (idempotent across reruns)."""
    state.setdefault("ui_state", UiState.EMPTY)
    state.setdefault("result", None)
    state.setdefault("error", None)


def is_generate_disabled(*, has_file: bool, jd_text: str, ui_state: str) -> bool:
    """Generate is disabled until a file is present AND the JD is non-empty, and while loading.

    The primary action stays disabled until both inputs are provided, and
    re-submission is blocked while a request is in flight (`loading`).
    """
    if ui_state == UiState.LOADING:
        return True
    return not has_file or not jd_text.strip()


def inputs_disabled(ui_state: str) -> bool:
    """The CV uploader and JD textarea are locked only while a request is in flight."""
    return ui_state == UiState.LOADING


def begin_loading(state: dict) -> None:
    """Enter `loading`, clearing any prior result/error first (re-run exit path)."""
    state["result"] = None
    state["error"] = None
    state["ui_state"] = UiState.LOADING


def on_success(state: dict, response: dict) -> None:
    """A 200 envelope -> `success`; flags ride the body and never flip to `error`."""
    state["result"] = response
    state["error"] = None
    state["ui_state"] = UiState.SUCCESS


def on_error(state: dict, payload: dict) -> None:
    """A non-2xx / network failure -> `error`; inputs are re-enabled."""
    state["error"] = payload
    state["result"] = None
    state["ui_state"] = UiState.ERROR


def is_success_response(status: int, body: dict) -> bool:
    """A response is a renderable success only on HTTP 200 carrying the full result envelope.

    Guards `render_result` from a 200 whose body isn't a valid result — e.g. a proxy's HTML
    error page that parsed to the transport error dict, or a wrong-shape JSON — by routing those
    to the error state instead of KeyError-crashing the success render.
    """
    return (
        status == 200
        and isinstance(body, dict)
        and {"cv", "ats_score", "cover_letter"} <= body.keys()
    )


# FlagKind values (cv_generator.FlagKind) that are informational notes, not actionable defects —
# rendered with a note treatment in the Recommendations panel. Every other kind, and any flag
# without a `kind`, is treated as an actionable warning.
_NOTE_FLAG_KINDS = frozenset({"cover_letter_gap", "cover_letter_no_requirements"})


def should_show_flags(flags: list[dict]) -> bool:
    """FlagsPanel is visible when there are flags.

    Honestly-omitted keywords are carried as a dedicated backend flag (the single "Missing
    (no CV evidence …)" entry), so visibility gates on flags alone — the missing keyword count
    is always shown separately in the AtsScorePanel.
    """
    return bool(flags)


def signed_delta(after_pct: float, before_pct: float) -> str:
    """Signed coverage-lift string for `st.metric` — sign text always present, never clamped.

    A negative lift (`after_pct < before_pct`) renders as a signed negative, not 0.
    A near-zero negative rounds to -0.0, which formats as "-0.0%" and makes st.metric show
    a red down-arrow (a false regression); normalize it to 0.0 so a no-change lift reads as 0.
    """
    delta = round(after_pct - before_pct, 1)
    if delta == 0:
        delta = 0.0
    return f"{delta:+.1f}%"


# --- Section text builders (copy-friendly plain text per CV section) ----------


def _contact_text(cv: dict) -> str:
    info = cv["personal_info"]
    parts = [info["name"], info["location"], info["email"]]
    if info.get("phone"):
        parts.append(info["phone"])
    # A link's title is Optional (a bare URL has no anchor text) — render just the URL then,
    # never a literal "None:" prefix (the `if p` filter below can't catch a non-empty f-string).
    parts.extend(
        f"{link['title']}: {link['url']}" if link.get("title") else link["url"]
        for link in info.get("links", [])
    )
    return "\n".join(p for p in parts if p)


def _summary_text(cv: dict) -> str:
    # Render only the prose `text`. The ATS scorer counts `summary.text` (not
    # relevant_skills — cv_generator._section_text), so score basis == render basis. The 🟡 JD
    # keywords are woven into the prose itself; relevant_skills stays in the data model but is no
    # longer shown as a separate line (which duplicated Skills→Relevant and broke the methodology).
    return cv["summary"]["text"]


def _skills_text(cv: dict) -> str:
    # Plain text is the single source of truth (regression repair): `st.code` shows this string
    # verbatim AND is the copy payload, so it must carry ZERO markdown — the `**header**` leak
    # rendered the asterisks literally and copied them. Header emphasis is a visual-render
    # concern only; the copy/source string is a bare "Header: kw, kw" line.
    lines = []
    for category in cv["skills"].get("categories", []):
        keywords = category.get("keywords") or []
        if not keywords:
            continue
        # An emergent header may legitimately be blank (the domain-universal ungrouped fallback):
        # render those keywords as a bare uncategorized line, never an invented "Header:" prefix.
        label = (category.get("category") or "").strip()
        lines.append(f"{label}: {', '.join(keywords)}" if label else ", ".join(keywords))
    return "\n".join(lines)


def _experience_text(cv: dict) -> str:
    # Render == score: render company_description and each bullet's prose. The ATS
    # scorer (cv_generator._experience_text) counts role/company/company_description plus the bullet
    # prose (description/impact/benefit); the action_verb is rendered as the bullet's lead-in but is
    # intentionally NOT part of the coverage basis. bullet.skills are structured tags the scorer no
    # longer counts, so they are NOT rendered either (rendering them would overstate coverage).
    blocks = []
    for exp in cv.get("experiences", []):
        span = " – ".join(d for d in (exp.get("start_date"), exp.get("end_date")) if d)
        header = f"{exp['role']} — {exp['company']}" + (f" ({span})" if span else "")
        lines = [header]
        if exp.get("company_description"):
            lines.append(exp["company_description"])
        for bullet in exp.get("bullets", []):
            line = f"  • {bullet['action_verb']} {bullet['description']}"
            if bullet.get("impact"):
                line += f" ({bullet['impact']})"
            if bullet.get("benefit"):
                line += f" — {bullet['benefit']}"
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _education_text(cv: dict) -> str:
    lines = []
    for edu in cv.get("education", []):
        year = edu.get("end_year") or ""
        lines.append(f"{edu['degree']}, {edu['institution']} {year}".strip())
    return "\n".join(lines)


def _projects_text(cv: dict) -> str:
    lines = []
    for proj in cv.get("projects", []):
        # description is Optional — omit the ": <desc>" tail rather than render "None".
        line = f"{proj['name']}: {proj['description']}" if proj.get("description") else proj["name"]
        # project.skills are scored (cv_generator._cv_to_text) — render them too.
        if proj.get("skills"):
            line += f" [{', '.join(proj['skills'])}]"
        lines.append(line)
    return "\n".join(lines)


def _certificates_text(cv: dict) -> str:
    lines = []
    for cert in cv.get("certificates", []):
        # issuer/year are Optional — drop the absent ones instead of rendering "None"/empty noise.
        year = cert.get("year") or ""
        issuer = cert.get("issuer") or ""
        head = f"{cert['title']} — {issuer}" if issuer else cert["title"]
        lines.append(f"{head} {year}".strip())
    return "\n".join(lines)


# Every section the pipeline can append to `section_order` — including the string-keyed
# "certificates" (cv_generator._section_order) — needs a builder here, or render_tailored_cv
# would skip content the backend returned. A section whose builder yields no text is skipped
# entirely. Keep this registry in sync with _SECTION_TITLES.
#
# "languages" is intentionally absent: spoken languages render once, inside the Skills block
# (the backend's "Spoken Languages" skill category), so the standalone Languages section is not
# rendered even though the backend still lists "languages" in section_order. Dropping it from
# this registry makes render_tailored_cv skip it cleanly. (Backlog: stop the backend emitting
# the languages section / section_order entry — out of scope for this render-only change.)
_SECTION_TEXT = {
    "contact": _contact_text,
    "summary": _summary_text,
    "skills": _skills_text,
    "experience": _experience_text,
    "education": _education_text,
    "projects": _projects_text,
    "certificates": _certificates_text,
}


def _section_text(cv: dict, section: str) -> str:
    builder = _SECTION_TEXT.get(section)
    return builder(cv) if builder else ""


# --- Render helpers (take an injected `st`; unit-tested with a MagicMock) ------


def render_header(st_obj: object) -> None:
    """AppHeader: product title + one-line caption. Static across all states."""
    st_obj.title("FitCV — ATS Resume Tailor")
    st_obj.caption("Upload your CV and paste a job description to get an ATS-tailored CV.")


def render_input_form(st_obj: object, *, inputs_off: bool, button_off: bool) -> tuple:
    """InputForm: the two inputs + Generate button. Returns (uploaded_file, jd_text, clicked).

    Each input carries a visible label (accessibility — never placeholder-only).
    """
    uploaded = st_obj.file_uploader(
        "Upload your CV (PDF, DOCX, or TXT)",
        type=ACCEPTED_FORMATS,
        accept_multiple_files=False,
        disabled=inputs_off,
        key="_cv_file",
    )
    jd_text = st_obj.text_area(
        "Paste the job description",
        height=300,
        disabled=inputs_off,
        key="_jd_text",
    )
    clicked = st_obj.button("Generate", type="primary", disabled=button_off)
    return uploaded, jd_text, clicked


def render_ats_panel(st_obj: object, ats: dict) -> None:
    """AtsScorePanel: coverage metric (value=after_pct, signed delta) + matched/missing counts.

    Honest-metric guard: when there were zero keywords to score (matched AND missing both
    empty), the backend's coverage is a vacuous 100% (`|matched| / 0` → 100). Never render
    that as a score — show a plain-language line instead. The normal metric+delta+counts path
    runs whenever any keyword exists. Render-only: the response contract is unchanged.
    """
    # Read sub-fields defensively: a contract-skewed 200 (proxy / partial body) that passed
    # is_success_response must degrade gracefully here, never KeyError-crash the success render.
    matched = ats.get("matched") or []
    missing = ats.get("missing") or []
    if not matched and not missing:
        st_obj.markdown(
            "No keywords were detected in this job description, so an ATS match score "
            "isn't available."
        )
        return
    after_pct = ats.get("after_pct", 0.0)
    before_pct = ats.get("before_pct", 0.0)
    st_obj.metric(
        label="ATS coverage",
        value=f"{after_pct:.0f}%",
        delta=signed_delta(after_pct, before_pct),
    )
    cols = st_obj.columns(2)
    with cols[0]:
        st_obj.markdown(f"**Matched keywords:** {len(matched)}")
    with cols[1]:
        st_obj.markdown(f"**Missing keywords:** {len(missing)}")


def render_flags(st_obj: object, flags: list[dict]) -> None:
    """FlagsPanel: a scannable "Recommendations" panel, one item per backend flag.

    Gated on the visibility rule — hidden entirely when there are no flags. Each flag renders as
    its own visually-separated item (never one concatenated blob). When the flag carries a
    `kind` (the backend SectionFlag.kind survives into the response), informational kinds use a
    note treatment (`st.info`) and everything else — every actionable issue, plus any flag with
    no `kind` — uses a warning treatment (`st.warning`). Messages are rendered VERBATIM as the
    backend emits them; this panel never rewords, strips, or re-appends text.
    """
    if not should_show_flags(flags):
        return
    st_obj.subheader("Recommendations")
    for flag in flags:
        message = flag["message"]
        if flag.get("kind") in _NOTE_FLAG_KINDS:
            st_obj.info(message)
        else:
            st_obj.warning(message)


def render_tailored_cv(st_obj: object, cv: dict) -> None:
    """TailoredCvView: one copy-friendly CvSectionBlock per non-empty `cv.section_order` section.

    `_section_order` always lists projects/experience/education even when empty, so a section
    whose builder yields no text is skipped entirely — no dangling heading + "—" placeholder.
    """
    for section in cv["section_order"]:
        text = _section_text(cv, section)
        if not text:
            continue
        st_obj.subheader(_SECTION_TITLES.get(section, section.title()))
        st_obj.code(text, language=None)


def render_cover_letter(st_obj: object, cover_letter: str) -> None:
    """CoverLetterExpander: collapsed expander wrapping a copy-friendly plain-text block."""
    with st_obj.expander("Cover letter"):
        st_obj.code(cover_letter, language=None)


def render_error(st_obj: object, error: str) -> None:
    """ErrorBanner: assertive alert with an actionable, plain-language message (never the key).

    The internal pipeline `stage` is not shown — it's developer jargon, and for client-side
    transport failures there is no real pipeline stage to name.
    """
    st_obj.error(error)


def render_result(st_obj: object, result: dict) -> None:
    """ResultArea (success): score panel, optional flags, tailored CV, cover-letter expander.

    Renders show/copy affordances only — no download/export or in-UI edit widgets.
    """
    render_ats_panel(st_obj, result["ats_score"])
    render_flags(st_obj, result.get("flags", []))  # flags is not required by is_success_response
    render_tailored_cv(st_obj, result["cv"])
    render_cover_letter(st_obj, result["cover_letter"])


# --- HTTP transport -----------------------------------------------------------


def call_generate(uploaded_file: object, jd_text: str) -> tuple[int, dict]:
    """POST the multipart request to the backend; return (status_code, json_body).

    Transport failures map to an `error`-state payload so the UI never crashes, and
    the caught exception is logged so the cause is never silently discarded. A read timeout
    (the server is up but the pipeline overran REQUEST_TIMEOUT_S) and a connectivity failure
    (the server is unreachable) are surfaced as distinct messages (timeout vs
    network_failure). The payload carries no fabricated pipeline `stage` — `stage` is a
    backend concept and no server stage was reached on a client-side transport failure. The
    OpenRouter key is never referenced here (server-side).
    """
    files = {"resume": (uploaded_file.name, uploaded_file.getvalue())}
    data = {"job_description": jd_text}
    try:
        resp = requests.post(
            f"{API_BASE_URL}{GENERATE_PATH}", files=files, data=data, timeout=REQUEST_TIMEOUT_S
        )
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as exc:
        # ConnectTimeout subclasses BOTH ConnectionError and Timeout, so it must be caught
        # BEFORE the read-Timeout branch — an unreachable backend is a connectivity failure,
        # not a pipeline overrun.
        logger.warning("call_generate could not reach the backend: %s", exc)
        return 0, {"error": "Could not reach the server. Is the backend running?"}
    except requests.exceptions.Timeout as exc:
        logger.warning("call_generate timed out after %ss: %s", REQUEST_TIMEOUT_S, exc)
        return 0, {"error": "The server took too long to respond. Try again."}
    except requests.exceptions.RequestException as exc:
        logger.warning("call_generate request failed: %s", exc)
        return 0, {"error": "Could not reach the server. Is the backend running?"}
    try:
        return resp.status_code, resp.json()
    except ValueError as exc:
        logger.warning("call_generate got a non-JSON response (HTTP %s): %s", resp.status_code, exc)
        return resp.status_code, {"error": "Unexpected server response."}


# --- Streamlit entrypoint -----------------------------------------------------


def _resolve_response(state: dict, status: int, body: dict) -> None:
    """Route a backend response into the success/error state.

    A 200 with the full result envelope is success; anything else (non-2xx, or a 200 whose body
    isn't a valid result) becomes an error — never a KeyError crash in the success render.
    """
    if is_success_response(status, body):
        on_success(state, body)
    else:
        payload = body if isinstance(body, dict) and "error" in body else {
            "error": "Unexpected server response."
        }
        on_error(state, payload)


def main() -> None:
    """Render the single-screen app and drive the four-state machine."""
    st.set_page_config(page_title="FitCV", layout="centered")
    init_session(st.session_state)
    ui_state = st.session_state["ui_state"]

    render_header(st)
    button_off = is_generate_disabled(
        has_file=uploaded_present(st), jd_text=current_jd(st), ui_state=ui_state
    )
    uploaded, jd_text, clicked = render_input_form(
        st, inputs_off=inputs_disabled(ui_state), button_off=button_off
    )

    if ui_state == UiState.LOADING:
        # The request runs on the dedicated `loading` rerun — the form above is already
        # rendered with inputs/button disabled, so a second submit can't re-fire /generate.
        # Inputs are read from session_state since the widgets are disabled this pass.
        with st.spinner("Tailoring your CV…"):
            status, body = call_generate(st.session_state.get("_cv_file"), current_jd(st))
        _resolve_response(st.session_state, status, body)
        st.rerun()
    elif clicked and uploaded is not None and jd_text.strip():
        # Enter `loading` and rerun so the next pass paints the disabled/spinner state before
        # the blocking call (the call itself happens on that rerun, above).
        begin_loading(st.session_state)
        st.rerun()

    ui_state = st.session_state["ui_state"]
    if ui_state == UiState.SUCCESS and st.session_state["result"] is not None:
        render_result(st, st.session_state["result"])
    elif ui_state == UiState.ERROR and st.session_state["error"] is not None:
        render_error(st, st.session_state["error"].get("error", "Something went wrong."))


def uploaded_present(st_obj: object) -> bool:
    """Whether the uploader currently holds a file (widget value via session_state key)."""
    return st_obj.session_state.get("_cv_file") is not None


def current_jd(st_obj: object) -> str:
    """The current JD textarea value (via session_state key), defaulting to empty."""
    return st_obj.session_state.get("_jd_text") or ""


if __name__ == "__main__":
    main()
