"""FastAPI delivery surface for the tailoring pipeline.

Exposes the stateless `POST /generate` endpoint plus a `GET /health` liveness probe. The
backend sits between the Streamlit UI and the model so the OpenRouter key stays server-side
only and is never returned.

This module also owns the transport-boundary response models: Pydantic mirrors of the
pipeline's `CVTemplate` / `AtsScore` / `SectionFlag` dataclasses (helprers/cv_template.py,
cv_generator.py). Field names are mirrored exactly to prevent drift.
"""

from __future__ import annotations

import asyncio
import logging
import os
from enum import StrEnum

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, computed_field

from cover_letter import generate_cover_letter
from cv_generator import (
    COVER_LETTER_SECTION,
    FlagKind,
    SectionFlag,
    TailoredResult,
    generate_tailored_cv,
)
from cv_generator import AtsScore as PipelineAtsScore
from extract import build_extract
from helprers.cv_template import ActionVerb, CVTemplate
from schemas import ErrorResponse as ExtractErrorResponse

# Load .env at import so OPENROUTER_API_KEY / MODEL_NAME reach the OpenRouter client the
# pipeline constructs lazily at call time. The backend runs as its own process (`uvicorn
# main:app`); without this the documented .env workflow leaves the key unset and every
# /generate fails at the extract stage. The key stays server-side only.
load_dotenv()

logger = logging.getLogger(__name__)

# --- Enums ---------------------------------------------------------------------


class ErrorStage(StrEnum):
    """Full pipeline error-stage set surfaced in the `{error, stage}` envelope.

    Union of the parsing subset (`parse`, `extract`) and the generation pipeline
    stages (`generate`, `validate`, `assemble`).
    """

    PARSE = "parse"
    EXTRACT = "extract"
    GENERATE = "generate"
    VALIDATE = "validate"
    ASSEMBLE = "assemble"


# --- CV mirror sub-models (mirror of helprers/cv_template.py dataclasses) ------
# These Pydantic models mirror the cv_template dataclasses field-by-field (drift guard). Do not
# rename fields without updating the corresponding dataclass — the response mappers copy by name.


class CVLink(BaseModel):
    """Mirror of `cv_template.Link`."""

    title: str | None = None
    url: str


class CVPersonalInfo(BaseModel):
    """Mirror of `cv_template.PersonalInfo`."""

    name: str
    location: str | None = None
    email: str
    phone: str | None = None
    links: list[CVLink] = Field(default_factory=list)


class CVSummary(BaseModel):
    """Mirror of `cv_template.Summary`."""

    text: str
    relevant_skills: list[str] = Field(default_factory=list)


class CVCategory(BaseModel):
    """Mirror of `cv_template.Category` — one JD-derived skills group."""

    category: str
    keywords: list[str] = Field(default_factory=list)


class CVSkills(BaseModel):
    """Mirror of `cv_template.Skills` — JD-derived domain categories."""

    categories: list[CVCategory] = Field(default_factory=list)


class CVBullet(BaseModel):
    """Mirror of `cv_template.BulletPoint` (action_verb serialized as its string value)."""

    action_verb: str
    description: str
    skills: list[str] = Field(default_factory=list)
    impact: str | None = None
    benefit: str | None = None


class CVExperience(BaseModel):
    """Mirror of `cv_template.Experience`."""

    role: str
    company: str
    company_description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    bullets: list[CVBullet] = Field(default_factory=list)


class CVEducation(BaseModel):
    """Mirror of `cv_template.Education`."""

    institution: str
    degree: str
    start_year: int | None = None
    end_year: int | None = None
    gpa: str | None = None


class CVProject(BaseModel):
    """Mirror of `cv_template.Project`."""

    name: str
    description: str | None = None
    skills: list[str] = Field(default_factory=list)
    link: CVLink | None = None


class CVCertificate(BaseModel):
    """Mirror of `cv_template.Certificate`."""

    title: str
    issuer: str | None = None
    year: int | None = None
    link: CVLink | None = None


class CVLanguage(BaseModel):
    """Mirror of `cv_template.Language`."""

    language: str
    level: str | None = None


class CVSchema(BaseModel):
    """Mirror of `cv_template.CVTemplate`.

    The top-level mirrors every CVTemplate field, including the render-config fields
    (`max_pages`..`background_color`) so the drift guard "preserves every field" holds in
    either direction. The UI only reads `section_order` + the content sections.
    """

    personal_info: CVPersonalInfo
    summary: CVSummary
    skills: CVSkills
    experiences: list[CVExperience] = Field(default_factory=list)
    education: list[CVEducation] = Field(default_factory=list)
    projects: list[CVProject] = Field(default_factory=list)
    certificates: list[CVCertificate] = Field(default_factory=list)
    languages: list[CVLanguage] = Field(default_factory=list)
    section_order: list[str] = Field(default_factory=list)
    max_pages: int = 1
    file_format: str = "PDF"
    font_family: list[str] = Field(default_factory=lambda: ["Helvetica", "Arial"])
    text_color: str = "black"
    background_color: str = "white"


# --- ATS / flags / response envelopes -----------------------------------------


class AtsScore(BaseModel):
    """Mirror of `cv_generator.AtsScore` — before→after keyword coverage."""

    before_pct: float
    after_pct: float
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coverage_pct(self) -> float:
        """Headline coverage == ``after_pct`` — computed like the dataclass, so they can't drift."""
        return self.after_pct


class Flag(BaseModel):
    """Mirror of `cv_generator.SectionFlag` — a non-fatal honest-gap marker.

    `kind` keeps the canonical `FlagKind` closed set so it survives into the
    OpenAPI schema; `section` stays an open string (`TargetSection | str` upstream — the
    "certificates"/"languages" sections are not TargetSection members).
    """

    section: str
    kind: FlagKind
    message: str


class GenerateResponse(BaseModel):
    """The 200 body for `POST /generate`."""

    cv: CVSchema
    cover_letter: str
    ats_score: AtsScore
    flags: list[Flag] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """The non-2xx body.

    `error` is actionable and never leaks the API key or a raw provider error.
    """

    error: str
    stage: ErrorStage


# --- Response mappers: dataclass -> Pydantic mirror ---------------------------
# Mappers copy by field name (drift guard). The only value transform is `action_verb`:
# upstream it is an `ActionVerb` enum (or a free string when the model returned an
# out-of-enum verb) — the mirror always carries its string.


def _map_link(link: object | None) -> CVLink | None:
    if link is None:
        return None
    return CVLink(title=link.title, url=link.url)


def _map_action_verb(value: object) -> str:
    return value.value if isinstance(value, ActionVerb) else str(value)


def _map_cv(cv: CVTemplate) -> CVSchema:
    """Map the pipeline `CVTemplate` dataclass tree to the `CVSchema` Pydantic mirror."""
    return CVSchema(
        personal_info=CVPersonalInfo(
            name=cv.personal_info.name,
            location=cv.personal_info.location,
            email=cv.personal_info.email,
            phone=cv.personal_info.phone,
            links=[CVLink(title=link.title, url=link.url) for link in cv.personal_info.links],
        ),
        summary=CVSummary(
            text=cv.summary.text,
            relevant_skills=list(cv.summary.relevant_skills),
        ),
        skills=CVSkills(
            categories=[
                CVCategory(category=category.category, keywords=list(category.keywords))
                for category in cv.skills.categories
            ],
        ),
        experiences=[
            CVExperience(
                role=exp.role,
                company=exp.company,
                company_description=exp.company_description,
                start_date=exp.start_date,
                end_date=exp.end_date,
                location=exp.location,
                bullets=[
                    CVBullet(
                        action_verb=_map_action_verb(bullet.action_verb),
                        description=bullet.description,
                        skills=list(bullet.skills),
                        impact=bullet.impact,
                        benefit=bullet.benefit,
                    )
                    for bullet in exp.bullets
                ],
            )
            for exp in cv.experiences
        ],
        education=[
            CVEducation(
                institution=edu.institution,
                degree=edu.degree,
                start_year=edu.start_year,
                end_year=edu.end_year,
                gpa=edu.gpa,
            )
            for edu in cv.education
        ],
        projects=[
            CVProject(
                name=proj.name,
                description=proj.description,
                skills=list(proj.skills),
                link=_map_link(proj.link),
            )
            for proj in cv.projects
        ],
        certificates=[
            CVCertificate(
                title=cert.title,
                issuer=cert.issuer,
                year=cert.year,
                link=_map_link(cert.link),
            )
            for cert in cv.certificates
        ],
        languages=[CVLanguage(language=lang.language, level=lang.level) for lang in cv.languages],
        section_order=list(cv.section_order),
        max_pages=cv.max_pages,
        file_format=cv.file_format,
        font_family=list(cv.font_family),
        text_color=cv.text_color,
        background_color=cv.background_color,
    )


def _map_ats(ats: PipelineAtsScore) -> AtsScore:
    """Map the pipeline `AtsScore` dataclass to its mirror (coverage_pct recomputed)."""
    return AtsScore(
        before_pct=ats.before_pct,
        after_pct=ats.after_pct,
        matched=list(ats.matched),
        missing=list(ats.missing),
    )


def _map_flag(flag: SectionFlag) -> Flag:
    """Map a pipeline `SectionFlag` to its mirror (`section` coerced to str; `kind` kept typed)."""
    return Flag(section=str(flag.section), kind=flag.kind, message=flag.message)


# --- Error envelope mapping ----------------------------------------------------
# stage -> HTTP status: bad input halts at 400; upstream model/transport at 502;
# internal pipeline failures (validate/assemble) at 500.

_STATUS_BY_STAGE: dict[str, int] = {
    ErrorStage.PARSE.value: 400,
    ErrorStage.EXTRACT.value: 502,
    ErrorStage.GENERATE.value: 502,
    ErrorStage.VALIDATE.value: 500,
    ErrorStage.ASSEMBLE.value: 500,
}

# Stable, job-seeker-actionable user message per generation stage. The generate /
# validate / assemble envelopes carry the raw `str(exc)` from the pipeline (only key-redacted),
# so provider JSON / status fragments would otherwise ship verbatim. We never forward that raw
# text — the user sees a stable message and the raw detail is logged server-side. The
# parse / extract stages already arrive as curated copy (this module's input guards and
# extract.py's `_provider_error`), so they pass through unchanged.
_STABLE_MESSAGE_BY_STAGE: dict[str, str] = {
    ErrorStage.GENERATE.value: "The tailoring service had a problem generating your CV. Try again.",
    ErrorStage.VALIDATE.value: "The tailored CV failed an internal check. Please try again.",
    ErrorStage.ASSEMBLE.value: "The tailored CV could not be assembled. Please try again.",
}

# Fallback for a stage the pipeline reports outside `ErrorStage`: an upstream failure
# (502) with a generic message, rather than letting `ErrorStage(stage)` raise and turn a
# handled error into an unhandled 500 that bypasses `_sanitize`.
_FALLBACK_STAGE = ErrorStage.EXTRACT
_FALLBACK_MESSAGE = "The request could not be completed. Please try again."


def _sanitize(message: str) -> str:
    """Belt-and-suspenders redaction: never echo the server-side OpenRouter key.

    Upstream helpers already redact provider errors; this is a second boundary so a key
    can never leak even if a future caller forwards a raw message.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key in message:
        # Same placeholder as extract._redact_api_key so redacted text is consistent.
        return message.replace(key, "***REDACTED***")
    return message


def _error(stage: str, message: str) -> JSONResponse:
    """Build the sanitized, stable non-2xx `{error, stage}` envelope for a pipeline stage.

    The message + `stage` are logged server-side first for debug detail — sanitized so
    the key never reaches the logs either, not only the HTTP body. An unknown
    `stage` falls back to a 502 extract envelope instead of raising inside the handler; a
    generate/validate/assemble stage is replaced with a stable message instead of shipping the
    raw provider text. `_sanitize` runs on whatever is emitted as a final key-leak guard.
    """
    logger.warning("generate pipeline error [stage=%s]: %s", stage, _sanitize(message))
    if stage in _STATUS_BY_STAGE:
        error_stage = ErrorStage(stage)
        user_message = _STABLE_MESSAGE_BY_STAGE.get(error_stage.value, message)
    else:
        error_stage = _FALLBACK_STAGE
        user_message = _FALLBACK_MESSAGE
    status = _STATUS_BY_STAGE[error_stage.value]
    body = ErrorResponse(error=_sanitize(user_message), stage=error_stage)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


# --- App ----------------------------------------------------------------------

app = FastAPI(
    title="FitCV Generate API",
    version="1.0.0",
    description="Stateless tailoring endpoint orchestrating extract -> tailor + cover letter.",
)


@app.exception_handler(RequestValidationError)
async def _on_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Map a missing/malformed multipart request to the {error, stage:parse} envelope.

    FastAPI's default handler returns a 422 {detail:[...]} body that doesn't match the locked
    {error, stage} contract; a parse-stage 400 keeps the documented shape for every
    client (including direct API callers), not just the bundled UI.
    """
    return _error(ErrorStage.PARSE.value, "Upload a CV file and paste a job description.")


@app.exception_handler(Exception)
async def _on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort boundary: an unexpected exception must NOT escape as a raw Starlette 500.

    The pipeline surfaces handled failures as typed `{error, stage}` values that `/generate`
    returns directly — those never reach here. This handler exists only for an UNEXPECTED
    exception (a bug: a `TypeError`/`KeyError`/`AttributeError` the endpoint did not anticipate).
    Without it such an exception ships a raw 500 that can leak the traceback (and any API key in
    it), bypassing the locked contract. We log the FULL exception server-side (traceback via
    `exc_info`, key-redacted) and return the sanitized generic `{error, stage}` envelope via
    `_error`. An unanticipated exception is NOT an internal CV-assembly failure, so it is
    attributed to the upstream `_FALLBACK_STAGE` (502-class) with the generic fallback message —
    never the raw `str(exc)` (which `_error` would forward verbatim for that stage and could leak
    internal detail). The full exception is still logged below. This LOGS, it never swallows — and
    it does not mask the typed envelopes, which take their existing return paths.
    """
    logger.exception("unhandled exception [path=%s]: %s", request.url.path, _sanitize(str(exc)))
    return _error(_FALLBACK_STAGE.value, _FALLBACK_MESSAGE)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for the UI base-URL check."""
    return {"status": "ok"}


async def _read_upload(resume: UploadFile) -> bytes:
    """Read the uploaded bytes via Starlette's async ``UploadFile.read`` (threadpool-offloaded).

    The route is ``async``, so the upload spool is read without blocking the event loop; kept as
    a one-line seam so a read failure (``OSError``) is injectable in tests and mapped to a
    ``stage=parse`` envelope by the caller.
    """
    return await resume.read()


@app.post("/generate", response_model=None)
async def generate(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
) -> GenerateResponse | JSONResponse:
    """Orchestrate the stateless tailoring pipeline into one JSON envelope.

    The route is ``async`` end-to-end: the LLM pipeline calls are true ``await``s against the
    ``AsyncOpenAI`` client and the blocking CV parse runs in a threadpool (``asyncio.to_thread``),
    so the event loop is never blocked while a request is in flight (GET /health stays responsive).
    `build_extract` runs once; the single shared `ExtractResult` is passed to both
    `generate_tailored_cv` and `generate_cover_letter` (no JD re-parse). Pipeline
    `{error, stage}` envelopes map to a sanitized non-2xx body; flags ride a 200.
    Nothing is persisted — the upload bytes live only in memory for the request.
    """
    # An empty/whitespace-only JD halts at stage=parse before any provider call.
    if not job_description or not job_description.strip():
        return _error(ErrorStage.PARSE.value, "Paste a job description to tailor against.")

    try:
        cv_bytes = await _read_upload(resume)
    except OSError:
        # An unreadable spool stays inside the typed envelope (stage=parse, the same input-stage
        # class as the empty-JD guard), not a bare 500.
        return _error(
            ErrorStage.PARSE.value, "Could not read the uploaded file. Re-upload and try again."
        )
    extract = await build_extract(cv_bytes, resume.filename or "", job_description)
    if isinstance(extract, ExtractErrorResponse):
        return _error(extract.stage.value, extract.error)

    # The tailored CV and the cover letter both consume the shared ExtractResult and
    # neither mutates it, so generate them concurrently. asyncio.gather returns results in argument
    # order (tailored, then cover), so the response is identical to the sequential path;
    # generate_tailored_cv is the first arg, so a synchronous failure there propagates before the
    # cover coroutine is even created.
    tailored, cover = await asyncio.gather(
        generate_tailored_cv(extract),
        generate_cover_letter(extract),
    )
    if isinstance(tailored, dict):
        return _error(tailored["stage"], tailored["error"])
    assert isinstance(tailored, TailoredResult)

    # The cover letter is a SUPPORTING feature — a generator failure degrades
    # gracefully (empty letter + a cover_letter_gap flag) instead of discarding the finished CV
    # and mislabeling it as a CV-generation failure. Only a core CV-pipeline failure 5xxs.
    if isinstance(cover, dict):
        logger.warning("cover-letter generation failed [stage=%s]; degrading", cover.get("stage"))
        cover_letter_text = ""
        cover_flags = [
            Flag(
                section=COVER_LETTER_SECTION,
                kind=FlagKind.COVER_LETTER_GAP,
                message="The cover letter could not be generated; your tailored CV is ready.",
            )
        ]
    else:
        cover_letter_text = cover.cover_letter
        cover_flags = [_map_flag(flag) for flag in cover.flags]

    # A malformed pipeline value (e.g. a non-integer year the model emitted) raises a
    # Pydantic ValidationError while mapping — keep it inside the {error, stage} envelope
    # rather than letting it escape as a raw 500 that breaks the locked contract.
    try:
        return GenerateResponse(
            cv=_map_cv(tailored.cv),
            cover_letter=cover_letter_text,
            ats_score=_map_ats(tailored.ats_score),
            flags=[_map_flag(flag) for flag in tailored.flags] + cover_flags,
        )
    except ValidationError as exc:
        return _error(ErrorStage.ASSEMBLE.value, f"Response validation failed: {exc}")
