"""Pydantic schemas + enums + constants for the shared extract pass.

Owns the internal `ExtractResult{facts, jd, flags}` envelope produced once per request
by `extract.build_extract` and reused by all downstream generation.
The mature `helprers/cv_template.py` dataclasses are unchanged and own the generated-CV
output; these schemas own the internal extract pass and are populated via
`model_validate()` from the hand-built tool-call JSON (see `tool_schemas.py`).

Resilience: real CVs vary — a header may omit location/phone, an entry may
omit dates/years/links, years may arrive as strings ("2023", "Expected 2026", "Present"),
and optional sub-objects may arrive null or malformed. The extract schema tolerates every
realistic OMISSION and coerces realistic TYPE variance (via mode="before" validators,
mirroring the existing JdAnalysis pattern) so a truthful-but-incomplete CV never 502s.
It still REJECTS structurally-broken core data: name/email and an entry's identifying
fields (role+company, institution+degree, link.url) remain required — an item missing its
core identity is dropped from the list rather than admitted as a hollow fact (truth-
preserving: better fewer real entries than a fabricated/empty one downstream).
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from helprers.cv_template import JobTarget

# --- Constants ----------------------------------------------------------------

#: Emit a non-fatal keyword-gap flag when keyword_count < MIN_KEYWORDS.
MIN_KEYWORDS: int = 5

#: Allowed upload formats; reject when format not in this set.
ACCEPTED_FORMATS: list[str] = ["pdf", "docx", "txt"]

#: Per-call OpenRouter provider ceiling, in seconds.
EXTRACT_TIMEOUT_S: int = 60

#: Reject uploads larger than this many megabytes before parsing.
MAX_UPLOAD_MB: int = 10

#: Cap on concurrent in-flight LLM calls per request (bounded fan-out; rate-limit safety).
#: The section generators fan out under an ``asyncio.Semaphore`` of this size, so a single
#: request never issues an unbounded burst against OpenRouter (``_RATE_LIMIT_ERROR`` must not be
#: provoked). Consumed by ``helprers.llm_model.bounded_gather`` / ``cv_generator``.
MAX_CONCURRENT_LLM_CALLS: int = 5

# --- Default model/endpoint config --------------------------------------------
# Plain default literals, NOT env reads. MODEL_NAME / OPENROUTER_BASE_URL (and the API
# key) are read at client-construction time in helprers/llm_model.py.

#: Default OpenRouter model id; overridable via the MODEL_NAME env var at construction.
DEFAULT_MODEL_NAME: str = "google/gemini-3.5-flash"

#: Default OpenAI-compatible base URL; overridable via the OPENROUTER_BASE_URL env var.
DEFAULT_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"


# --- Coercion helpers (resilience) --------------------------------------------
# Shared mode="before" coercers. They normalize realistic model output variance WITHOUT
# weakening truth: they convert types and drop malformed list items, they never invent
# values.


def _coerce_optional_str(value: Any) -> Any:
    """Normalize a scalar to a clean str or None.

    Empty / whitespace / placeholder strings ("not provided", "n/a", "none", "-",
    "unknown") collapse to None so a model-emitted placeholder never reaches a field
    as if it were real data. Non-empty strings are stripped. Non-strings (numbers) are
    stringified. Lists/dicts are rejected as None (a scalar field can't hold them).
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.lower() in {"not provided", "n/a", "na", "none", "null", "-", "unknown"}:
            return None
        return cleaned
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _coerce_optional_year(value: Any) -> Any:
    """Normalize a year to an int or None.

    Accepts an int, or a string that is a CLEAN single year — year-only ("2023") or a short
    word-qualified single year ("Expected 2026"). A string carrying a RANGE or extra alphanumerics
    around the year ("Cohort 2024-2025", "Batch 2019-A", "2024-present") is ambiguous, so it
    returns None rather than GUESSING the first 4-digit run (which would plant a wrong/fabricated-
    looking year). This is a narrow normalizer, not a date parser ("Present", "" → None).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # json.loads parses 1e400 → inf and the NaN token → nan; int(inf)/int(nan) RAISE (an
        # OverflowError/ValueError that is NOT a pydantic error and escapes model_validate). A
        # non-finite year is unparseable → None, preserving the never-502 resilience contract.
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        import re

        # Optional letter/space/dot qualifier, then exactly one 19xx/20xx year, optional trailing
        # dot, and NOTHING else — a range or an id around the year fails the full match → None.
        match = re.fullmatch(r"[A-Za-z. ]*((?:19|20)\d{2})\.?", value.strip())
        return int(match.group(1)) if match else None
    return None


def _coerce_str_list(value: Any) -> list[str]:
    """Normalize a value to a list of non-empty strings.

    Accepts a list (keeps only non-blank string items, stripped), a single string
    (wrapped), or anything else (→ []). Drops null/non-string list items rather than
    failing — a malformed bullet/skill never crashes the whole extract.
    """
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _drop_empty_skill_groups(value: Any) -> Any:
    """Drop skill groups that carry no non-blank item (identity = at least one item).

    A skill group's category label is OPTIONAL (a flat skills list has no header), so identity
    is its items, not its label. A group with an empty/all-blank ``items`` is hollow — dropped
    here rather than admitted, mirroring ``_drop_malformed_items`` for the keyed sub-models.
    Accepts raw dicts (the tool-call path via ``model_validate``) and constructed instances.
    """
    if not isinstance(value, list):
        return value
    kept = []
    for item in value:
        if isinstance(item, dict):
            items = _coerce_str_list(item.get("items"))
        elif isinstance(item, BaseModel):
            items = _coerce_str_list(getattr(item, "items", None))
        else:
            continue
        if items:
            kept.append(item)
    return kept


def _drop_malformed_items(value: Any, required_keys: tuple[str, ...]) -> Any:
    """For a list-of-objects field, drop items that lack their required identity keys.

    Keeps the extract resilient: one malformed experience/education/link entry is
    dropped (with its core identity absent it would be a hollow or fabricated fact
    downstream) rather than failing validation of the entire CV. Non-list input is
    passed through untouched so Pydantic raises its normal error if truly wrong-typed.

    Accepts both raw dicts (the tool-call JSON path via ``model_validate``) and already
    constructed ``BaseModel`` instances (the direct-construction path used in tests and
    by callers that build sub-models first). A valid instance is kept verbatim; only an
    item that is neither, or whose identity keys are missing/blank, is dropped.
    """
    if not isinstance(value, list):
        return value
    kept = []
    for item in value:
        if isinstance(item, dict):
            present = all(item.get(key) not in (None, "") for key in required_keys)
        elif isinstance(item, BaseModel):
            present = all(getattr(item, key, None) not in (None, "") for key in required_keys)
        else:
            present = False
        if present:
            kept.append(item)
    return kept


# --- Enums ---------------------------------------------------------------------


class CandidateLevel(StrEnum):
    """Seniority inferred from the job description."""

    NEW_GRAD = "new_grad"
    ENTRY = "entry"
    MID = "mid"
    SENIOR_IC = "senior_ic"
    MANAGER = "manager"
    DIRECTOR = "director"


class KeywordTier(StrEnum):
    """Evidence tier of a JD keyword in the two-tier evidence model.

    Field-agnostic — the tier is about the KIND of evidence a keyword needs, not any one
    profession. ``concrete`` — a specific NAMED thing stated literally in the CV: a named
    tool, software, system, certification/license, instrument, standard, or technology
    (e.g. "Epic EHR", "Westlaw", "AWS", "QuickBooks", "ACLS", "CNC"): surfaced ONLY when the
    term appears literally in the source CV; inference is forbidden (the hard fabrication
    line). ``competency`` — a method / practice / capability (e.g. "triage", "case strategy",
    "systems thinking", "curriculum design"): surfaced ONLY when the generator names a source
    anchor that a deterministic check confirms exists verbatim. Untagged keywords default to
    ``concrete`` — the conservative tier that requires literal evidence and never launders a
    named thing via an anchor.
    """

    CONCRETE = "concrete"
    COMPETENCY = "competency"


class TargetSection(StrEnum):
    """Target CV section a keyword maps to in the keyword plan."""

    CONTACT = "contact"
    SUMMARY = "summary"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    PROJECTS = "projects"


class ErrorStage(StrEnum):
    """This feature's subset of the pipeline error stages."""

    PARSE = "parse"
    EXTRACT = "extract"


# --- Error envelope -----------------------------------------------------------


class ErrorResponse(BaseModel):
    """Actionable error envelope returned on a halted stage. Never leaks the key."""

    error: str = Field(..., description="Actionable message; never leaks the API key")
    stage: ErrorStage = Field(..., description="Stage at which the failure occurred")


# --- CV facts sub-models ------------------------------------------------------
# Collections default []; non-core scalars are Optional with coercers so a CV that omits a
# field validates. CORE identity fields stay required (truth: an entry without its identity
# is dropped upstream by _drop_malformed_items, never admitted hollow).


class FactsLink(BaseModel):
    """A labelled hyperlink (e.g. LinkedIn, GitHub, portfolio)."""

    title: str | None = None  # a bare URL with no label is still a usable link
    url: str  # a link without a URL is meaningless — required (malformed links dropped upstream)

    @field_validator("title", mode="before")
    @classmethod
    def _norm_title(cls, v: Any) -> Any:
        return _coerce_optional_str(v)


class FactsPersonalInfo(BaseModel):
    """Contact block parsed from the CV header."""

    name: str  # every CV identifies a person
    email: str  # the primary contact channel; required
    location: str | None = None  # many CVs omit location
    phone: str | None = None
    links: list[FactsLink] = Field(default_factory=list)

    @field_validator("location", "phone", mode="before")
    @classmethod
    def _norm_optional(cls, v: Any) -> Any:
        return _coerce_optional_str(v)

    @field_validator("links", mode="before")
    @classmethod
    def _drop_bad_links(cls, v: Any) -> Any:
        return _drop_malformed_items(v, ("url",))


class FactsExperience(BaseModel):
    """A single work-experience entry.

    Dates are free-form strings, now Optional+coerced so an entry missing a date
    doesn't fail. role+company are the identity — required (an experience with neither
    is dropped upstream). Bullets verbatim, malformed items dropped.
    """

    role: str  # identity (entry dropped upstream if missing)
    company: str  # identity
    company_description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    bullets: list[str] = Field(default_factory=list)

    @field_validator(
        "company_description", "start_date", "end_date", "location", mode="before"
    )
    @classmethod
    def _norm_optional(cls, v: Any) -> Any:
        return _coerce_optional_str(v)

    @field_validator("bullets", mode="before")
    @classmethod
    def _norm_bullets(cls, v: Any) -> Any:
        return _coerce_str_list(v)


class FactsEducation(BaseModel):
    """A single education entry."""

    institution: str  # identity
    degree: str  # identity
    start_year: int | None = None
    end_year: int | None = None
    gpa: str | None = None

    @field_validator("start_year", "end_year", mode="before")
    @classmethod
    def _norm_year(cls, v: Any) -> Any:
        return _coerce_optional_year(v)

    @field_validator("gpa", mode="before")
    @classmethod
    def _norm_gpa(cls, v: Any) -> Any:
        return _coerce_optional_str(v)


class FactsProject(BaseModel):
    """A single project entry."""

    name: str  # identity
    description: str | None = None  # a project may be named without a blurb
    skills: list[str] = Field(default_factory=list)
    link: FactsLink | None = None

    @field_validator("description", mode="before")
    @classmethod
    def _norm_description(cls, v: Any) -> Any:
        return _coerce_optional_str(v)

    @field_validator("skills", mode="before")
    @classmethod
    def _norm_skills(cls, v: Any) -> Any:
        return _coerce_str_list(v)

    @field_validator("link", mode="before")
    @classmethod
    def _drop_bad_link(cls, v: Any) -> Any:
        # a malformed link object (no url) collapses to None rather than failing
        if isinstance(v, dict) and not v.get("url"):
            return None
        return v


class FactsCertificate(BaseModel):
    """A single certificate entry."""

    title: str  # identity
    issuer: str | None = None  # a certificate may be listed without a named issuer
    year: int | None = None
    link: FactsLink | None = None

    @field_validator("issuer", mode="before")
    @classmethod
    def _norm_issuer(cls, v: Any) -> Any:
        return _coerce_optional_str(v)

    @field_validator("year", mode="before")
    @classmethod
    def _norm_year(cls, v: Any) -> Any:
        return _coerce_optional_year(v)

    @field_validator("link", mode="before")
    @classmethod
    def _drop_bad_link(cls, v: Any) -> Any:
        if isinstance(v, dict) and not v.get("url"):
            return None
        return v


class FactsLanguage(BaseModel):
    """A single language proficiency entry (e.g. Native, C1, B2)."""

    language: str  # identity
    level: str | None = None  # a language may be listed without a formal level

    @field_validator("level", mode="before")
    @classmethod
    def _norm_level(cls, v: Any) -> Any:
        return _coerce_optional_str(v)


class FactsSkillGroup(BaseModel):
    """A skill group from the candidate's own Skills section (their label + their items).

    Truthful source of record for the candidate's declared skills: the category label and item
    strings are captured VERBATIM (never invented or reworded at extract time). ``category`` is
    optional — a flat skills list with no header is valid; identity is at least one non-blank
    item (an empty group is dropped upstream by ``CVFacts._drop_bad_skills``). Spoken/natural
    languages are NOT stored here — they live in ``CVFacts.languages`` (dedup).
    """

    category: str | None = None
    items: list[str] = Field(default_factory=list)

    @field_validator("category", mode="before")
    @classmethod
    def _norm_category(cls, v: Any) -> Any:
        return _coerce_optional_str(v)

    @field_validator("items", mode="before")
    @classmethod
    def _norm_items(cls, v: Any) -> Any:
        return _coerce_str_list(v)


class CVFacts(BaseModel):
    """Structured, truthful representation of the parsed CV.

    The factual source of record for downstream generation; all lists default empty.
    Malformed list items (missing their identity fields) are dropped before validation
    so one bad entry never fails the whole extract — resilience without admitting hollow
    facts (truth-preserving).
    """

    personal_info: FactsPersonalInfo
    experiences: list[FactsExperience] = Field(default_factory=list)
    education: list[FactsEducation] = Field(default_factory=list)
    projects: list[FactsProject] = Field(default_factory=list)
    certificates: list[FactsCertificate] = Field(default_factory=list)
    languages: list[FactsLanguage] = Field(default_factory=list)
    skills: list[FactsSkillGroup] = Field(default_factory=list)

    @field_validator("experiences", mode="before")
    @classmethod
    def _drop_bad_experiences(cls, v: Any) -> Any:
        return _drop_malformed_items(v, ("role", "company"))

    @field_validator("education", mode="before")
    @classmethod
    def _drop_bad_education(cls, v: Any) -> Any:
        return _drop_malformed_items(v, ("institution", "degree"))

    @field_validator("projects", mode="before")
    @classmethod
    def _drop_bad_projects(cls, v: Any) -> Any:
        return _drop_malformed_items(v, ("name",))

    @field_validator("certificates", mode="before")
    @classmethod
    def _drop_bad_certificates(cls, v: Any) -> Any:
        return _drop_malformed_items(v, ("title",))

    @field_validator("languages", mode="before")
    @classmethod
    def _drop_bad_languages(cls, v: Any) -> Any:
        return _drop_malformed_items(v, ("language",))

    @field_validator("skills", mode="before")
    @classmethod
    def _drop_bad_skills(cls, v: Any) -> Any:
        return _drop_empty_skill_groups(v)


# --- JD analysis (unchanged except resilience parity) -------------------------


class JdAnalysis(BaseModel):
    """Job-description analysis: requirements, keywords, and a keyword→section plan."""

    role_title: str | None = None  # a JD may lack an explicit title line
    company: str | None = None  # a JD may not name the company
    keywords: list[str] = Field(default_factory=list)
    requirements_must: list[str] = Field(default_factory=list)
    requirements_nice: list[str] = Field(default_factory=list)
    keyword_plan: dict[str, TargetSection] = Field(default_factory=dict)
    keyword_tiers: dict[str, KeywordTier] = Field(default_factory=dict)
    candidate_level: CandidateLevel = CandidateLevel.MID

    @field_validator("role_title", "company", mode="before")
    @classmethod
    def _norm_optional(cls, v: Any) -> Any:
        return _coerce_optional_str(v)

    @field_validator("keywords", "requirements_must", "requirements_nice", mode="before")
    @classmethod
    def _norm_lists(cls, v: Any) -> Any:
        return _coerce_str_list(v)

    @field_validator("candidate_level", mode="before")
    @classmethod
    def _default_unknown_level(cls, value: Any) -> Any:
        """Map an unknown / missing seniority to a neutral default (mid)."""
        if isinstance(value, CandidateLevel):
            return value
        try:
            return CandidateLevel(value)
        except (ValueError, TypeError):
            return CandidateLevel.MID

    @field_validator("keyword_plan", mode="before")
    @classmethod
    def _drop_unknown_sections(cls, value: Any) -> Any:
        """Drop keyword_plan entries whose section is outside TargetSection."""
        if not isinstance(value, dict):
            return {}
        allowed = {section.value for section in TargetSection}
        kept: dict[str, str] = {}
        for keyword, section in value.items():
            section_value = section.value if isinstance(section, TargetSection) else section
            if section_value in allowed:
                kept[keyword] = section_value
        return kept

    @field_validator("keyword_tiers", mode="before")
    @classmethod
    def _drop_unknown_tiers(cls, value: Any) -> Any:
        """Normalize keyword_tiers to a {keyword: tier} dict, dropping out-of-vocab tiers.

        Accepts BOTH shapes: the tool-call array of ``{"keyword": ..., "tier": ...}`` objects
        (the model emits an array — a single object-map alongside keyword_plan degraded
        gemini's function call, see tool_schemas), and a plain ``{keyword: tier}`` dict (direct
        construction / contract). An out-of-vocab tier is dropped rather than fatal; the dropped
        keyword then resolves to the CONCRETE default via ``tier_of`` (literal evidence required).
        """
        if isinstance(value, list):
            value = {
                item["keyword"]: item.get("tier")
                for item in value
                if isinstance(item, dict) and item.get("keyword")
            }
        if not isinstance(value, dict):
            return {}
        allowed = {tier.value for tier in KeywordTier}
        kept: dict[str, str] = {}
        for keyword, tier in value.items():
            tier_value = tier.value if isinstance(tier, KeywordTier) else tier
            if tier_value in allowed:
                kept[keyword] = tier_value
        return kept

    def tier_of(self, keyword: str) -> KeywordTier:
        """Resolve a keyword's evidence tier case-insensitively (default CONCRETE).

        CONCRETE is the conservative default for an untagged keyword: it requires literal
        CV evidence and can never be surfaced via a competency anchor (tier governs).
        """
        target = keyword.strip().lower()
        for tagged, tier in self.keyword_tiers.items():
            if tagged.strip().lower() == target:
                return tier
        return KeywordTier.CONCRETE

    def to_job_target(self) -> JobTarget:
        """Return a `cv_template.JobTarget` view of this analysis for downstream reuse."""
        return JobTarget(
            title=self.role_title or "",
            company=self.company or "",
            keywords=list(self.keywords),
        )


# --- Extract envelope ---------------------------------------------------------


class ExtractResult(BaseModel):
    """The single shared extract produced once per request."""

    facts: CVFacts
    jd: JdAnalysis
    flags: list[str] = Field(default_factory=list)
