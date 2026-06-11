"""Section-wise tailored-CV pipeline.

Hosts the pipeline scaffolding: tuning constants, the result/validation structures
and `FlagKind` enum, and the `{ error, stage }` envelope helper. The pipeline functions
(`generate_tailored_cv`, `validate_section`, `score_ats`, `assemble_and_gate`,
per-section generators) and the shared infrastructure live here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from openai import OpenAIError

from extract import _redact_api_key
from helprers.cv_template import (
    ActionVerb,
    BulletPoint,
    Category,
    Certificate,
    CVTemplate,
    Education,
    Experience,
    Language,
    Link,
    PersonalInfo,
    Project,
    Skills,
    Summary,
)
from helprers.llm_model import LLMModel, ProviderResponseError, bounded_gather
from helprers.prompts import Prompts
from helprers.text_preprocessing import TextPreprocessing
from schemas import (
    MAX_CONCURRENT_LLM_CALLS,
    MIN_KEYWORDS,
    CandidateLevel,
    CVFacts,
    ExtractResult,
    JdAnalysis,
    KeywordTier,
    TargetSection,
)
from tool_schemas import (
    generate_certificate_tool_schema,
    generate_education_tool_schema,
    generate_experience_tool_schema,
    generate_language_tool_schema,
    generate_project_tool_schema,
    generate_skills_tool_schema,
    generate_summary_tool_schema,
)

logger = logging.getLogger(__name__)

__all__ = [
    "REGEN_CAP",
    "COVERAGE_TARGET_PCT",
    "MIN_KEYWORDS",
    "SUMMARY_MIN_LINES",
    "SUMMARY_MAX_LINES",
    "BULLET_MAX_CHARS",
    "MAX_ROLES_NO_WARNING",
    "COMPANY_DESC_MIN_CHARS",
    "MAX_PAGES",
    "LINES_PER_PAGE",
    "FlagKind",
    "PipelineStage",
    "StandardHeading",
    "GLOBAL_SECTION",
    "SectionFlag",
    "AtsScore",
    "SectionValidation",
    "TailoredResult",
    "CoverLetterResult",
    "COVER_LETTER_MAX_WORDS",
    "COVER_LETTER_REGEN_CAP",
    "COVER_LETTER_SECTION",
    "to_pipeline_error",
    "generate_tailored_cv",
    "validate_section",
    "score_ats",
    "assemble_and_gate",
    "estimate_page_count",
]

# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline constants
# ──────────────────────────────────────────────────────────────────────────────

#: Regenerate a failing section at most this many times, then emit-with-flag.
REGEN_CAP: int = 2

#: Global gate passes when after_pct >= this percentage; below → did-not-converge.
COVERAGE_TARGET_PCT: int = 75

# MIN_KEYWORDS is re-exported from schemas.py (single source of truth) rather than
# redefined, so the extract-stage gap check and the ATS check cannot drift. Listed in
# __all__ so it is a real re-export.

#: Summary must be at least this many lines (error when below).
SUMMARY_MIN_LINES: int = 3

#: Summary must be at most this many lines (error when above).
SUMMARY_MAX_LINES: int = 5

#: Bullet `description` longer than this many chars warns.
BULLET_MAX_CHARS: int = 120

#: More than this many roles triggers a one-page-pressure warning.
MAX_ROLES_NO_WARNING: int = 3

#: `company_description` shorter than this many chars warns.
COMPANY_DESC_MIN_CHARS: int = 20

#: One-page budget enforced at the global gate.
MAX_PAGES: int = 1

#: Estimated rendered lines that fit on a single page; the global gate compresses
#: overflow above ``MAX_PAGES * LINES_PER_PAGE`` (internal heuristic).
LINES_PER_PAGE: int = 50

#: Skills rendered per line when estimating page count; each skill group costs
#: ``ceil(len(group) / SKILLS_PER_LINE)`` lines so a skill-heavy CV is not flattened to
#: ~3 lines and silently slipped past the one-page gate (internal heuristic).
SKILLS_PER_LINE: int = 6

# Cover-letter tuning constants. Consumed by cover_letter.py, kept beside the CV pipeline
# constants so the shared FlagKind/SectionFlag and these caps live together.

#: Regenerate the cover letter once when word_count(draft) > this; if still over, flag.
COVER_LETTER_MAX_WORDS: int = 300

#: Max regens for a cover letter that is over-length OR structurally malformed (regens <= cap;
#: follows the same capped-regen pattern as ``REGEN_CAP``). Raised to 2 so a single budget covers
#: both a compress pass and a structural re-roll (cover_letter._is_well_structured).
COVER_LETTER_REGEN_CAP: int = 2


# ──────────────────────────────────────────────────────────────────────────────
#  Result / validation structures + enums
# ──────────────────────────────────────────────────────────────────────────────


class FlagKind(StrEnum):
    """Non-fatal flag kinds carried to the API `flags[]` array.

    These values are canonical — the API/UI layer consumes them verbatim. Do not rename
    without updating the downstream UI mapping.
    """

    CAPPED_SECTION = "capped_section"
    DID_NOT_CONVERGE = "did_not_converge"
    ONE_PAGE_PRESSURE = "one_page_pressure"
    UNMET_COVERAGE = "unmet_coverage"
    COVER_LETTER_GAP = "cover_letter_gap"
    COVER_LETTER_OVER_LENGTH = "cover_letter_over_length"
    COVER_LETTER_NO_REQUIREMENTS = "cover_letter_no_requirements"


class PipelineStage(StrEnum):
    """Pipeline stages reported in the `{ error, stage }` envelope."""

    GENERATE = "generate"
    VALIDATE = "validate"
    ASSEMBLE = "assemble"


class StandardHeading(StrEnum):
    """ATS-standard section headings; a non-standard heading is an error."""

    SUMMARY = "Summary"
    SKILLS = "Skills"
    EXPERIENCE = "Experience"
    EDUCATION = "Education"
    PROJECTS = "Projects"
    CERTIFICATIONS = "Certifications"
    LANGUAGES = "Languages"


#: Sentinel ``section`` for flags that are a CV-wide / global-gate outcome, not attributable
#: to one rendered section: coverage shortfall and residual one-page overflow.
GLOBAL_SECTION = "global"

#: Sentinel ``section`` for cover-letter flags (parallels GLOBAL_SECTION). The
#: cover letter is not a TargetSection, so its gap / over-length flags carry this value.
COVER_LETTER_SECTION = "cover_letter"


@dataclass
class SectionFlag:
    """A non-fatal marker for a section (or the global gate).

    ``section`` is ``TargetSection | str`` because two rendered sections — "certificates"
    and "languages" — are not TargetSection members. The flag carries the real section
    identity verbatim so the consumer never mis-attributes a capped languages/certificates
    section to the summary.
    """

    section: TargetSection | str
    kind: FlagKind
    message: str


@dataclass
class AtsScore:
    """Before→after keyword-coverage result — the visible "Checker"."""

    before_pct: float
    after_pct: float
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Coverage is a percentage in [0, 100] by construction (100.0 is valid — full coverage).
        # Fail loud if a scoring bug produces an out-of-range value rather than silently clamping:
        # the error then surfaces through the {error, stage} envelope instead of shipping a
        # nonsensical ATS number to the user.
        for name, value in (("before_pct", self.before_pct), ("after_pct", self.after_pct)):
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"AtsScore.{name} must be in [0, 100], got {value}")

    @property
    def coverage_pct(self) -> float:
        """The headline coverage == ``after_pct`` (computed, never stored, so they can't drift)."""
        return self.after_pct


@dataclass
class SectionValidation:
    """Deterministic per-section validation outcome.

    `errors` are blocking (trigger regeneration); `warnings` are non-blocking
    (including ActionVerb out-of-enum).
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TailoredResult:
    """The pipeline output returned by `generate_tailored_cv`."""

    cv: CVTemplate
    ats_score: AtsScore
    flags: list[SectionFlag] = field(default_factory=list)


@dataclass
class CoverLetterResult:
    """The cover-letter output returned by `generate_cover_letter`.

    Follows the ``TailoredResult`` shape — a result field plus non-fatal ``flags`` — minus
    the ATS score. Not persisted. ``flags`` carries non-fatal ``cover_letter_over_length`` /
    ``cover_letter_no_requirements`` markers (default empty); a residual over-length or a JD
    with no extractable requirements is never a fatal error. (The ``cover_letter_gap`` kind is
    reserved for the degradation path — an empty letter when generation fails outright — not
    emitted by this success path.)
    """

    cover_letter: str
    flags: list[SectionFlag] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
#  Error envelope helper
# ──────────────────────────────────────────────────────────────────────────────


def to_pipeline_error(stage: PipelineStage, message: str) -> dict[str, str]:
    """Build the `{ error, stage }` envelope for a halted pipeline stage.

    Reuses `extract._redact_api_key` so the OpenRouter API key and raw provider errors
    never leak to the client. `stage` is one of generate|validate|assemble; today only
    `generate` is ever emitted — validation/coverage gaps surface as flags, never an
    envelope — so validate/assemble are reserved for future halting stages.
    """
    return {"error": _redact_api_key(message), "stage": stage.value}


# ──────────────────────────────────────────────────────────────────────────────
#  Cleanup + small mapping helpers
# ──────────────────────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Run the shared cleanup over a generated text field.

    clean() (strip + humanize) then remove_ai_tells(english=True) — the exact order the
    text-cleanup layer mandates, so coverage matching runs on cleaned text and a homoglyph
    never undercounts.
    """
    return TextPreprocessing.remove_ai_tells(TextPreprocessing.clean(text), english=True)


def _require_fields(
    data: dict, keys: tuple[str, ...], context: str, *, non_empty: bool = False
) -> None:
    """Raise an actionable ``ProviderResponseError`` for a missing/null required field.

    ``call_tool`` already raises ``ProviderResponseError`` for a malformed provider reply,
    so anything reaching the builders should be a parsed dict. A model that omits OR nulls a
    required field is a provider-shape gap (not a bug in our code) — a null is treated like a
    missing field, because the builders would otherwise crash with a bare ``TypeError``
    (``_clean_text(None)`` / iterating ``None``) that escapes the orchestrator's typed-error
    handler as a 500. We name the field and raise the typed generate-stage error here instead.

    ``non_empty``: when set, the listed keys must be NON-BLANK STRINGS — a
    present-but-empty/whitespace value (or a non-string) is a
    provider-shape gap in the same class as a missing field. ``_require_fields`` alone
    accepts ``""`` (``data.get(key) is None`` is False), so a required text field that
    arrives blank silently slips through; callers whose field is the whole payload (the
    cover letter) or a rendered line (a CV section's text) pass ``non_empty=True``.
    """
    if not isinstance(data, dict):
        raise ProviderResponseError(f"{context} is not a JSON object.")
    missing = [key for key in keys if data.get(key) is None]
    if missing:
        raise ProviderResponseError(
            f"{context} missing required field(s): {', '.join(missing)}."
        )
    if non_empty:
        blank = [key for key in keys if not (isinstance(data[key], str) and data[key].strip())]
        if blank:
            raise ProviderResponseError(
                f"{context} has empty or non-text required field(s): {', '.join(blank)}."
            )


def _string_list(value: object) -> list[str]:
    """Return the non-empty string items of a model-supplied list field (shared shape guard).

    A function-calling model does not guarantee the shape of an array field. A bare string
    would be iterated CHARACTER by character (one junk item per char), a dict by
    keys, and non-string items rendered as repr junk. We accept ONLY a list and keep only its
    non-blank string items; any other shape (or a missing field) yields ``[]`` — a malformed
    field is degraded, never fatal. Shared by the CV skills lists and the cover-letter
    ``gaps`` field so the guard lives once at the primitive, not re-patched per feature.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _coerce_action_verb(value: str) -> ActionVerb | str:
    """Coerce a model action verb to ActionVerb when it is a member, else keep the str.

    Truth-preserving: membership is warning-only, so an out-of-enum verb is
    stored verbatim rather than forced into the enum (which would raise).
    """
    try:
        return ActionVerb(value)
    except ValueError:
        return value


def _build_link(data: dict | None) -> Link | None:
    """Build a `Link` from tool-call JSON, or None when absent."""
    if not data:
        return None
    # Only `url` is the link's identity (a bare URL with no anchor text is a valid link — the
    # extract makes `title` Optional). `title` is kept only when a clean non-blank string.
    _require_fields(data, ("url",), "Link")
    title = data.get("title")
    title = title.strip() if isinstance(title, str) and title.strip() else None
    return Link(title=title, url=data["url"])


def _section_user_prompt(template: str, facts: CVFacts, jd: JdAnalysis) -> str:
    """Fill a section user-prompt template with the per-request facts / JD / plan."""
    plan = {keyword: section.value for keyword, section in jd.keyword_plan.items()}
    return template.format(
        facts=facts.model_dump_json(),
        jd=jd.model_dump_json(),
        keyword_plan=json.dumps(plan),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Per-section generators
# ──────────────────────────────────────────────────────────────────────────────


async def _generate_summary(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> Summary:
    """Generate the tailored summary (🟡 keyword surface) via function calling."""
    result = await llm.call_tool(
        Prompts.GENERATE_SUMMARY_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_SUMMARY_USER, facts, jd),
        generate_summary_tool_schema(),
    )
    _require_fields(result, ("text", "relevant_skills"), "Summary")
    _require_fields(result, ("text",), "Summary", non_empty=True)
    return Summary(
        text=_clean_text(result["text"]),
        relevant_skills=[_clean_text(s) for s in _string_list(result["relevant_skills"])],
    )


#: Header for the one evidenced-but-not-JD category permitted in Skills (the gold standard).
SPOKEN_LANGUAGES_CATEGORY = "Spoken Languages"

#: Domain-neutral FORMAT bounds for an emergent Skills category header. The
#: VOCABULARY of a header emerges from the candidate's own field (never a hardcoded taxonomy);
#: only its SHAPE is constrained, deterministically and field-agnostically: a header is normalized
#: to a single concept of at most ``LABEL_MAX_WORDS`` words and ``LABEL_MAX_CHARS`` chars.
LABEL_MAX_CHARS: int = 24
LABEL_MAX_WORDS: int = 3

#: Splitter for a MULTI-CONCEPT header: once a header is judged multi-concept
#: (see ``_HEADER_JOIN_RE``), it is split on every concept-joiner — ",", "&", "/", or the
#: standalone word "and" (word-boundary, case-insensitive — never the "and" inside "Brand" /
#: "Standards") — and reduced to ONE concept. Structural only, no vocabulary.
_HEADER_SPLIT_RE = re.compile(r"\s*(?:,|&|/|\band\b)\s*", re.IGNORECASE)

#: Conjunction joiners only (NOT comma), used to COUNT joiners so the single-joiner rule can be
#: applied: a label is multi-concept ONLY when it has a comma OR more than one of these. A single
#: "&", "/", or "and" ("Health & Safety", "CI/CD", "Research and Development") is part of an ATOMIC
#: header and is kept whole — never split.
_HEADER_JOIN_RE = re.compile(r"\s*(?:&|/|\band\b)\s*", re.IGNORECASE)

#: Strip a dangling leading/trailing conjunction the word/char cap may leave on an atomic header
#: ("Programming Languages & Tools" → cap → "Programming Languages &" → "Programming Languages").
_HEADER_DANGLING_JOIN_RE = re.compile(
    r"^\s*(?:&|/|\band\b)\s*|\s*(?:&|/|\band\b)\s*$", re.IGNORECASE
)

#: Markdown / emphasis control chars a free-text label may carry. ``_clean_text`` alone leaves
#: paired emphasis like ``**Bold**`` intact, so the header is stripped of these first to guarantee
#: the rendered AND copied Skills text carries ZERO markup (plain-text invariant).
_LABEL_MARKDOWN_RE = re.compile(r"[*_`#~]+")


def _clean_label(raw: object) -> str:
    """Strip markdown/control chars from a free-text category header, then run shared cleanup.

    The model emits the category header as free text and may wrap it in markdown (``**Tools**``).
    Non-string input collapses to the empty label (rendered ungrouped). Deterministic — the label
    SHAPE is constrained downstream, but its markup is always removed here so render == copy text.
    """
    if not isinstance(raw, str):
        return ""
    return _clean_text(_LABEL_MARKDOWN_RE.sub("", raw))


def _normalize_header(raw: object) -> str:
    """Reduce ANY model-proposed category header to a clean single-concept label.

    Pure + domain-neutral: STRUCTURAL rules only — no vocabulary, no taxonomy, no profession
    assumptions, so it behaves identically for a nurse, lawyer, chef, or developer. A clean header
    is an INVARIANT BY CONSTRUCTION here (never validated-and-rejected): a malformed header is
    unrepresentable rather than a regen trigger. Steps:

    1. Strip markdown / control chars and collapse whitespace (``_clean_label``).
    2. single-joiner rule: a label is MULTI-CONCEPT only when it has a comma OR more than one
       conjunction joiner ("&", "/", or the word "and"). A SINGLE "&", "/", or "and" is part of an
       ATOMIC header and is kept WHOLE — "Health & Safety", "CI/CD", "Research and Development",
       "Frameworks/Libraries" survive verbatim. ONLY a multi-concept label is split (on every
       joiner, comma included) and reduced to ONE segment — the one with the GREATEST word count;
       ties broken by FIRST (``max`` is first-wins). So "CI/CD, Testing & AI Integration" →
       "AI Integration".
    3. Cap to <= ``LABEL_MAX_WORDS`` words (drop trailing words), then hard-trim to <=
       ``LABEL_MAX_CHARS`` chars. Finally strip any dangling leading/trailing conjunction the cap
       may have left ("Programming Languages & Tools" → cap → "Programming Languages &" →
       "Programming Languages").

    The result is the header that renders AND is copied. An empty result (markup-only / blank
    input) is the SOLE headerless case — that one category renders ungrouped, never an invented
    label. Coverage-neutral: only the cosmetic label is shaped; no keyword text is touched.
    """
    cleaned = _clean_label(raw)
    if not cleaned:
        return ""
    multi_concept = ("," in cleaned) or (len(_HEADER_JOIN_RE.findall(cleaned)) > 1)
    if multi_concept:
        segments = [seg for seg in (s.strip() for s in _HEADER_SPLIT_RE.split(cleaned)) if seg]
        if len(segments) > 1:
            cleaned = max(segments, key=lambda seg: len(seg.split()))
    words = cleaned.split()
    if len(words) > LABEL_MAX_WORDS:
        cleaned = " ".join(words[:LABEL_MAX_WORDS])
    if len(cleaned) > LABEL_MAX_CHARS:
        cleaned = cleaned[:LABEL_MAX_CHARS].strip()
    return _HEADER_DANGLING_JOIN_RE.sub("", cleaned).strip()


def _normalize_skill_headers(skills: object) -> None:
    """Assembly-time invariant: normalize EVERY emergent category header to a clean shape AND
    merge siblings that collapse to the same label.

    Replaces the old validate→reject→regen→ungrouped-fallback path. ``_normalize_header`` makes a
    malformed header unrepresentable, so there is nothing to reject, no header regen, and no
    internal diagnostic to leak. ``_generate_skills`` groups keywords by the RAW model label, so
    two raw labels that normalize to the SAME header (e.g. "Programming Languages & Tools" and
    "Programming Languages, Frameworks" → "Programming Languages") would otherwise render as
    duplicate sibling lines; normalizing then MERGING by normalized label collapses them into one
    category — keywords concatenated with case-insensitive first-seen dedup, first-seen label order
    preserved, empty merged labels dropped. Spoken Languages is facts-sourced and already canonical,
    so it is left untouched and kept LAST (render order). ``Skills.provenance`` is keyed by
    keyword (not category), so it is unaffected. Coverage-neutral: only the cosmetic label/grouping
    changes; the keyword text the scorer reads is never touched.
    """
    categories = getattr(skills, "categories", None) or []
    spoken = [c for c in categories if c.category == SPOKEN_LANGUAGES_CATEGORY]
    merged: dict[str, Category] = {}
    order: list[str] = []
    for category in categories:
        if category.category == SPOKEN_LANGUAGES_CATEGORY:
            continue
        label = _normalize_header(category.category)
        bucket = merged.get(label)
        if bucket is None:
            bucket = Category(category=label, keywords=[])
            merged[label] = bucket
            order.append(label)
        seen = {kw.strip().lower() for kw in bucket.keywords}
        for kw in category.keywords:
            key = kw.strip().lower()
            if key not in seen:
                seen.add(key)
                bucket.keywords.append(kw)
    skills.categories = [merged[label] for label in order if merged[label].keywords] + spoken


@dataclass
class SkillProvenance:
    """Validation-time evidence record for one surfaced Skills keyword (two-tier model).

    ``tier`` is the authoritative tier (``jd.tier_of``), not the model's self-report, so a
    concrete keyword can never be laundered through a competency anchor. ``anchor_ref`` is the
    verbatim source line a competency was surfaced against (None for a concrete keyword, whose
    evidence is its own literal presence). ``declared`` is True when the keyword is one of the
    candidate's own ``facts.skills`` items (a FACT, kept by default) vs an ADDED JD keyword — the
    cross-section subsumption pass reads it so a declared skill is never deduped away. Carried on
    ``Skills.provenance``; never rendered.
    """

    keyword: str
    tier: KeywordTier
    anchor_ref: str | None = None
    declared: bool = False


def _anchor_traces(anchor_ref: object, source_text_norm: str) -> bool:
    """True when ``anchor_ref`` is a (normalized) verbatim substring of the source CV text.

    Deterministic anchor-existence trace (no LLM): the generator must copy the anchor verbatim
    from the CV, so a case-insensitive substring match against the homoglyph-normalized source
    confirms the anchor EXISTS. It does NOT judge whether the anchor demonstrates the competency
    — that mapping is the generator's accepted Tier-2 judgment. Empty/non-string → never traces.
    """
    if not isinstance(anchor_ref, str):
        return False
    needle = _normalize_for_match(anchor_ref).strip().lower()
    if not needle:
        return False
    return needle in source_text_norm.lower()


def _surface_skill(
    item: object,
    jd: JdAnalysis,
    jd_keywords: set[str],
    declared: set[str],
    evidence: str,
    outer_label: str,
) -> tuple[str, SkillProvenance, str] | None:
    """Decide whether one model-emitted skill survives (union-with-tailoring, not intersect).

    Returns ``(keyword, provenance, emergent_label)`` or None. The label is the model's own
    per-keyword ``category`` (a free-text header derived from the candidate's field), falling
    back to the group's ``outer_label`` when the per-keyword field is absent; it is cleaned of
    markdown but otherwise carried verbatim (format-checked downstream — never coerced into a
    fixed taxonomy).

    Inclusion policy (locked): a keyword the candidate DECLARED (``declared`` — sourced from
    ``facts.skills``) is a FACT and surfaces ungated — no JD membership, no tier bar, no anchor
    (the candidate stated it). Otherwise the keyword is an ADDITION and the two-tier standard
    governed by the tier (``jd.tier_of`` — tier governs, never the model's self-report) is the
    ONLY gated path: it must be a JD keyword AND meet its tier's evidence bar — CONCRETE → the term
    is literally present in the CV (inference forbidden); COMPETENCY → a model-supplied
    ``anchor_ref`` traces verbatim to the source CV. A non-declared, non-JD keyword, or a
    competency with no/untraceable anchor, never surfaces. The bare-umbrella drop is model-side
    judgment (skills prompt), not a hardcoded code-side lexicon.
    """
    if isinstance(item, str):
        keyword_raw, anchor_raw, label_raw = item, None, None
    elif isinstance(item, dict):
        keyword_raw = item.get("keyword")
        anchor_raw = item.get("anchor_ref")
        label_raw = item.get("category")
    else:
        return None
    # Shape-guard EVERY model-supplied field (the function-calling model does not guarantee an
    # array item's shape): a non-string keyword (e.g. {"keyword": 3}) would crash `_clean_text`
    # with a bare TypeError that escapes the typed-error handler as a 500 — drop it instead.
    # ``anchor_raw``/``label_raw`` are already isinstance-guarded downstream.
    if not isinstance(keyword_raw, str):
        return None
    keyword = _clean_text(keyword_raw)
    if not keyword:
        return None
    label_source = label_raw if isinstance(label_raw, str) and label_raw.strip() else outer_label
    label = _clean_label(label_source)
    # BASE: a declared skill is a truthful fact — surfaced by default, no JD membership or anchor.
    # Its provenance tier is forced CONCRETE (the declared item is literally in the evidence text),
    # so the deterministic validator keeps it via the literal-presence path, never the anchor path.
    if keyword.strip().lower() in declared:
        prov = SkillProvenance(
            keyword=keyword, tier=KeywordTier.CONCRETE, anchor_ref=None, declared=True
        )
        return keyword, prov, label
    # ADD (the only gated path): a JD keyword the candidate has not declared, under the tier rules.
    if keyword.strip().lower() not in jd_keywords:
        return None
    tier = jd.tier_of(keyword)
    if tier is KeywordTier.CONCRETE:
        if not _term_present(keyword, evidence):
            return None  # named thing with no literal CV presence — the interview-failure case
        return keyword, SkillProvenance(keyword=keyword, tier=tier, anchor_ref=None), label
    if not _anchor_traces(anchor_raw, evidence):
        return None  # competency with no/untraceable anchor — dropped
    return keyword, SkillProvenance(keyword=keyword, tier=tier, anchor_ref=anchor_raw), label


def _declared_skill_index(facts: CVFacts) -> dict[str, tuple[str, str]]:
    """Map each declared skill (normalized) → (verbatim item, its group's category label).

    The candidate's own Skills section (``facts.skills``) is the truthful source for the BASE
    skills: every declared item is a fact, surfaced by default and grouped under its own category
    label (the generator seeds/reframes that label toward the JD). First occurrence wins on a
    case-insensitive duplicate so the same skill is never stored twice.
    """
    index: dict[str, tuple[str, str]] = {}
    for group in facts.skills:
        label = group.category or ""
        for item in group.items:
            cleaned = _clean_text(item)
            key = cleaned.strip().lower()
            if key and key not in index:
                index[key] = (cleaned, label)
    return index


def _spoken_language_keywords(facts: CVFacts) -> list[str]:
    """Render the candidate's real spoken languages as dense keywords ("English fluent").

    Sourced directly from ``facts.languages`` (always evidenced), so the Spoken Languages
    category is the only evidenced-but-not-JD group that may surface in Skills, and it can
    never introduce an unevidenced keyword.
    """
    out: list[str] = []
    for lang in facts.languages:
        name = (lang.language or "").strip()
        if not name:
            continue
        level = (lang.level or "").strip()
        out.append(f"{name} {level.lower()}" if level else name)
    return out


async def _generate_skills(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> Skills:
    """Generate the skills section (🟡🟠) as JD-derived categories via function calling.

    Union-with-tailoring: the candidate's DECLARED skills
    (``facts.skills``) are facts — surfaced by default and grouped under their own category labels
    (reframed/foregrounded toward the JD by the generator), never dropped merely for not being a JD
    keyword. Any declared skill the model omits is backfilled here so the inventory is never
    silently lost. ADDING a JD keyword the candidate has NOT declared is the only gated path:
    CONCRETE keywords must appear literally in the CV (no inference), COMPETENCY keywords must carry
    an anchor_ref that traces verbatim to the source. Coverage can rise only by surfacing
    genuinely-evidenced keywords, never by inventing them. Per surfaced keyword a
    ``SkillProvenance`` (keyword, tier, anchor_ref) rides ``Skills.provenance`` for the
    deterministic validator — never rendered or scored.

    Emergent categories: the model groups each surfaced keyword under a short
    header it DERIVES from the candidate's own field (free text — no fixed taxonomy), seeded from
    the candidate's own group labels. Surviving keywords are regrouped by that emergent label and
    emitted in STABLE first-surfaced order; the label's SHAPE is format-checked downstream
    (validator + post-cap fallback), never coerced into a hardcoded vocabulary. Spoken languages
    are appended last from ``facts.languages`` — the one evidenced-but-not-JD category permitted.
    """
    result = await llm.call_tool(
        Prompts.GENERATE_SKILLS_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_SKILLS_USER, facts, jd),
        generate_skills_tool_schema(),
    )
    _require_fields(result, ("categories",), "Skills")
    # Shape-guard: a non-list ``categories`` (string/dict) degrades to empty here rather than
    # being iterated per char/key — the empty Skills is then caught by the validator + regen,
    # but we avoid spending a regen cycle on a shape we can reject immediately.
    raw_categories = result["categories"] if isinstance(result["categories"], list) else []
    evidence = _normalize_for_match(_facts_to_text(facts))
    jd_keywords = {kw.strip().lower() for kw in jd.keywords if kw.strip()}
    declared = _declared_skill_index(facts)
    declared_norm = set(declared.keys())
    # Regroup every surviving keyword by its emergent label (first-surfaced order preserved).
    by_label: dict[str, list[str]] = {}
    label_order: list[str] = []
    provenance: list[SkillProvenance] = []
    surfaced_norm: set[str] = set()

    def _add(keyword: str, label: str, prov: SkillProvenance) -> None:
        key = keyword.strip().lower()
        if key in surfaced_norm:
            return  # cross-section dedup: a skill is surfaced once, under its first label
        surfaced_norm.add(key)
        if label not in by_label:
            by_label[label] = []
            label_order.append(label)
        by_label[label].append(keyword)
        provenance.append(prov)

    for raw in raw_categories:
        if not isinstance(raw, dict):
            continue
        raw_keywords = raw.get("keywords")
        if not isinstance(raw_keywords, list):
            continue  # a non-list keywords field degrades to empty (never iterated per char)
        outer_label = raw.get("category") if isinstance(raw.get("category"), str) else ""
        for item in raw_keywords:
            surfaced = _surface_skill(item, jd, jd_keywords, declared_norm, evidence, outer_label)
            if surfaced is not None:
                keyword, prov, label = surfaced
                _add(keyword, label, prov)
    # Backfill: every declared skill is a fact, included by default (policy #1) — add any the
    # model omitted, under its own (seeded) category label, after the foregrounded model set.
    for key, (original, cat_label) in declared.items():
        if key not in surfaced_norm:
            _add(
                original,
                _clean_label(cat_label),
                SkillProvenance(
                    keyword=original, tier=KeywordTier.CONCRETE, anchor_ref=None, declared=True
                ),
            )
    categories = [
        Category(category=label, keywords=by_label[label])
        for label in label_order
        if by_label[label]
    ]
    spoken = _spoken_language_keywords(facts)
    if spoken:
        categories.append(Category(category=SPOKEN_LANGUAGES_CATEGORY, keywords=spoken))
    return Skills(categories=categories, provenance=provenance)


def _build_experience(item: dict) -> Experience:
    """Map one experience tool-call entry to an `Experience` dataclass."""
    _require_fields(item, ("role", "company"), "Experience entry")
    bullets = []
    for bullet in item.get("bullets", []):
        _require_fields(bullet, ("action_verb", "description"), "Experience bullet")
        bullets.append(
            BulletPoint(
                action_verb=_coerce_action_verb(bullet["action_verb"]),
                description=_clean_text(bullet["description"]),
                skills=list(bullet.get("skills", [])),
                impact=_clean_text(bullet["impact"]) if bullet.get("impact") else None,
                benefit=_clean_text(bullet["benefit"]) if bullet.get("benefit") else None,
            )
        )
    # start_date/end_date are absent-able facts (an ongoing role, or a source that omits dates):
    # a blank/whitespace value is a truthful "no date" (None), never a fabricated one; a genuine
    # value ("2021-03"/"Present") is kept. The integrity guard still rejects an INVENTED date.
    start_date = (item.get("start_date") or "").strip() or None
    end_date = (item.get("end_date") or "").strip() or None
    return Experience(
        role=item["role"],
        company=item["company"],
        company_description=_clean_text(item.get("company_description") or ""),
        start_date=start_date,
        end_date=end_date,
        location=item.get("location"),
        bullets=bullets,
    )


async def _generate_experience(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> list[Experience]:
    """Generate the experience section; companies/dates are validated against source."""
    if not facts.experiences:
        return []
    result = await llm.call_tool(
        Prompts.GENERATE_EXPERIENCE_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_EXPERIENCE_USER, facts, jd),
        generate_experience_tool_schema(),
    )
    return [_build_experience(item) for item in result.get("experiences", [])]


async def _generate_education(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> list[Education]:
    """Pass-through education from source facts; [] when none present."""
    if not facts.education:
        return []
    result = await llm.call_tool(
        Prompts.GENERATE_EDUCATION_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_EDUCATION_USER, facts, jd),
        generate_education_tool_schema(),
    )
    items = result.get("education", [])
    for item in items:
        _require_fields(item, ("institution", "degree"), "Education entry")
    return [
        Education(
            institution=item["institution"],
            degree=item["degree"],
            start_year=item.get("start_year"),
            end_year=item.get("end_year"),
            gpa=item.get("gpa"),
        )
        for item in items
    ]


async def _generate_project(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> list[Project]:
    """Pass-through projects ONLY when present in source (never invented)."""
    if not facts.projects:
        return []
    result = await llm.call_tool(
        Prompts.GENERATE_PROJECT_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_PROJECT_USER, facts, jd),
        generate_project_tool_schema(),
    )
    items = result.get("projects", [])
    for item in items:
        # `name` is the project's identity; `description` is Optional (the extract permits a named
        # project with no blurb — requiring it here only pressures the model to fabricate one).
        _require_fields(item, ("name",), "Project")
    return [
        Project(
            name=item["name"],
            description=_clean_text(item["description"])
            if isinstance(item.get("description"), str) and item["description"].strip()
            else None,
            skills=list(item.get("skills", [])),
            link=_build_link(item.get("link")),
        )
        for item in items
    ]


async def _generate_certificate(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> list[Certificate]:
    """Pass-through certificates ONLY when present in source (never invented)."""
    if not facts.certificates:
        return []
    result = await llm.call_tool(
        Prompts.GENERATE_CERTIFICATE_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_CERTIFICATE_USER, facts, jd),
        generate_certificate_tool_schema(),
    )
    items = result.get("certificates", [])
    for item in items:
        # `title` is the certificate's identity; `issuer` is Optional (the extract permits a
        # certificate with no named issuer — requiring it only pressures the model to invent one).
        _require_fields(item, ("title",), "Certificate")
    return [
        Certificate(
            title=item["title"],
            issuer=item["issuer"].strip()
            if isinstance(item.get("issuer"), str) and item["issuer"].strip()
            else None,
            year=item.get("year") or 0,
            link=_build_link(item.get("link")),
        )
        for item in items
    ]


async def _generate_language(facts: CVFacts, jd: JdAnalysis, llm: LLMModel) -> list[Language]:
    """Pass-through languages ONLY when present in source (never invented)."""
    if not facts.languages:
        return []
    result = await llm.call_tool(
        Prompts.GENERATE_LANGUAGE_SYSTEM,
        _section_user_prompt(Prompts.GENERATE_LANGUAGE_USER, facts, jd),
        generate_language_tool_schema(),
    )
    items = result.get("languages", [])
    for item in items:
        # `language` is identity; `level` is an absent-able fact (a CV may list a language with no
        # stated proficiency) — requiring it only pressures the model to fabricate one.
        _require_fields(item, ("language",), "Language")
    return [
        Language(
            language=item["language"],
            level=(item.get("level") or "").strip() or None,
        )
        for item in items
    ]


def _carry_personal_info(facts: CVFacts) -> PersonalInfo:
    """Carry contact info truthfully from CVFacts — never model-generated."""
    source = facts.personal_info
    return PersonalInfo(
        name=source.name,
        location=source.location,
        email=source.email,
        phone=source.phone,
        links=[Link(title=link.title, url=link.url) for link in source.links],
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Section validation
# ──────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^\S+@\S+\.\S+$")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

#: Allowed language proficiency levels (LanguageLevel).
_LANGUAGE_LEVELS = frozenset(
    {
        "Native",
        "Fluent",
        "Professional",
        "Intermediate",
        "Basic",
        "A1",
        "A2",
        "B1",
        "B2",
        "C1",
        "C2",
    }
)

#: Normalized (casefolded) language levels for factual-integrity comparison.
_LANGUAGE_LEVELS_NORM = {level.lower() for level in _LANGUAGE_LEVELS}

#: Section name → ATS StandardHeading (contact carries no rendered heading).
#: Keyed by rendered-section name — TargetSection values plus the two string-keyed
#: pass-through sections ("certificates", "languages") that are not TargetSection members.
#: Every rendered section MUST appear here (or in _NO_HEADING); a gap force-fails that
#: section's heading check on every run (the languages bug).
_SECTION_HEADINGS: dict[str, StandardHeading] = {
    TargetSection.SUMMARY.value: StandardHeading.SUMMARY,
    TargetSection.SKILLS.value: StandardHeading.SKILLS,
    TargetSection.EXPERIENCE.value: StandardHeading.EXPERIENCE,
    TargetSection.EDUCATION.value: StandardHeading.EDUCATION,
    TargetSection.PROJECTS.value: StandardHeading.PROJECTS,
    "certificates": StandardHeading.CERTIFICATIONS,
    "languages": StandardHeading.LANGUAGES,
}

# The ATS MIN_KEYWORDS floor is applied to the skills section — the primary JD-keyword
# surface. Applying it per-section (summary/experience too) would over-constrain and cap
# legitimate sections; the CV-level keyword presence is carried by
# skills.categories[].keywords, which is what coverage scoring reads.
_KEYWORD_BEARING = {TargetSection.SKILLS.value}

#: Sections that have no rendered ATS heading (so the heading check is skipped).
_NO_HEADING = {TargetSection.CONTACT.value}


def _norm_key(value: str | None) -> str:
    """Normalize a value for factual-integrity comparison (trim + casefold).

    Comparing generated content to source on a normalized key catches INVENTED
    content without falsely rejecting a truthfully recased company/level/institution
    ("Acme corp" traces to source "Acme Corp"; level "native" matches "Native").
    """
    return (value or "").strip().lower()


def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    """Count distinct JD keywords present (whole-word) in ``text``.

    Word-boundary matching (via ``_term_present``), consistent with ``score_ats`` —
    a substring like "Go" never falsely matches inside "Google".
    """
    normalized = _normalize_for_match(text)
    seen = {kw.strip().lower() for kw in keywords if kw.strip()}
    return sum(1 for kw in seen if _term_present(kw, normalized))


def _validate_contact(personal: PersonalInfo) -> SectionValidation:
    """Parse-friendly contact checks: name, email structural, links url."""
    result = SectionValidation()
    if not (personal.name or "").strip():
        result.errors.append("Contact name is empty.")
    if not _EMAIL_RE.match(personal.email or ""):
        result.errors.append(f"Contact email is not parse-friendly: {personal.email!r}")
    # phone is optional and presence-only — never an error.
    for link in personal.links:
        if not _URL_RE.match(link.url):
            result.errors.append(f"Link url must match ^https?://: {link.url!r}")
    return result


def _validate_summary(summary: Summary) -> SectionValidation:
    """Summary 3–5 line check (error when out of range)."""
    result = SectionValidation()
    lines = summary.line_count()
    if lines < SUMMARY_MIN_LINES or lines > SUMMARY_MAX_LINES:
        result.errors.append(
            f"Summary must be {SUMMARY_MIN_LINES}–{SUMMARY_MAX_LINES} lines (got {lines})."
        )
    return result


def _validate_bullet(bullet: BulletPoint, company: str, result: SectionValidation) -> None:
    """Per-bullet writing checks (warnings) + ActionVerb membership (warning)."""
    if not isinstance(bullet.action_verb, ActionVerb):
        result.warnings.append(f"action_verb outside ActionVerb in {company}: {bullet.action_verb}")
    if len(bullet.description) > BULLET_MAX_CHARS:
        result.warnings.append(
            f"Bullet description exceeds {BULLET_MAX_CHARS} chars in {company}."
        )
    if bullet.impact and not any(char.isdigit() for char in bullet.impact):
        result.warnings.append(f"Bullet impact has no digit in {company}: {bullet.impact}")


def _validate_experience(
    experiences: list[Experience], facts: CVFacts
) -> SectionValidation:
    """Writing warnings + factual-integrity (company/date ∈ source — errors)."""
    result = SectionValidation()
    # A non-empty source CV whose generated experience came back empty silently dropped
    # the candidate's real work history — a blocking error, never a vacuous pass.
    if facts.experiences and not experiences:
        result.errors.append(
            "Experience section is empty but the source CV has work history."
        )
        return result
    source_companies = {_norm_key(e.company) for e in facts.experiences}
    source_spans = {
        (_norm_key(e.company), _norm_key(e.start_date), _norm_key(e.end_date))
        for e in facts.experiences
    }
    for exp in experiences:
        company_key = _norm_key(exp.company)
        if company_key not in source_companies:
            result.errors.append(f"Company not found in source CV: {exp.company}")
        elif (
            company_key,
            _norm_key(exp.start_date),
            _norm_key(exp.end_date),
        ) not in source_spans:
            result.errors.append(
                f"Dates for {exp.company} do not match source: "
                f"{exp.start_date}–{exp.end_date}"
            )
        if len(exp.company_description) < COMPANY_DESC_MIN_CHARS:
            result.warnings.append(f"company_description too short for {exp.company}.")
        for bullet in exp.bullets:
            _validate_bullet(bullet, exp.company, result)
    return result


def _validate_languages(languages: list[Language], facts: CVFacts) -> SectionValidation:
    """A PRESENT language level ∈ allowed set (error) + factual integrity (∈ source — error).

    Both checks compare on a normalized key, so a recased but truthful level
    ("native") or language name traces to source instead of being capped as fabrication.
    """
    result = SectionValidation()
    source_languages = {_norm_key(lang.language) for lang in facts.languages}
    for lang in languages:
        # A missing level is a truthful absence (the source listed a language with no proficiency),
        # not an error; only a PRESENT-but-unrecognized level is capped.
        if lang.level and _norm_key(lang.level) not in _LANGUAGE_LEVELS_NORM:
            result.errors.append(f"Invalid language level {lang.level!r} for {lang.language}.")
        if _norm_key(lang.language) not in source_languages:
            result.errors.append(f"Language not found in source CV: {lang.language}")
    return result


def _validate_education(education: list[Education], facts: CVFacts) -> SectionValidation:
    """Factual integrity for education (institution/degree ∈ source — error).

    Education is model-generated, so a fabricated institution/degree must block (the
    prior validate_section had no education branch, letting an invented Stanford/PhD ship).
    Compared on a normalized key so a truthfully recased entry is not falsely rejected.
    """
    result = SectionValidation()
    source_institutions = {_norm_key(e.institution) for e in facts.education}
    source_pairs = {
        (_norm_key(e.institution), _norm_key(e.degree)) for e in facts.education
    }
    for edu in education:
        institution_key = _norm_key(edu.institution)
        if institution_key not in source_institutions:
            result.errors.append(f"Institution not found in source CV: {edu.institution}")
        elif (institution_key, _norm_key(edu.degree)) not in source_pairs:
            result.errors.append(
                f"Degree for {edu.institution} does not match source: {edu.degree}"
            )
    return result


def _validate_named_passthrough(
    items: list, source_names: set[str], label: str, name_attr: str
) -> SectionValidation:
    """Factual integrity for projects/certificates: every name ∈ source (error)."""
    result = SectionValidation()
    for item in items:
        name = getattr(item, name_attr)
        if name not in source_names:
            result.errors.append(f"{label} not found in source CV: {name}")
    return result


def _find_provenance(provenance: list, keyword: str) -> SkillProvenance | None:
    """Return the ``SkillProvenance`` recorded for ``keyword`` (case-insensitive), or None."""
    target = keyword.strip().lower()
    for prov in provenance:
        if getattr(prov, "keyword", "").strip().lower() == target:
            return prov
    return None


def _validate_skills(skills: Skills, facts: CVFacts, jd: JdAnalysis) -> SectionValidation:
    """Tier-aware deterministic evidence gate for the Skills section.

    Applies the standard for each rendered keyword's tier (``jd.tier_of`` — tier governs, so
    a concrete keyword is never accepted via the anchor path, preventing laundering a named tech
    through a loosely-related anchor):

    - CONCRETE → BLOCKING ERROR (regen to cap, then flag; never a silent pass) when the term is
      not literally present in the source CV (the fabrication / interview-failure line).
    - COMPETENCY → requires a ``SkillProvenance`` anchor_ref that deterministically traces to the
      source CV (substring). A missing or untraceable anchor → DROP the keyword in place
      (deterministic, not an error). The validator confirms anchor EXISTENCE only, never whether
      it truly demonstrates the competency (the accepted Tier-2 relaxation, on the generator).
    - NOISE (non-blocking warning): a keyword evidenced but NOT named by the JD is role-irrelevant
      noise; dropped upstream at generation, so here it is only a warning.

    NON-VACUOUS: a Skills section with no surviving keyword (empty to begin with, or emptied by
    competency drops) is a blocking error — never a vacuous pass. The Spoken Languages category is
    exempt from the tier rules: it is sourced from ``facts.languages`` (always evidenced) and is
    the gold standard's only evidenced-but-not-JD group.
    """
    result = SectionValidation()
    evidence = _normalize_for_match(_facts_to_text(facts))
    jd_keywords = {kw.strip().lower() for kw in jd.keywords if kw.strip()}
    declared = set(_declared_skill_index(facts).keys())
    provenance = getattr(skills, "provenance", []) or []
    for category in skills.categories:
        # Header FORMAT is NOT validated here: a clean single-concept header is an invariant
        # enforced BY CONSTRUCTION at assembly (_normalize_skill_headers), never a regen trigger.
        # This gate keeps only the evidence rules below — non-vacuous, concrete-literal-or-error,
        # competency-anchor-or-drop.
        if category.category == SPOKEN_LANGUAGES_CATEGORY:
            continue
        kept: list[str] = []
        for keyword in category.keywords:
            # A declared skill is a fact (BASE): kept ungated — no tier bar, no noise warning.
            # It is literally present in the evidence text (facts.skills ⊂ _facts_to_text).
            if keyword.strip().lower() in declared:
                kept.append(keyword)
                continue
            if jd.tier_of(keyword) is KeywordTier.COMPETENCY:
                prov = _find_provenance(provenance, keyword)
                if prov is not None and _anchor_traces(prov.anchor_ref, evidence):
                    kept.append(keyword)
                # else: untraceable / missing anchor → drop the competency (deterministic)
                continue
            # CONCRETE (incl. untagged default): literal evidence required.
            kept.append(keyword)
            if not _term_present(keyword, evidence):
                result.errors.append(
                    f"Skills keyword not evidenced in the CV (fabrication): {keyword}"
                )
            elif keyword.strip().lower() not in jd_keywords:
                result.warnings.append(
                    f"Skills keyword not named by the JD (noise): {keyword}"
                )
        category.keywords = kept
    # NON-VACUOUS: count only REAL skill keywords. The facts-sourced Spoken Languages category is
    # always evidenced but is NOT professional-skill evidence — excluding it stops a CV whose every
    # real/declared skill was dropped (untraceable anchors / absent concretes) from passing as
    # "languages-only", which would skip regen and ship a Skills section with no skills.
    real_skill_count = sum(
        len(category.keywords)
        for category in skills.categories
        if category.category != SPOKEN_LANGUAGES_CATEGORY
    )
    if real_skill_count == 0:
        result.errors.append("Skills section is empty (no categories or no keywords).")
    return result


def validate_section(
    section_name: TargetSection | str,
    section_obj: object,
    facts: CVFacts,
    jd: JdAnalysis,
) -> SectionValidation:
    """Deterministically validate one generated section.

    Errors are blocking (trigger regeneration); warnings are non-blocking. Wraps the
    cv_template field-level rules (Summary.line_count, Link/email/language checks,
    ActionVerb membership) and adds ATS-coverage + factual-integrity checks.
    """
    name = section_name.value if isinstance(section_name, TargetSection) else section_name
    result = SectionValidation()

    # ATS heading check: every rendered section maps to a StandardHeading.
    if name not in _NO_HEADING and name not in _SECTION_HEADINGS:
        result.errors.append(f"Section heading is not ATS-standard: {name}")

    if name == TargetSection.CONTACT.value and isinstance(section_obj, PersonalInfo):
        _merge(result, _validate_contact(section_obj))
    elif name == TargetSection.SUMMARY.value and isinstance(section_obj, Summary):
        _merge(result, _validate_summary(section_obj))
    elif name == TargetSection.SKILLS.value and isinstance(section_obj, Skills):
        _merge(result, _validate_skills(section_obj, facts, jd))
    elif name == TargetSection.EXPERIENCE.value:
        _merge(result, _validate_experience(list(section_obj), facts))
    elif name == TargetSection.EDUCATION.value:
        _merge(result, _validate_education(list(section_obj), facts))
    elif name == TargetSection.PROJECTS.value:
        names = {p.name for p in facts.projects}
        _merge(result, _validate_named_passthrough(list(section_obj), names, "Project", "name"))
    elif name == "certificates":
        names = {c.title for c in facts.certificates}
        _merge(
            result,
            _validate_named_passthrough(list(section_obj), names, "Certificate", "title"),
        )
    elif name == "languages":
        _merge(result, _validate_languages(list(section_obj), facts))

    # ATS keyword floor: keyword-bearing sections must surface >= MIN_KEYWORDS.
    # Only enforce when the JD itself offers >= MIN_KEYWORDS distinct keywords — a JD with
    # fewer is an upstream keyword-gap (flagged at extract), not a skills-section failure, so
    # the floor would otherwise be unsatisfiable and cap a well-tailored section every run.
    if name in _KEYWORD_BEARING:
        available = len({kw.strip().lower() for kw in jd.keywords if kw.strip()})
        if (
            available >= MIN_KEYWORDS
            and _count_keyword_matches(_section_text(section_obj), jd.keywords) < MIN_KEYWORDS
        ):
            result.errors.append(
                f"Fewer than {MIN_KEYWORDS} JD keywords present in {name}."
            )
    return result


def _merge(into: SectionValidation, other: SectionValidation) -> None:
    """Fold ``other`` errors/warnings into ``into`` (in place)."""
    into.errors.extend(other.errors)
    into.warnings.extend(other.warnings)


# ──────────────────────────────────────────────────────────────────────────────
#  Section-local regeneration loop
# ──────────────────────────────────────────────────────────────────────────────


async def _resolve_section(
    section_name: TargetSection | str,
    generate_fn: Callable[[], Awaitable[object]],
    validate_fn: Callable[[object], SectionValidation],
    *,
    cap: int = REGEN_CAP,
) -> tuple[object, SectionFlag | None]:
    """Generate→validate a section, regenerating only it up to ``cap`` times.

    Returns the resolved section object and an optional ``capped_section`` flag. On
    cap-hit the last attempt is emitted with the flag (never an infinite loop):
    ``attempts < cap`` retries, ``attempts == cap`` → emit + flag.

    Async I/O — ``generate_fn`` is a coroutine factory (the section generator awaits the
    provider), so each attempt is ``await``ed. ``validate_fn`` is fast pure-CPU
    deterministic validation and is called DIRECTLY (never threadpooled). The per-section
    regen loop stays sequential by design — a regen depends on the prior attempt's verdict.
    """
    last_obj: object = None
    for attempt in range(cap + 1):
        last_obj = await generate_fn()
        validation = validate_fn(last_obj)
        if not validation.errors:
            return last_obj, None
        if attempt == cap:
            # Carry the real section identity (incl. string-keyed languages/certificates).
            # The raw validation errors are an INTERNAL diagnostic (rule text the user
            # cannot act on, e.g. "violates format rules"). Log them server-side for
            # debuggability; the user-facing flag carries ONLY the section identity, so no
            # internal rule text can reach the panel for any section.
            logger.warning(
                "Section '%s' still failing after %d retries: %s",
                section_name,
                cap,
                "; ".join(validation.errors),
            )
            flag = SectionFlag(
                section=section_name,
                kind=FlagKind.CAPPED_SECTION,
                message=(
                    f"The '{section_name}' section could not be fully optimized — try re-running."
                ),
            )
            return last_obj, flag
    return last_obj, None


# ──────────────────────────────────────────────────────────────────────────────
#  Assembly + one-page global gate
# ──────────────────────────────────────────────────────────────────────────────

_JUNIOR_LEVELS = frozenset({CandidateLevel.NEW_GRAD, CandidateLevel.ENTRY})


def _section_order(level: CandidateLevel, cv: CVTemplate) -> list[str]:
    """Level-driven render order: juniors lead with education.

    Appends the populated string-keyed sections (certificates/languages, which are not
    TargetSection members) so a downstream renderer iterating ``section_order`` never drops
    them.
    """
    if level in _JUNIOR_LEVELS:
        order = [
            TargetSection.CONTACT,
            TargetSection.SUMMARY,
            TargetSection.SKILLS,
            TargetSection.EDUCATION,
            TargetSection.EXPERIENCE,
            TargetSection.PROJECTS,
        ]
    else:
        order = [
            TargetSection.CONTACT,
            TargetSection.SUMMARY,
            TargetSection.SKILLS,
            TargetSection.EXPERIENCE,
            TargetSection.EDUCATION,
            TargetSection.PROJECTS,
        ]
    rendered = [section.value for section in order]
    if cv.certificates:
        rendered.append("certificates")
    if cv.languages:
        rendered.append("languages")
    return rendered


def _dedup_preserve(values: list[str]) -> list[str]:
    """Drop case-insensitive duplicates, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _dedup_subsumed_skills(skills: Skills) -> None:
    """Drop a Skills phrase a longer RETAINED phrase already contains whole-word.

    Intra-section noise repair: a short phrase ("test automation") fully contained — on word
    boundaries — in a longer phrase kept in the SAME section ("Mobile/Web/API test automation")
    adds ZERO ATS coverage (the scorer matches the keyword via the longer phrase) and only eats
    the one-page budget. Runs across ALL categories (whole-section), reusing the scorer's
    word-boundary matcher (``_term_present``), so "Java" is never subsumed by "JavaScript".

    Keeps the longer subsuming phrase, drops the subsumed one. COVERAGE-NEUTRAL by construction:
    the retained phrase carries the dropped phrase's exact tokens on the same boundaries, so every
    JD keyword that matched the dropped phrase still matches the retained one. Strict subsumption
    only (case-insensitive exact duplicates are handled upstream by ``_dedup_preserve``); the
    relation is a strict order by length, so the maximal phrase in any chain is never dropped and
    the section never empties. Order-stable.

    A DECLARED skill (``SkillProvenance.declared`` — one of the candidate's own ``facts.skills``)
    is a FACT kept by default, so it is NEVER dropped here even when a longer phrase in
    another category subsumes it; only an ADDED keyword can be deduped. Still coverage-neutral.

    The Spoken Languages category is EXEMPT (matching ``_validate_skills`` /
    ``_normalize_skill_headers``): its keywords are ``facts.languages``-sourced, carry no
    provenance, and must never be dropped just for being a whole-word substring of a longer skill
    phrase in another category — that would silently delete a real spoken-language fact (truth).
    """
    provenance = getattr(skills, "provenance", []) or []
    subsumable = [c for c in skills.categories if c.category != SPOKEN_LANGUAGES_CATEGORY]
    rendered = [keyword for category in subsumable for keyword in category.keywords]
    normalized = [_normalize_for_match(keyword) for keyword in rendered]
    declared = [
        bool(getattr(_find_provenance(provenance, keyword), "declared", False))
        for keyword in rendered
    ]
    drop = [False] * len(rendered)
    for i, x_norm in enumerate(normalized):
        if not x_norm or declared[i]:
            continue  # a declared skill is a fact — never subsumed away
        for j, y_norm in enumerate(normalized):
            # Strict whole-word subsumption: X is a whole-word substring of a *different*,
            # necessarily longer Y. casefold-equal is exact-dup (handled upstream), not
            # subsumption — skip it so a case variant is never mistaken for a longer phrase.
            if i != j and x_norm.casefold() != y_norm.casefold() and _term_present(
                rendered[i], y_norm
            ):
                drop[i] = True
                break
    # Degenerate guard (defensive — unreachable for strict subsumption, since the longest phrase
    # has no container): never empty the section; keep the single longest, first-seen on tie.
    if rendered and all(drop):
        keep = max(range(len(rendered)), key=lambda k: len(normalized[k]))
        drop[keep] = False
    index = 0
    for category in subsumable:
        kept: list[str] = []
        for keyword in category.keywords:
            if not drop[index]:
                kept.append(keyword)
            index += 1
        category.keywords = kept


def _dedup_cv_content(cv: CVTemplate) -> None:
    """Dedup: skill lists + repeated bullet descriptions WITHIN each role.

    Bullet dedup is scoped per role, NOT one set spanning all experiences — two distinct
    roles legitimately share a bullet (e.g. "Led the migration"), and a global set would drop
    the second role's copy and could empty that role.
    """
    for category in cv.skills.categories:
        category.keywords = _dedup_preserve(category.keywords)
    # Whole-section pass (after exact-dup dedup): drop a phrase already implied word-for-word by
    # a longer retained phrase — coverage-neutral noise removal that frees one-page budget.
    _dedup_subsumed_skills(cv.skills)
    for exp in cv.experiences:
        seen_bullets: set[str] = set()
        kept: list[BulletPoint] = []
        for bullet in exp.bullets:
            key = bullet.description.strip().lower()
            if key and key not in seen_bullets:
                seen_bullets.add(key)
                kept.append(bullet)
        exp.bullets = kept


def estimate_page_count(cv: CVTemplate) -> int:
    """Estimate rendered page count from a deterministic line budget."""
    lines = 2  # contact block
    lines += max(cv.summary.line_count(), 1)
    # Count skills BY VOLUME (ceil per group), not a flat 1-line-per-group, so a
    # skill-heavy CV registers real overflow instead of slipping past the one-page gate.
    # Skip the facts-sourced Spoken Languages category — it is already counted once via
    # ``len(cv.languages)`` below, so counting its skills-category volume too would double-count
    # the languages and inflate the estimate (spurious compression / one-page-pressure).
    for category in cv.skills.categories:
        if category.category == SPOKEN_LANGUAGES_CATEGORY:
            continue
        if category.keywords:
            lines += -(-len(category.keywords) // SKILLS_PER_LINE)  # ceil division
    for exp in cv.experiences:
        lines += 2 + len(exp.bullets)
    lines += len(cv.education)
    lines += len(cv.projects)
    lines += len(cv.certificates)
    lines += len(cv.languages)
    return max(1, -(-lines // LINES_PER_PAGE))  # ceil division


def _bullet_carries_keyword(bullet: BulletPoint, protected: list[str]) -> bool:
    """True when a bullet surfaces any protected JD keyword (never trim these)."""
    if not protected:
        return False
    text = _normalize_for_match(
        " ".join(
            [bullet.description, *bullet.skills, bullet.impact or "", bullet.benefit or ""]
        )
    )
    return any(_term_present(keyword, text) for keyword in protected)


def _trim_one_bullet(cv: CVTemplate, protected: list[str]) -> bool:
    """Drop one trimmable (non-keyword) bullet from the richest role; True if one was dropped.

    Keyword-bearing bullets are skipped so the coverage surface survives compression. A
    role is never emptied (only roles with >1 bullet are candidates).
    """
    candidates = sorted(
        (exp for exp in cv.experiences if len(exp.bullets) > 1),
        key=lambda exp: len(exp.bullets),
        reverse=True,
    )
    for exp in candidates:
        for index in range(len(exp.bullets) - 1, -1, -1):
            if not _bullet_carries_keyword(exp.bullets[index], protected):
                del exp.bullets[index]
                return True
    return False


def _compress_to_one_page(cv: CVTemplate, keywords: list[str]) -> None:
    """Best-effort trim of non-keyword experience bullets toward the one-page budget.

    Trims the richest role's last non-keyword bullet repeatedly (keeping >=1 per role). This
    is best-effort, NOT a guarantee: when no removable bullet remains the loop exits even if
    the CV still overflows — content it can't touch (skills, education, projects, certificates,
    languages, and keyword-bearing bullets) is left intact. The caller gates on the real page
    estimate and flags any residual overflow.

    Compression never removes the Skills keywords (skills.categories[].keywords) NOR a bullet
    carrying a JD keyword, so the coverage surface is preserved while prose overflow is shed;
    the score then reflects the final CV without a spurious did-not-converge.
    """
    protected = [keyword for keyword in keywords if keyword.strip()]
    while estimate_page_count(cv) > MAX_PAGES:
        if not _trim_one_bullet(cv, protected):
            break


def assemble_and_gate(
    sections: dict, candidate_level: CandidateLevel, *, keywords: list[str] | None = None
) -> tuple[CVTemplate, list[SectionFlag]]:
    """Assemble passing sections under the one-page global gate.

    Sets level-driven ``section_order``, deduplicates cross-section content, compresses
    overflow (keyword-aware — never trimming a bullet that carries a JD ``keywords`` term),
    and emits a ``one_page_pressure`` flag for many-role pressure AND for any residual
    overflow that compression could not shed (the real one-page gate).
    """
    cv = CVTemplate(
        personal_info=sections["personal_info"],
        summary=sections["summary"],
        skills=sections["skills"],
        experiences=list(sections.get("experiences", [])),
        education=list(sections.get("education", [])),
        projects=list(sections.get("projects", [])),
        certificates=list(sections.get("certificates", [])),
        languages=list(sections.get("languages", [])),
        section_order=[],
    )
    cv.section_order = _section_order(candidate_level, cv)
    flags: list[SectionFlag] = []
    _dedup_cv_content(cv)
    # Supplementary signal: many roles foreshadow one-page pressure (not the gate itself).
    if len(cv.experiences) > MAX_ROLES_NO_WARNING:
        flags.append(
            SectionFlag(
                section=TargetSection.EXPERIENCE,
                kind=FlagKind.ONE_PAGE_PRESSURE,
                message=(
                    f"{len(cv.experiences)} roles exceed {MAX_ROLES_NO_WARNING}; "
                    "content may be compressed toward the one-page budget."
                ),
            )
        )
    _compress_to_one_page(cv, keywords or [])
    # The real one-page gate: flag genuine residual overflow on the actual page
    # estimate so a >1-page CV never ships silently as "converged".
    if estimate_page_count(cv) > MAX_PAGES:
        flags.append(
            SectionFlag(
                # Residual overflow can originate in any section (education, projects,
                # skills, …), not necessarily experience — attribute it to the global gate.
                section=GLOBAL_SECTION,
                kind=FlagKind.ONE_PAGE_PRESSURE,
                message=(
                    "CV still exceeds one page after compressing experience bullets; remove a "
                    "non-experience entry (education, project, certificate, or skills) to fit."
                ),
            )
        )
    return cv, flags


# ──────────────────────────────────────────────────────────────────────────────
#  ATS keyword-coverage scoring
# ──────────────────────────────────────────────────────────────────────────────


def _section_text(section_obj: object) -> str:
    """Serialize a single section object to plain text for keyword matching."""
    if isinstance(section_obj, Summary):
        # Score only the prose `text` — the sole summary field rendered/copied (app._summary_text).
        # `relevant_skills` is still produced but no longer rendered, so scoring it would overstate
        # the copied resume (score basis must equal render basis). 🟡 keywords are woven into
        # `text` by the summary prompt, so the prose IS the coverage source.
        return section_obj.text
    if isinstance(section_obj, Skills):
        return " ".join(
            keyword for category in section_obj.categories for keyword in category.keywords
        )
    if isinstance(section_obj, list):
        return " ".join(_experience_text(item) for item in section_obj)
    return str(section_obj)


def _experience_text(exp: Experience) -> str:
    """Flatten one experience entry to the PROSE the UI renders (coverage basis).

    Render == score: ``bullet.skills`` are structured tags the UI does not render as
    prose, so they are NOT part of the coverage basis — counting them would overstate the
    resume the user actually copies. role/company/company_description and the bullet
    prose (description/impact/benefit) ARE rendered, so they stay.
    """
    parts = [exp.role, exp.company, exp.company_description or ""]
    for bullet in exp.bullets:
        parts.append(bullet.description)
        if bullet.impact:
            parts.append(bullet.impact)
        if bullet.benefit:
            parts.append(bullet.benefit)
    return " ".join(parts)


def _cv_to_text(cv: CVTemplate) -> str:
    """Serialize the whole tailored CV to plain text for coverage matching."""
    parts = [
        _section_text(cv.summary),
        _section_text(cv.skills),
        _section_text(cv.experiences),
    ]
    for edu in cv.education:
        parts.append(f"{edu.institution} {edu.degree}")
    for proj in cv.projects:
        parts.append(f"{proj.name} {proj.description or ''} {' '.join(proj.skills)}")
    for cert in cv.certificates:
        parts.append(f"{cert.title} {cert.issuer or ''}")
    for lang in cv.languages:
        parts.append(f"{lang.language} {lang.level or ''}")
    return " ".join(parts)


def _facts_to_text(facts: CVFacts) -> str:
    """Serialize the original parsed CV facts to plain text (the before basis).

    Covers the SAME field set as `_cv_to_text`/`_experience_text` — including the pass-through
    education/languages AND experience company_description — so a keyword identical before and
    after is not counted as before→after lift. The candidate's DECLARED
    skills (``facts.skills``) ARE included now: the source CV literally listed them, so counting
    them in the before-basis keeps a tailored declared skill honest lift-wise (a skill the CV
    already had is not scored as before→after lift). The generated summary has no source-facts
    counterpart, so it is intentionally absent (its coverage IS genuine lift).
    """
    parts: list[str] = []
    for exp in facts.experiences:
        parts.extend([exp.role, exp.company, exp.company_description or "", *exp.bullets])
    for edu in facts.education:
        parts.extend([edu.institution, edu.degree])
    for proj in facts.projects:
        parts.extend([proj.name, proj.description or "", *proj.skills])
    for cert in facts.certificates:
        parts.extend([cert.title, cert.issuer or ""])
    for lang in facts.languages:
        parts.extend([lang.language, lang.level or ""])
    for group in facts.skills:
        parts.extend(group.items)
    return " ".join(parts)


def _coverage(matched: list[str], must: list[str]) -> float:
    """Coverage percentage `|matched| / |must| × 100` (100.0 when nothing required)."""
    if not must:
        return 100.0
    return round(len(matched) / len(must) * 100, 2)


def _normalize_for_match(text: str) -> str:
    """Homoglyph-normalize for keyword matching WITHOUT collapsing whitespace.

    Uses `normalize_input` (char-layer demangle + invisible strip), NOT `clean()`.
    `clean()`'s watermark heuristic collapses inter-word spaces, which destroys
    the word boundaries coverage matching depends on — that destruction is exactly what let
    the old raw-substring match appear to work while false-matching ("Go" ⊂ "Google").
    """
    return TextPreprocessing.normalize_input(text)


def _term_present(term: str, text: str) -> bool:
    """Whole-word (case-insensitive) presence of ``term`` in ``text``.

    A word-char boundary is asserted ONLY on the side where ``term`` itself starts/ends in an
    alphanumeric char, so a short token never falsely matches inside a larger word ("Go" not in
    "Google", "Java" not in "JavaScript") YET a symbol/dotted keyword still matches inside a real
    token whose boundary is non-word (".NET" in "ASP.NET Core", "C#" matches). The old code
    asserted a boundary unconditionally and so missed those. Empty terms
    never match. Both ``term`` and ``text`` are homoglyph-normalized by the caller.
    """
    normalized = _normalize_for_match(term)
    if not normalized:
        return False
    pattern = re.escape(normalized)
    left = r"(?<!\w)" if normalized[0].isalnum() else ""
    right = r"(?!\w)" if normalized[-1].isalnum() else ""
    return re.search(rf"{left}{pattern}{right}", text, re.IGNORECASE) is not None


def score_ats(cv: CVTemplate, jd: JdAnalysis, *, original_cv_text: str) -> AtsScore:
    """Compute before→after JD keyword coverage on normalized text.

    The coverage basis is `jd.keywords` — the short JD KEYWORDS, NOT the multi-word
    `requirements_must` phrases: a tailored CV satisfies a requirement by reframing, not
    by quoting the requirement sentence verbatim, so phrase matching would peg coverage near
    zero and fire a spurious did-not-converge. Matching runs on homoglyph-normalized text so a
    homoglyph cannot silently break a match, and on word boundaries so substrings
    don't false-match. `missing[]` is the full honest-gap list (never truncated).
    The before/after bases serialize the SAME field set (`_cv_to_text` / `_facts_to_text`,
    incl. company_description) so a pass-through keyword isn't counted as lift.
    """
    keywords = _dedup_preserve(jd.keywords)
    after_text = _normalize_for_match(_cv_to_text(cv))
    before_text = _normalize_for_match(original_cv_text)
    matched = [term for term in keywords if _term_present(term, after_text)]
    missing = [term for term in keywords if not _term_present(term, after_text)]
    before_matched = [term for term in keywords if _term_present(term, before_text)]
    after_pct = _coverage(matched, keywords)
    before_pct = _coverage(before_matched, keywords)
    return AtsScore(
        before_pct=before_pct,
        after_pct=after_pct,
        matched=matched,
        missing=missing,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


async def generate_tailored_cv(
    extract: ExtractResult, *, llm: LLMModel | None = None
) -> TailoredResult | dict[str, str]:
    """Orchestrate the section-wise tailoring pipeline.

    plan → per-section generate+validate+regen(cap) → carry personal_info → assemble +
    one-page gate → score → flags. Validation/coverage gaps surface as `flags[]`, never
    an error; only provider/transport failures return the `{ error, stage }` envelope.
    """
    llm = llm or LLMModel()
    # One semaphore per request bounds the section fan-out to MAX_CONCURRENT_LLM_CALLS
    # in-flight provider calls (rate-limit safety). Created on the running loop (no cross-loop
    # reuse) and never shared across requests.
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    facts, jd = extract.facts, extract.jd
    # Carry the upstream keyword-gap flag(s) from the extract stage into the output
    # (honest gaps). They surface as UNMET_COVERAGE on the global gate sentinel.
    flags: list[SectionFlag] = [
        SectionFlag(section=GLOBAL_SECTION, kind=FlagKind.UNMET_COVERAGE, message=message)
        for message in extract.flags
    ]
    try:
        sections, section_flags = await _generate_all_sections(facts, jd, llm, semaphore)
    except (ProviderResponseError, OpenAIError) as exc:
        # Provider-SHAPE failures (malformed reply, or a model-omitted/null required
        # field via _require_fields). Openai transport errors (timeout/network/auth) raised
        # by call_tool — extract.build_extract catches these for the same call, so the
        # generation path must too, or a provider timeout escapes as a 500. Both map to the
        # redacted {error, stage:generate} envelope. A bare KeyError is still NOT caught, so a
        # bug in OUR mapping surfaces loudly instead of being mislabeled a provider failure.
        return to_pipeline_error(PipelineStage.GENERATE, str(exc))
    flags.extend(section_flags)

    cv, gate_flags = assemble_and_gate(sections, jd.candidate_level, keywords=jd.keywords)
    flags.extend(gate_flags)

    ats = score_ats(cv, jd, original_cv_text=_facts_to_text(facts))
    flags.extend(_coverage_flags(ats))
    return TailoredResult(cv=cv, ats_score=ats, flags=flags)


def _coverage_flags(ats: AtsScore) -> list[SectionFlag]:
    """Build the de-duplicated omission flags for a below-target coverage outcome.

    Exactly two entries, never the old run-on: ONE coverage line carrying the before→after
    numbers (no embedded keyword list), and — when there are honestly-omitted keywords — ONE
    missing line carrying the keyword list only. Both attribute to the global-gate sentinel
    (coverage is a CV-wide outcome, not a skills-section failure). At or above target there
    is no omission panel. The roles / residual one-page-pressure warnings are emitted elsewhere.
    """
    if ats.after_pct >= COVERAGE_TARGET_PCT:
        return []
    flags = [
        SectionFlag(
            section=GLOBAL_SECTION,
            kind=FlagKind.DID_NOT_CONVERGE,
            message=(
                f"Coverage {ats.after_pct}% (before {ats.before_pct}% → after "
                f"{ats.after_pct}%); below {COVERAGE_TARGET_PCT}% target."
            ),
        )
    ]
    if ats.missing:
        flags.append(
            SectionFlag(
                section=GLOBAL_SECTION,
                kind=FlagKind.UNMET_COVERAGE,
                message=(
                    "Missing (no CV evidence — omitted, not fabricated): "
                    + ", ".join(ats.missing)
                ),
            )
        )
    return flags


async def _generate_all_sections(
    facts: CVFacts, jd: JdAnalysis, llm: LLMModel, semaphore: asyncio.Semaphore
) -> tuple[dict, list[SectionFlag]]:
    """Generate every section through the regeneration loop; collect cap-hit flags.

    The 7 sections fan out concurrently under ``bounded_gather`` (capped by ``semaphore``);
    each section's INTERNAL regen loop stays sequential. Determinism is preserved:
    the section generators take only ``(facts, jd, llm)`` with no cross-section runtime dependency,
    ``bounded_gather`` returns results in the SAME order as ``specs`` (never completion order), and
    every result/flag is mapped back to its FIXED dict key / spec-order slot. So the
    generated content for a given (facts, jd) is identical whether run sequentially or concurrently
    — assembly then re-orders by ``section_order`` downstream. ``personal_info`` is a pure-CPU carry
    (no provider call), so it is built directly; ``_normalize_skill_headers`` runs on the skills
    RESULT after the gather (moved here from between sections — it only mutates skills, so order is
    behaviour-neutral).
    """
    sections: dict = {"personal_info": _carry_personal_info(facts)}

    # FIXED (key, section-name, generator) order — the determinism anchor for the fan-out.
    specs: list[tuple[str, TargetSection | str, Callable[[], Awaitable[object]]]] = [
        ("summary", TargetSection.SUMMARY, lambda: _generate_summary(facts, jd, llm)),
        ("skills", TargetSection.SKILLS, lambda: _generate_skills(facts, jd, llm)),
        ("experiences", TargetSection.EXPERIENCE, lambda: _generate_experience(facts, jd, llm)),
        ("education", TargetSection.EDUCATION, lambda: _generate_education(facts, jd, llm)),
        ("projects", TargetSection.PROJECTS, lambda: _generate_project(facts, jd, llm)),
        ("certificates", "certificates", lambda: _generate_certificate(facts, jd, llm)),
        ("languages", "languages", lambda: _generate_language(facts, jd, llm)),
    ]

    async def _resolve_one(
        name: TargetSection | str, generate_fn: Callable[[], Awaitable[object]]
    ) -> tuple[object, SectionFlag | None]:
        return await _resolve_section(
            name, generate_fn, lambda o: validate_section(name, o, facts, jd)
        )

    resolved = await bounded_gather(
        [_resolve_one(name, generate_fn) for _key, name, generate_fn in specs], semaphore
    )

    flags: list[SectionFlag] = []
    for (key, _name, _generate_fn), (obj, flag) in zip(specs, resolved):
        sections[key] = obj
        if flag is not None:
            flags.append(flag)
    # Header invariant: normalize every emergent category header to a clean single-concept
    # shape at assembly — malformed headers are unrepresentable, never a reject/regen trigger.
    _normalize_skill_headers(sections["skills"])
    return sections, flags

