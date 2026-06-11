"""Cover-letter generator.

A single function-calling step consumes the shared ``ExtractResult`` and writes a truthful,
point-by-point JD requirement → CV evidence cover letter inside a full letter envelope
(salutation → opening → body → closing → sign-off), enforces an anti-pattern ban-list at the
prompt level, and runs a deterministic, SHAPE-only gate (``_is_well_structured``) plus the word
cap through one capped regen path (a residual over-length letter is flagged, never truncated; a
residually unstructured letter is shipped without a user-facing flag), then runs the shared
cleanup pass and returns ``CoverLetterResult{cover_letter, flags}``.

Reuses the CV pipeline's ``call_tool`` → ``_require_fields`` → ``_clean_text`` pattern and the
shared ``SectionFlag`` / ``PipelineStage`` / ``to_pipeline_error`` infrastructure from
``cv_generator.py``.
"""

from __future__ import annotations

import re

from openai import OpenAIError

from cv_generator import (
    COVER_LETTER_MAX_WORDS,
    COVER_LETTER_REGEN_CAP,
    COVER_LETTER_SECTION,
    CoverLetterResult,
    FlagKind,
    PipelineStage,
    SectionFlag,
    _clean_text,
    _require_fields,
    to_pipeline_error,
)
from helprers.llm_model import LLMModel, ProviderResponseError
from helprers.prompts import Prompts
from schemas import CVFacts, ExtractResult, JdAnalysis
from tool_schemas import generate_cover_letter_tool_schema

__all__ = ["generate_cover_letter"]

#: Greeting prefixes that open an English-letter salutation line (shape, not content).
_SALUTATION_PREFIXES = ("dear", "hello", "hi", "to whom", "greetings")

#: Closing words that open a sign-off line ("Sincerely,", "Best regards", …). Matched on the
#: first word of the last block; the candidate's name follows on the next line (shape only).
_SIGN_OFF_WORDS = (
    "sincerely",
    "regards",
    "best",
    "respectfully",
    "faithfully",
    "cordially",
    "yours",
    "thank you",
    "thanks",
    "warm",
    "kind",
)


def _word_count(text: str) -> int:
    """Whitespace-split token count — the deterministic word-cap metric."""
    return len(text.split())


def _paragraph_blocks(text: str) -> list[str]:
    """Split into blank-line-separated, non-empty paragraph blocks (the structural unit)."""
    return [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]


def _is_salutation(block: str) -> bool:
    """A salutation is a single short greeting line (English-letter convention; shape only)."""
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    line = lines[0].strip().lower()
    return line.startswith(_SALUTATION_PREFIXES) or line.endswith(",")


def _is_sign_off(block: str) -> bool:
    """A sign-off block is a closing word line followed by a name line (≥2 lines; shape only)."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    first = lines[0].rstrip(",").lower()
    return any(first.startswith(word) for word in _SIGN_OFF_WORDS)


def _is_well_structured(text: str) -> bool:
    """Deterministic SHAPE gate: ≥3 blocks, a salutation first, a sign-off last.

    Catches the "single flat blob" failure (no blank-line separation ⇒ one block ⇒ fails) and a
    letter missing its salutation or sign-off. This is a string/shape assertion only — it never
    scores content quality and never invokes an LLM judge.
    """
    blocks = _paragraph_blocks(text)
    if len(blocks) < 3:
        return False
    return _is_salutation(blocks[0]) and _is_sign_off(blocks[-1])


def _quality(clean_text: str) -> tuple[bool, bool, int]:
    """Rank a cleaned draft for the capped regen loop: prefer well-structured, then within the
    word cap, then shorter. A higher tuple wins; ties keep the incumbent (strict ``>``)."""
    return (
        _is_well_structured(clean_text),
        _word_count(clean_text) <= COVER_LETTER_MAX_WORDS,
        -_word_count(clean_text),
    )


def _cover_letter_user_prompt(
    facts: CVFacts, jd: JdAnalysis, *, previous_draft: str | None = None
) -> str:
    """Build the cover-letter user prompt from the shared extract's facts + jd.

    The JD analysis is consumed as-is from ``extract.jd`` — ``analyze_jd`` is never invoked
    here. Mirrors ``cv_generator._section_user_prompt``'s JSON-dump style. When
    ``previous_draft`` is given (the compress retry), the actual over-length draft is
    EMBEDDED so the model compresses THAT text rather than blindly re-generating:
    ``call_tool`` sends only [system, user] with no history, so a "rewrite the previous
    draft" directive that does not include the draft has nothing to act on.
    """
    prompt = Prompts.GENERATE_COVER_LETTER_USER.format(
        facts=facts.model_dump_json(),
        jd=jd.model_dump_json(),
    )
    if previous_draft is not None:
        prompt += (
            f"\n\nYour previous draft was {_word_count(previous_draft)} words, over the "
            f"{COVER_LETTER_MAX_WORDS}-word limit. Rewrite the draft below to at most "
            f"{COVER_LETTER_MAX_WORDS} words, preserving every requirement→evidence point; "
            f"cut filler and repetition, never the underlying evidence.\n\n"
            f"Previous draft:\n{previous_draft}"
        )
    return prompt


async def _draft(
    llm: LLMModel, facts: CVFacts, jd: JdAnalysis, *, previous_draft: str | None = None
) -> dict:
    """Call the cover-letter tool once and return its guarded arguments.

    ``_require_fields(..., non_empty=True)`` enforces that ``text`` is present
    AND a non-blank string — a model that omits it, nulls it, or returns an empty/whitespace (or
    non-string) ``text`` is a provider-shape gap mapped to the generate-stage envelope upstream.
    This is the SAME guard the CV section text now uses, kept at the shared primitive rather than
    re-implemented inline here.
    """
    result = await llm.call_tool(
        Prompts.GENERATE_COVER_LETTER_SYSTEM,
        _cover_letter_user_prompt(facts, jd, previous_draft=previous_draft),
        generate_cover_letter_tool_schema(),
    )
    _require_fields(result, ("text",), "Cover letter", non_empty=True)
    return result


async def generate_cover_letter(
    extract: ExtractResult, *, llm: LLMModel | None = None
) -> CoverLetterResult | dict[str, str]:
    """Generate a truthful, well-structured cover letter from the shared extract.

    Reuses ``extract.jd`` as-is — the JD is never re-parsed. All decisions run on ONE
    basis: the CLEANED (delivered) text, since cleanup can both drop AND add whitespace tokens.
    A draft that is over-length OR structurally malformed (``_is_well_structured`` — a
    SHAPE gate, never an LLM judge) triggers a capped regen (≤ ``COVER_LETTER_REGEN_CAP``): an
    over-length draft is GIVEN to the model to shorten, a purely structural failure re-asks for a
    fresh well-enveloped letter. The best draft seen is kept (``_quality`` — structured, then
    within-cap, then shorter); after the cap a residual over-length letter is flagged (never
    truncated) but a residually unstructured letter ships WITHOUT a user-facing flag.
    Provider/transport failures — including client construction with no API key and a
    model-omitted/blank required ``text`` — return the redacted ``{error, stage="generate"}``
    envelope.
    """
    facts, jd = extract.facts, extract.jd
    try:
        # Construct the client INSIDE the try so a missing-key OpenAIError at construction
        # maps to the redacted envelope (mirrors build_extract), then draft the first letter —
        # a failure here is fatal (nothing deliverable to fall back to).
        llm = llm or LLMModel()
        first = await _draft(llm, facts, jd)
    except (ProviderResponseError, OpenAIError) as exc:
        # openai transport errors (timeout / network / auth) and provider-shape failures
        # both map to the redacted generate-stage envelope (mirrors generate_tailored_cv).
        return to_pipeline_error(PipelineStage.GENERATE, str(exc))

    # Single basis — clean each draft ONCE and reuse its cleaned form for the gate, the
    # quality comparison, the empty-check, and the displayed residual count.
    best, best_clean = first, _clean_text(first["text"])
    compressed = False
    for _ in range(COVER_LETTER_REGEN_CAP):
        # An empty cleaned draft is a provider-shape failure (handled below), not something a
        # structural re-roll can fix — stop regenerating and fall through to the empty-check.
        if not best_clean.strip():
            break
        over_length = _word_count(best_clean) > COVER_LETTER_MAX_WORDS
        if not over_length and _is_well_structured(best_clean):
            break  # within the cap AND well-structured — nothing left to improve
        try:
            # An over-length draft is handed back to be compressed; a purely
            # structural failure re-asks fresh (no draft to shorten). Best-effort — a failed
            # regen (timeout / blip) keeps the best draft so far rather than hard-failing.
            regen = await _draft(
                llm, facts, jd, previous_draft=best["text"] if over_length else None
            )
        except (ProviderResponseError, OpenAIError):
            break
        if over_length:
            compressed = True
        regen_clean = _clean_text(regen["text"])
        # Keep the strictly-better draft (structured > within-cap > shorter); ties keep the
        # incumbent, so a compress that backfired into a longer draft is never delivered.
        if _quality(regen_clean) > _quality(best_clean):
            best, best_clean = regen, regen_clean

    cleaned = best_clean
    # An empty letter AFTER cleanup is a provider-shape failure — the letter is the
    # whole payload, so shipping "" as success is worse than the redacted generate envelope
    # (a non-empty raw draft can clean to "", e.g. pure chatbot-residue or all-emoji).
    if not cleaned.strip():
        return to_pipeline_error(
            PipelineStage.GENERATE, "Cover letter is empty after the cleanup pass."
        )

    flags: list[SectionFlag] = []
    residual = _word_count(cleaned)
    if residual > COVER_LETTER_MAX_WORDS:
        # Non-fatal (no truncation): cutting an evidence point mid-sentence is worse than a
        # slightly-long, fully-truthful letter. The message reflects what
        # actually ran — claim a compress pass only when one produced a draft.
        pass_note = (
            "after a compress pass" if compressed else "the compress retry did not complete"
        )
        flags.append(
            SectionFlag(
                section=COVER_LETTER_SECTION,
                kind=FlagKind.COVER_LETTER_OVER_LENGTH,
                message=(
                    f"Cover letter is {residual} words, over the {COVER_LETTER_MAX_WORDS}-word "
                    f"limit ({pass_note}); trim a less-relevant point to fit."
                ),
            )
        )
    # A JD with no structured must-have requirements cannot be matched point by point —
    # surface it honestly rather than silently shipping a generic letter (honest gaps).
    if not jd.requirements_must:
        flags.append(
            SectionFlag(
                section=COVER_LETTER_SECTION,
                kind=FlagKind.COVER_LETTER_NO_REQUIREMENTS,
                message=(
                    "The job description provided no structured must-have requirements to match "
                    "point by point; the letter is written from the candidate's facts alone."
                ),
            )
        )
    # The cover letter no longer self-reports per-requirement omissions. A letter that
    # cannot back a requirement still OMITS it (generation rule — never fabricated), but that is
    # not surfaced here; the ATS missing-keywords flag is the single "what's missing" signal.
    return CoverLetterResult(cover_letter=cleaned, flags=flags)
