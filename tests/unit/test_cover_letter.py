"""Unit tests for the cover-letter generator.

The OpenRouter client is the conftest ``mock_llm`` seam — ``call_tool`` returns the
tool-call dict the model would have produced, so ``generate_cover_letter`` is exercised
without any network access. This module hosts the provider-mock fixtures and a sample
``ExtractResult`` builder plus the cases that drive each TDD cycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

from cover_letter import (
    _is_salutation,
    _is_sign_off,
    _is_well_structured,
    generate_cover_letter,
)
from cv_generator import (
    COVER_LETTER_MAX_WORDS,
    COVER_LETTER_REGEN_CAP,
    COVER_LETTER_SECTION,
    CoverLetterResult,
    FlagKind,
)
from helprers.llm_model import ProviderResponseError
from schemas import (
    CandidateLevel,
    CVFacts,
    ExtractResult,
    FactsExperience,
    FactsLink,
    FactsPersonalInfo,
    JdAnalysis,
    TargetSection,
)
from tool_schemas import generate_cover_letter_tool_schema


def make_facts(**overrides: object) -> CVFacts:
    """Build a truthful sample ``CVFacts`` (the only permitted evidence source)."""
    base: dict[str, object] = {
        "personal_info": FactsPersonalInfo(
            name="Ada Lovelace",
            location="London, UK",
            email="ada@example.com",
            phone="+44 20 1234",
            links=[FactsLink(title="LinkedIn", url="https://linkedin.com/in/ada")],
        ),
        "experiences": [
            FactsExperience(
                role="Analyst",
                company="Analytical Engines",
                start_date="1840",
                end_date="1843",
                bullets=["Designed the first published algorithm in Python and FastAPI work"],
            ),
        ],
    }
    base.update(overrides)
    return CVFacts(**base)


def make_jd(**overrides: object) -> JdAnalysis:
    """Build a sample ``JdAnalysis``; ``requirements_must`` is the coverage source set."""
    base: dict[str, object] = {
        "role_title": "Senior Engineer",
        "company": "Globex",
        "keywords": ["Python", "FastAPI", "Pydantic", "PostgreSQL", "Docker"],
        "requirements_must": ["Python", "FastAPI"],
        "keyword_plan": {
            "Python": TargetSection.SKILLS,
            "FastAPI": TargetSection.EXPERIENCE,
        },
        "candidate_level": CandidateLevel.SENIOR_IC,
    }
    base.update(overrides)
    return JdAnalysis(**base)


def structured(
    body_words: int = 20, *, body: str | None = None, salutation: str | None = None
) -> str:
    """Build a fully-enveloped letter that PASSES ``_is_well_structured`` (salutation → body →
    sign-off, blank-line separated). Total word count = (salutation words) + body words + 3
    ("Sincerely," + "Ada Lovelace"); with the default "Dear Hiring Manager," that is body_words+6.
    """
    head = salutation if salutation is not None else "Dear Hiring Manager,"
    block = body if body is not None else " ".join(["word"] * body_words)
    return f"{head}\n\n{block}\n\nSincerely,\nAda Lovelace"


@pytest.fixture
def extract() -> ExtractResult:
    """A sample shared ``ExtractResult`` consumed by ``generate_cover_letter``."""
    return ExtractResult(facts=make_facts(), jd=make_jd(), flags=[])


# ──────────────────────────────────────────────────────────────────────────────
#  generate_cover_letter orchestrator
# ──────────────────────────────────────────────────────────────────────────────
async def test_cover_letter_structured_point_by_point(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    draft = structured(
        body=(
            "On Python I designed the first published algorithm at Analytical Engines. "
            "On FastAPI I delivered services there."
        )
    )
    mock_llm.call_tool.return_value = {"text": draft}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    # A well-structured, within-limit first draft is delivered with no regen.
    assert mock_llm.call_tool.call_count == 1
    # Evidence prose preserved point by point.
    for requirement in extract.jd.requirements_must:
        assert requirement in result.cover_letter
    # Structured against jd.requirements_must — they reach the model via the user prompt,
    # and the system prompt carries the point-by-point methodology.
    system, user, schema = mock_llm.call_tool.call_args.args
    for requirement in extract.jd.requirements_must:
        assert requirement in user
    assert "point" in system.lower()
    assert schema["name"] == "generate_cover_letter"


async def test_success_returns_cover_letter_result_with_empty_flags(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    draft = structured(body="A concise truthful cover letter grounded in real facts.")
    mock_llm.call_tool.return_value = {"text": draft}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert result.cover_letter == draft
    assert result.flags == []


# ──────────────────────────────────────────────────────────────────────────────
#  error handling (generate stage)
# ──────────────────────────────────────────────────────────────────────────────
async def test_provider_error_returns_redacted_generate_envelope(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    mock_llm.call_tool.side_effect = ProviderResponseError(
        "Provider rejected the request with bearer sk-or-test-dummy"
    )

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, dict)
    assert result["stage"] == "generate"
    assert result["error"]  # an actionable message is surfaced
    assert "sk-or-test-dummy" not in result["error"]  # API key redacted


async def test_openai_error_maps_to_generate_stage_with_distinct_message(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    mock_llm.call_tool.side_effect = OpenAIError("Request timed out after EXTRACT_TIMEOUT_S")

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, dict)
    assert result["stage"] == "generate"
    assert "timed out" in result["error"]  # distinct, actionable per-failure message


# ──────────────────────────────────────────────────────────────────────────────
#  consume extract.jd, no re-parse
# ──────────────────────────────────────────────────────────────────────────────
async def test_jd_is_reused_not_reparsed(mock_llm: MagicMock) -> None:
    # A sentinel requirement that appears ONLY via requirements_must — NOT in jd.keywords or
    # the facts — so 'the requirement reached the prompt' fails if requirements_must
    # serialization is dropped (it can no longer be satisfied by a keyword/facts leak).
    jd = make_jd(requirements_must=["Distributed tracing"])
    extract = ExtractResult(facts=make_facts(), jd=jd, flags=[])
    mock_llm.call_tool.return_value = {"text": structured(body="Evidence grounded in facts.")}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    # No re-parse: a well-structured within-limit draft makes exactly ONE function call (a
    # re-parse would be a second call_tool with the analyze schema), and it is the cover tool.
    assert mock_llm.call_tool.call_count == 1
    _system, user, schema = mock_llm.call_tool.call_args.args
    assert schema["name"] == "generate_cover_letter"
    assert jd.role_title in user  # extract.jd reused as-is
    assert "Distributed tracing" in user  # requirements_must specifically serialized


# ──────────────────────────────────────────────────────────────────────────────
#  truth-preserving generation + field guard
# ──────────────────────────────────────────────────────────────────────────────
async def test_truth_preserving_rule_drives_generation(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    draft = structured(body="Evidence drawn from real CV facts.")
    mock_llm.call_tool.return_value = {"text": draft}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    # Assert on the system prompt actually SENT to the model (call_args), not the module
    # constant — a regression that dropped or blanked the prompt must fail this test.
    sent_system = mock_llm.call_tool.call_args.args[0].lower()
    assert "never invent" in sent_system  # no fabrication
    assert "cv facts" in sent_system  # claims drawn only from CVFacts
    # The generator adds no claims of its own — the prose is the model's evidence draft.
    assert result.cover_letter == draft


async def test_missing_text_field_is_guarded_by_require_fields(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    mock_llm.call_tool.return_value = {"unexpected": "no text"}  # model omitted required `text`

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, dict)
    assert result["stage"] == "generate"
    assert "text" in result["error"].lower()  # the guard names the missing field


# ──────────────────────────────────────────────────────────────────────────────
#  anti-pattern ban-list
# ──────────────────────────────────────────────────────────────────────────────
async def test_anti_pattern_ban_list_enforced(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    mock_llm.call_tool.return_value = {
        "text": structured(body="I built data pipelines in Python at Analytical Engines.")
    }

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    # Enforcement mechanism: the ban-list is carried in the prompt actually SENT to
    # the model (call_args), not merely present in the module constant.
    sent_system = mock_llm.call_tool.call_args.args[0].lower()
    assert "ban-list" in sent_system
    assert "fan letter" in sent_system
    assert "drama" in sent_system
    assert "generic" in sent_system  # generic adjectives


# ──────────────────────────────────────────────────────────────────────────────
#  word cap ~300, capped compress-regen
# ──────────────────────────────────────────────────────────────────────────────
async def test_over_length_draft_triggers_single_compress_regen(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    over = " ".join(["word"] * (COVER_LETTER_MAX_WORDS + 50))  # strictly over the limit
    within = structured(body="A concise rewritten cover letter grounded in real facts.")
    mock_llm.call_tool.side_effect = [{"text": over}, {"text": within}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 2  # one compress-regen, then the loop ends
    assert result.cover_letter == within
    assert result.flags == []
    # The regen call instructs the model to fit within the word limit.
    _system, regen_user, _schema = mock_llm.call_tool.call_args_list[1].args
    assert str(COVER_LETTER_MAX_WORDS) in regen_user


async def test_residual_over_length_emits_non_fatal_flag(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    over1 = " ".join(["word"] * (COVER_LETTER_MAX_WORDS + 50))  # 350
    residual_words = COVER_LETTER_MAX_WORDS + 20  # 320 — the shortest, kept as best
    over2 = " ".join(["beta"] * residual_words)
    over3 = " ".join(["gamma"] * (COVER_LETTER_MAX_WORDS + 40))  # 340 — not shorter, dropped
    mock_llm.call_tool.side_effect = [{"text": over1}, {"text": over2}, {"text": over3}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 1 + COVER_LETTER_REGEN_CAP  # capped regens
    assert result.cover_letter == over2  # no truncation — the shortest draft kept intact
    over_flags = [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_OVER_LENGTH]
    assert len(over_flags) == 1
    assert over_flags[0].section == COVER_LETTER_SECTION
    assert str(residual_words) in over_flags[0].message  # residual word count named


async def test_within_limit_draft_no_regen_no_flag(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    mock_llm.call_tool.return_value = {
        "text": structured(body="A concise within-limit cover letter from real facts.")
    }

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 1  # no regen
    assert result.flags == []


# ──────────────────────────────────────────────────────────────────────────────
#  shared text cleanup pass
# ──────────────────────────────────────────────────────────────────────────────
async def test_shared_cleanup_pass_applied_to_output(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    dirty = structured(
        body="I hope this helps \U0001F680 — Python plays a crucial role in my FastAPI work."
    )
    mock_llm.call_tool.return_value = {"text": dirty}

    result = await generate_cover_letter(extract, llm=mock_llm)

    cover_letter = result.cover_letter
    assert "\U0001F680" not in cover_letter  # emoji stripped
    assert "—" not in cover_letter  # em-dash humanized
    assert "I hope this helps" not in cover_letter  # chatbot-residue ban-phrase
    assert "plays a crucial role" not in cover_letter  # stock-phrase ban-phrase
    # Genuine evidence content survives the cleanup pass.
    assert "Python" in cover_letter
    assert "FastAPI" in cover_letter


# ──────────────────────────────────────────────────────────────────────────────
#  deterministic structural gate (salutation → body → sign-off)
# ──────────────────────────────────────────────────────────────────────────────
async def test_flat_blob_letter_fails_structure_and_triggers_regen(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    flat_blob = (
        "I am a strong candidate for this role and my Python and FastAPI work proves it "
        "across many projects, just one undifferentiated block of prose with no envelope."
    )
    well_formed = structured(body="On Python and FastAPI I delivered real systems.")
    mock_llm.call_tool.side_effect = [{"text": flat_blob}, {"text": well_formed}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 2  # the flat blob failed the gate → one regen
    assert result.cover_letter == well_formed  # the structured re-roll is delivered


async def test_well_structured_letter_passes_gate_with_no_regen(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    well_formed = structured(
        body=(
            "Regarding Python, I designed the first published algorithm at Analytical Engines. "
            "Regarding FastAPI, I built the services there."
        )
    )
    mock_llm.call_tool.return_value = {"text": well_formed}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 1  # already well-structured → no regen
    assert result.cover_letter == well_formed
    assert result.flags == []


async def test_residual_unstructured_letter_ships_without_user_facing_flag(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # Every draft is a flat within-limit blob — the gate never passes; after the cap the best
    # draft is shipped WITHOUT any user-facing flag (structure is a shape gate, not a
    # surfaced gap). The candidate still receives a (truthful) letter.
    blob = "A truthful but flat within-limit letter with Python and FastAPI evidence."
    mock_llm.call_tool.return_value = {"text": blob}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert result.cover_letter == blob  # the best (only) draft is delivered, never truncated
    assert mock_llm.call_tool.call_count == 1 + COVER_LETTER_REGEN_CAP  # regened up to the cap
    assert result.flags == []  # NO structural flag surfaced to the user


# ──────────────────────────────────────────────────────────────────────────────
#  cover-letter gap surfacing removed
# ──────────────────────────────────────────────────────────────────────────────
async def test_success_path_emits_no_cover_letter_gap_flag(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # The letter covers only what the CV evidences and silently omits the rest (no fabrication).
    mock_llm.call_tool.return_value = {
        "text": structured(body="On Python I designed the first published algorithm.")
    }

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert result.cover_letter  # the letter is still returned
    assert [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_GAP] == []
    assert all("Missing" not in f.message for f in result.flags)


async def test_stray_legacy_gaps_field_is_ignored(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # Even if a model still returns the dropped `gaps` field, it never becomes a flag.
    mock_llm.call_tool.return_value = {
        "text": structured(body="On Python I designed the first published algorithm."),
        "gaps": ["FastAPI", "Kubernetes"],
    }

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_GAP] == []


def test_cover_letter_tool_schema_has_no_gaps_field() -> None:
    """The `gaps` property is removed from the tool contract (no consumer left)."""
    schema = generate_cover_letter_tool_schema()
    assert "gaps" not in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["text"]


# ──────────────────────────────────────────────────────────────────────────────
#  Robustness fixes
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("blank", ["", "   ", "\n\t  \n"])
async def test_empty_or_blank_text_is_provider_shape_failure(
    mock_llm: MagicMock, extract: ExtractResult, blank: str
) -> None:
    mock_llm.call_tool.return_value = {"text": blank}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, dict)  # never a CoverLetterResult with an empty letter
    assert result["stage"] == "generate"
    assert result["error"]


async def test_failed_compress_regen_falls_back_to_first_draft(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    over = " ".join(["word"] * (COVER_LETTER_MAX_WORDS + 50))  # over-length first draft
    mock_llm.call_tool.side_effect = [
        {"text": over},
        OpenAIError("Request timed out during compress"),  # optional regen blips
    ]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)  # NOT a hard {error, stage} envelope
    assert result.cover_letter == over  # the deliverable first draft is preserved
    over_flags = [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_OVER_LENGTH]
    assert len(over_flags) == 1  # residual over-length flag still surfaced (non-fatal)


async def test_word_cap_exact_boundary_no_regen_no_flag(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # structured() adds 6 envelope words, so body of MAX-6 lands the letter exactly at the cap.
    exact = structured(body_words=COVER_LETTER_MAX_WORDS - 6)
    assert len(exact.split()) == COVER_LETTER_MAX_WORDS  # exactly at the cap
    mock_llm.call_tool.return_value = {"text": exact}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 1  # `>` gate: exactly-at-cap does NOT regen
    assert result.flags == []  # nor flag


async def test_word_cap_one_over_boundary_triggers_regen(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    over_by_one = structured(body_words=COVER_LETTER_MAX_WORDS - 5)  # one over the cap
    assert len(over_by_one.split()) == COVER_LETTER_MAX_WORDS + 1
    within = structured(body="A concise rewritten cover letter from real facts.")
    mock_llm.call_tool.side_effect = [{"text": over_by_one}, {"text": within}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 2  # one-over the cap DOES regen
    assert result.cover_letter == within


async def test_residual_count_reflects_cleaned_text_not_raw(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # First draft is over the cap (triggers the regen gate); the regen is over the cap by raw
    # count, but its extra tokens are emoji that cleanup strips, dropping it to <= cap AND it is
    # well-structured, so the loop ends and no over-length flag is shown.
    raw_over = " ".join(["word"] * (COVER_LETTER_MAX_WORDS + 10))
    emoji_padded = structured(body=" ".join(["word"] * 290 + ["\U0001F680"] * 10))
    mock_llm.call_tool.side_effect = [{"text": raw_over}, {"text": emoji_padded}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert "\U0001F680" not in result.cover_letter  # emoji stripped by cleanup
    assert len(result.cover_letter.split()) <= COVER_LETTER_MAX_WORDS  # delivered <= cap
    # Cleaned text is within the limit, so no over-length flag is shown (no false ">300").
    assert [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_OVER_LENGTH] == []


# ──────────────────────────────────────────────────────────────────────────────
#  Correctness fix set — single cleaned basis + the compress-regen actually compresses
# ──────────────────────────────────────────────────────────────────────────────
# single cleaned basis: the empty-check, word gate, AND over-length flag all run on
# the CLEANED (delivered) text, because cleanup can both DROP and ADD whitespace tokens.


async def test_draft_cleaning_to_empty_is_provider_shape_failure(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # A non-empty RAW draft the cleanup pass reduces to "" (pure chatbot-residue ban-phrase).
    mock_llm.call_tool.return_value = {"text": "Of course!"}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, dict)  # never a CoverLetterResult with an empty letter
    assert result["stage"] == "generate"
    assert result["error"]


async def test_word_gate_uses_cleaned_count_not_raw(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    # 299 plain words + one emoji-GLUED token: raw count is exactly 300 (would NOT gate), but
    # cleanup turns the emoji into a space, splitting that token -> cleaned count 301, which
    # MUST trip the gate. The regen is well-structured + within, ending the loop.
    glued = " ".join(["word"] * 299 + ["alpha\U0001F680beta"])
    assert len(glued.split()) == COVER_LETTER_MAX_WORDS  # raw is exactly at the cap
    within = structured(body="A concise rewritten letter from real facts.")
    mock_llm.call_tool.side_effect = [{"text": glued}, {"text": within}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 2  # gate saw the cleaned 301 -> compress attempted


async def test_over_length_message_omits_compress_claim_when_no_compress_ran(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    over = " ".join(["word"] * (COVER_LETTER_MAX_WORDS + 50))
    mock_llm.call_tool.side_effect = [{"text": over}, OpenAIError("timeout during compress")]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    over_flags = [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_OVER_LENGTH]
    assert len(over_flags) == 1
    # The compress attempt errored, so the message must NOT claim a compress pass happened.
    assert "compress pass" not in over_flags[0].message


# the compress-regen actually compresses: it SENDS the over-length draft and keeps the
# SHORTER of the drafts (never a longer regen).


async def test_compress_regen_includes_first_draft_text(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    first = "MARKER_FIRST_DRAFT " + " ".join(["word"] * (COVER_LETTER_MAX_WORDS + 10))
    within = structured(body="Short rewritten letter from real facts.")
    mock_llm.call_tool.side_effect = [{"text": first}, {"text": within}]

    await generate_cover_letter(extract, llm=mock_llm)

    assert mock_llm.call_tool.call_count == 2
    regen_user = mock_llm.call_tool.call_args_list[1].args[1]
    assert "MARKER_FIRST_DRAFT" in regen_user  # the actual draft is sent to be compressed


async def test_compress_regen_never_delivers_longer_than_first(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    first = " ".join(["alpha"] * (COVER_LETTER_MAX_WORDS + 5))  # 305 words, over the cap
    longer = " ".join(["beta"] * (COVER_LETTER_MAX_WORDS + 50))  # 350 words (regen backfired)
    # Both regens backfire into longer drafts; the shorter first draft is kept across the cap.
    mock_llm.call_tool.side_effect = [{"text": first}, {"text": longer}, {"text": longer}]

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert result.cover_letter == first  # the shorter first draft is kept, not a 350-word regen
    over_flags = [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_OVER_LENGTH]
    assert len(over_flags) == 1


# the client is constructed INSIDE the try, so a missing-key OpenAIError at
# construction maps to the redacted {error, stage=generate} envelope (mirrors build_extract).


async def test_missing_credentials_maps_to_generate_envelope(
    extract: ExtractResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = await generate_cover_letter(extract)  # default llm=None -> constructed inside the try

    assert isinstance(result, dict)  # not a raw crash
    assert result["stage"] == "generate"
    assert result["error"]


# an empty requirements_must surfaces an honest flag rather than silently returning a
# generic, unstructured letter (the point-by-point design has nothing to match).


async def test_empty_requirements_must_emits_flag(mock_llm: MagicMock) -> None:
    jd = make_jd(requirements_must=[])
    extract = ExtractResult(facts=make_facts(), jd=jd, flags=[])
    mock_llm.call_tool.return_value = {
        "text": structured(body="A general but truthful letter from real facts.")
    }

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)  # not a hard failure
    assert result.cover_letter  # the letter is still returned
    no_req = [f for f in result.flags if f.kind == FlagKind.COVER_LETTER_NO_REQUIREMENTS]
    assert len(no_req) == 1  # the JD provided no structured requirements — surfaced honestly


# ──────────────────────────────────────────────────────────────────────────────
#  Truth-preserving / neutrality — thin evidence & no-company salutation
# ──────────────────────────────────────────────────────────────────────────────
async def test_thin_evidence_letter_passes_with_one_body_paragraph(
    mock_llm: MagicMock, extract: ExtractResult
) -> None:
    thin = structured(body="On Python I designed the first published algorithm.")  # 1 body block
    mock_llm.call_tool.return_value = {"text": thin}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 1  # a one-paragraph letter is still well-structured
    assert result.cover_letter == thin
    assert result.flags == []


async def test_no_company_jd_accepts_neutral_salutation(mock_llm: MagicMock) -> None:
    jd = make_jd(company=None)
    extract = ExtractResult(facts=make_facts(), jd=jd, flags=[])
    neutral = structured(
        salutation="Dear Hiring Manager,",
        body="On Python and FastAPI I delivered real systems at Analytical Engines.",
    )
    mock_llm.call_tool.return_value = {"text": neutral}

    result = await generate_cover_letter(extract, llm=mock_llm)

    assert isinstance(result, CoverLetterResult)
    assert mock_llm.call_tool.call_count == 1  # neutral-salutation letter passes the gate
    assert result.cover_letter.startswith("Dear Hiring Manager,")  # no fabricated company


@pytest.mark.parametrize(
    "block",
    [
        "Dear Hiring Manager,",  # canonical prefix
        "Hello Team,",  # another prefix
        "To whom it may concern,",  # multi-word prefix
        "With great interest in this role,",  # no prefix, but a single line ending in a comma
    ],
)
def test_is_salutation_accepts_prefixes_and_comma_fallback(block: str) -> None:
    assert _is_salutation(block) is True


@pytest.mark.parametrize(
    "block",
    [
        "I am applying for the backend engineer position.",  # body line: no prefix, no comma
        "Dear Hiring Manager,\nI write regarding the role.",  # two lines → not a lone salutation
    ],
)
def test_is_salutation_rejects_non_greetings(block: str) -> None:
    assert _is_salutation(block) is False


@pytest.mark.parametrize(
    "block",
    [
        "Sincerely,\nAda Lovelace",
        "Best regards,\nAda Lovelace",  # "best" sign-off word
        "Warm regards,\nAda Lovelace",  # "warm" sign-off word
        "Thank you,\nAda Lovelace",  # multi-word sign-off
        "Yours faithfully,\nAda Lovelace",  # "yours"
    ],
)
def test_is_sign_off_accepts_word_variety(block: str) -> None:
    assert _is_sign_off(block) is True


@pytest.mark.parametrize(
    "block",
    [
        "Sincerely,",  # one line only → no name line
        "I appreciate your time.\nAda Lovelace",  # first word is not a sign-off word
    ],
)
def test_is_sign_off_rejects_non_closings(block: str) -> None:
    assert _is_sign_off(block) is False


def test_is_well_structured_requires_salutation_body_sign_off() -> None:
    good = "Dear Hiring Manager,\n\nI deliver Python systems.\n\nBest regards,\nAda Lovelace"
    flat_blob = "Dear Hiring Manager, I deliver Python systems. Best regards, Ada Lovelace"

    assert _is_well_structured(good) is True
    assert _is_well_structured(flat_blob) is False  # single block → fails the shape gate


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
