"""Unit tests for the mechanical AI-tell cleanup layer.

Behavior-level tests for ``TextPreprocessing`` as the shared, stateless string→string
cleanup pass that both the CV and cover-letter generators call before keyword matching.
Char-level cleanup (invisibles/control stripped + homoglyph→Latin) is reused from the
existing ``strip``/``humanize``/``clean``; the new ``remove_ai_tells`` owns emoji
(language-agnostic) and the ban-list phrases (``CHATBOT_RESIDUE_BANLIST`` /
``STOCK_PHRASE_BANLIST``, gated English-only). Title-Case → sentence case is NOT
a cleanup rule — it is handled as downstream prompt guidance in CV/cover-letter generation,
since a regex cannot tell an AI heading from a real job title.
"""

from __future__ import annotations

import pytest

from helprers.text_preprocessing import TextPreprocessing

__all__ = ["TextPreprocessing"]

# ---------------------------------------------------------------------------
# Char-level cleanup strips watermarks.
# Reuse-verification: strip/clean already own this; these characterize the
# guarantee (detect()==0, homoglyph→Latin) on the existing code path.
# ---------------------------------------------------------------------------

_ZERO_WIDTH = "​"  # zero-width space
_BIDI = "‮"  # right-to-left override (BiDi control)
_CONTROL = "\x07"  # bell control char
_CYRILLIC_O = "о"  # Cyrillic 'о' homoglyph of Latin 'o'


def test_invisibles_and_control_chars_stripped() -> None:
    text = f"Sec{_ZERO_WIDTH}ure Pyth{_BIDI}on Da{_CONTROL}ta"

    cleaned = TextPreprocessing.clean(text)

    assert cleaned == "Secure Python Data"
    assert TextPreprocessing.detect(cleaned) == {}


def test_homoglyph_normalized_inside_mixed_script_keyword() -> None:
    dirty = f"Pyth{_CYRILLIC_O}n"  # reads "Python" but middle 'о' is Cyrillic

    cleaned = TextPreprocessing.clean(dirty)

    assert cleaned == "Python"
    assert cleaned.isascii()


def test_received_to_char_cleaned_removes_invisible_input_chars() -> None:
    text = f"Sec{_ZERO_WIDTH}ure{_BIDI} text{_CONTROL}"

    stripped = TextPreprocessing.strip(text)

    assert TextPreprocessing._INVISIBLE_INPUT_PATTERN.search(stripped) is None


# ---------------------------------------------------------------------------
# Mechanical AI-tells removed, English only.
# remove_ai_tells owns emoji (language-agnostic) and the ban-list phrases (English-only).
# Title-Case is downstream prompt guidance, NOT a cleanup rule. Curly quotes and em/en-dashes
# stay in humanize.
# ---------------------------------------------------------------------------

_ROCKET = "🚀"  # U+1F680
_PARTY = "🎉"  # U+1F389


def test_emoji_removed_from_body_and_headings() -> None:
    text = f"Results {_ROCKET}\nGreat Work {_PARTY} done"

    out = TextPreprocessing.remove_ai_tells(text)

    assert _ROCKET not in out
    assert _PARTY not in out
    assert out.isascii()


def test_misc_technical_emoji_removed() -> None:
    for emoji in ("⌚", "⏰"):
        out = TextPreprocessing.remove_ai_tells(f"Ship {emoji} now")

        assert emoji not in out


def test_misc_technical_legitimate_glyphs_preserved() -> None:
    for glyph in ("⌘", "⌀", "⌥", "⌨", "⏎", "⏏"):
        text = f"Uses {glyph} daily"

        assert glyph in TextPreprocessing.remove_ai_tells(text)


def test_chatbot_residue_phrases_removed_case_insensitive() -> None:
    for phrase in TextPreprocessing.CHATBOT_RESIDUE_BANLIST:
        text = f"Built the API. {phrase.upper()} Shipped it."

        out = TextPreprocessing.remove_ai_tells(text)

        assert phrase.lower() not in out.lower()
        assert "Built the API." in out
        assert "Shipped it." in out


def test_stock_phrases_removed_case_insensitive() -> None:
    for phrase in TextPreprocessing.STOCK_PHRASE_BANLIST:
        text = f"The team {phrase.title()} every quarter."

        out = TextPreprocessing.remove_ai_tells(text)

        assert phrase.lower() not in out.lower()
        assert "The team" in out


def test_let_me_know_if_removed_in_realistic_form() -> None:
    out = TextPreprocessing.remove_ai_tells("Let me know if you need anything...")

    assert "let me know if" not in out.lower()
    assert "you need anything" in out


def test_phrase_removal_leaves_no_punctuation_artifact() -> None:
    out = TextPreprocessing.remove_ai_tells("The system plays a crucial role, scaling fast.")

    assert " ," not in out
    assert "  " not in out
    assert out == "The system, scaling fast."


def test_banlist_phrase_inside_word_not_matched() -> None:
    text = "The dashboard displays a crucial role-based matrix."

    assert TextPreprocessing.remove_ai_tells(text) == text


def test_banlist_standalone_phrase_still_removed() -> None:
    out = TextPreprocessing.remove_ai_tells("It plays a crucial role here.")

    assert "plays a crucial role" not in out.lower()
    assert "It" in out and "here." in out


def test_phrase_between_punctuation_leaves_no_double_punct() -> None:
    comma = TextPreprocessing.remove_ai_tells("Fast, plays a crucial role, reliable.")
    assert ",," not in comma
    assert comma == "Fast, reliable."

    colon = TextPreprocessing.remove_ai_tells("Note: plays a crucial role; ships fast.")
    assert ":;" not in colon
    assert colon == "Note: ships fast."


def test_adjacent_punctuation_preserved_verbatim() -> None:
    for text in ("std::vector<int>", "Wrote Foo::bar and ns::Class", "Wait... really?"):
        assert TextPreprocessing.remove_ai_tells(text) == text


def test_title_case_cv_facts_preserved_verbatim() -> None:
    for fact in ("Senior Software Engineer", "New York", "AWS Cloud Architect"):
        assert TextPreprocessing.remove_ai_tells(fact) == fact


def test_em_dash_density_zero_after_clean() -> None:
    text = "Scaled systems — fast — and reliably."

    cleaned = TextPreprocessing.clean(text)

    assert cleaned.count("—") == TextPreprocessing.MAX_EM_DASH_DENSITY
    assert cleaned.count("—") == 0
    assert "-" in cleaned


def test_curly_quotes_normalized_by_humanize_not_remove_ai_tells() -> None:
    text = "He said “ship it” and ‘done’"

    cleaned = TextPreprocessing.clean(text)
    assert "“" not in cleaned and "”" not in cleaned
    assert "‘" not in cleaned and "’" not in cleaned
    assert '"ship it"' in cleaned

    # remove_ai_tells leaves curly quotes untouched — they are humanize's domain.
    assert "“" in TextPreprocessing.remove_ai_tells(text)


def test_english_flag_gates_banlist_only_not_emoji() -> None:
    text = f"Certainly! El equipo plays a crucial role {_ROCKET}"

    non_english = TextPreprocessing.remove_ai_tells(text, english=False)
    assert _ROCKET not in non_english  # emoji are language-agnostic
    assert "plays a crucial role" in non_english  # ban-list NOT applied
    assert "Certainly!" in non_english

    english = TextPreprocessing.remove_ai_tells(text, english=True)
    assert _ROCKET not in english  # emoji still stripped
    assert "plays a crucial role" not in english  # ban-list IS applied
    assert "Certainly!" not in english


def test_tells_removed_transition_no_emoji_no_banlist() -> None:
    text = f"Built APIs {_ROCKET}. Of course! The system stands as a testament to scale."

    out = TextPreprocessing.remove_ai_tells(text)

    assert _ROCKET not in out
    for phrase in (
        *TextPreprocessing.CHATBOT_RESIDUE_BANLIST,
        *TextPreprocessing.STOCK_PHRASE_BANLIST,
    ):
        assert phrase.lower() not in out.lower()


def test_non_english_emoji_adjacent_punctuation_leaves_no_orphan() -> None:
    assert (
        TextPreprocessing.remove_ai_tells(f"Listo {_ROCKET}, equipo.", english=False)
        == "Listo, equipo."
    )
    assert (
        TextPreprocessing.remove_ai_tells("Hecho ⏰.", english=False) == "Hecho."
    )


def test_emoji_replacement_preserves_token_boundary() -> None:
    assert TextPreprocessing.remove_ai_tells(f"Delivered{_ROCKET}results") == "Delivered results"
    assert TextPreprocessing.remove_ai_tells(f"Done {_ROCKET} now") == "Done now"


def test_cr_lf_acronyms_preserved_verbatim() -> None:
    text = "Built CRM integrations, led SCRUM, designed CRUD APIs, managed LFS storage"

    cleaned = TextPreprocessing.clean(text)

    for token in ("CRM", "SCRUM", "CRUD", "LFS"):
        assert token in cleaned


def test_standalone_cr_lf_marker_still_stripped() -> None:
    assert TextPreprocessing.clean("Section CR Title LF here") == "Section Title here"


def test_inter_letter_period_in_abbreviations_and_emails_preserved() -> None:
    text = "I hold a Ph.D., shipped U.S. systems, e.g. payments; contact ada@example.com"

    cleaned = TextPreprocessing.clean(text)

    for token in ("Ph.D.", "U.S.", "e.g.", "ada@example.com"):
        assert token in cleaned


def test_inter_letter_middle_dot_watermark_still_stripped() -> None:
    cleaned = TextPreprocessing.clean("Py·thon")
    assert "·" not in cleaned  # the middle-dot watermark is gone
    assert cleaned == "Py thon"


# ---------------------------------------------------------------------------
# Coverage computed on cleaned text.
# The transformer guarantees ASCII-safe, detect()==0 output that keyword
# matching consumes — match basis is match(clean(text)).
# ---------------------------------------------------------------------------


def test_homoglyph_keyword_matches_latin_after_clean() -> None:
    keyword = "Python"
    dirty = f"Pyth{_CYRILLIC_O}n"  # Cyrillic 'о' spoof inside the keyword

    assert dirty != keyword
    assert TextPreprocessing.clean(dirty) == keyword


def test_clean_output_detect_zero_ready_for_matching() -> None:
    text = f"Lead{_ZERO_WIDTH} Backend{_BIDI} Engineer with Pyth{_CYRILLIC_O}n"

    cleaned = TextPreprocessing.clean(text)

    assert TextPreprocessing.detect(cleaned) == {}
    assert cleaned.isascii()


# ---------------------------------------------------------------------------
# Shared pass for CV and cover letter.
# The same transformer applied to both outputs yields identical cleaning.
# ---------------------------------------------------------------------------


def test_same_clean_pass_applied_to_cv_and_cover_letter() -> None:
    raw = f"Led{_ZERO_WIDTH} the team — shipped “fast”"

    cv_clean = TextPreprocessing.clean(raw)  # CV section body
    cover_letter_clean = TextPreprocessing.clean(raw)  # cover-letter body

    assert cv_clean == cover_letter_clean
    assert cv_clean == 'Led the team - shipped "fast"'
    assert TextPreprocessing.detect(cv_clean) == {}


# ---------------------------------------------------------------------------
# Idempotent, fail-loud re-run reports zero.
# clean is pure/idempotent; an internal exception must propagate (fail loud).
# ---------------------------------------------------------------------------


def test_clean_is_idempotent() -> None:
    text = f"Scaled{_ZERO_WIDTH} systems — “fast” and Pyth{_CYRILLIC_O}n"

    once = TextPreprocessing.clean(text)
    twice = TextPreprocessing.clean(once)

    assert twice == once
    assert TextPreprocessing.detect(twice) == {}


def test_remove_ai_tells_is_idempotent() -> None:
    text = f"Built APIs {_ROCKET}. Of course! It stands as a testament to scale."

    once = TextPreprocessing.remove_ai_tells(text)
    twice = TextPreprocessing.remove_ai_tells(once)

    assert twice == once


def test_remove_ai_tells_idempotent_with_adjacent_punctuation() -> None:
    text = f"Fast, plays a crucial role, {_ROCKET} reliable. Of course!"

    once = TextPreprocessing.remove_ai_tells(text)
    twice = TextPreprocessing.remove_ai_tells(once)

    assert twice == once


def test_rerun_watermark_count_is_zero() -> None:
    text = f"Hidden{_ZERO_WIDTH}{_BIDI}markers{_CONTROL} everywhere"

    counts = TextPreprocessing.detect(TextPreprocessing.clean(text))

    assert sum(counts.values()) == TextPreprocessing.WATERMARK_COUNT_AFTER_PASS
    assert sum(counts.values()) == 0


def test_remove_ai_tells_fails_loud_on_internal_exception() -> None:
    with pytest.raises(TypeError):
        TextPreprocessing.remove_ai_tells(123)  # type: ignore[arg-type]


def test_clean_fails_loud_on_internal_exception() -> None:
    with pytest.raises(TypeError):
        TextPreprocessing.clean(123)  # type: ignore[arg-type]
