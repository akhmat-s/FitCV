from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Pattern


class TextPreprocessing:
    """Detects and removes common textual watermarks and humanizes text.

    All functionality is exposed via *class* methods so that you can use it
    without instantiating. If you prefer an object instance, instantiation is
    cheap and stateless.
    """

    # ---------------------------------------------------------------------
    #  Character classes that frequently serve as hidden markers
    # ---------------------------------------------------------------------
    _REGEXES: Dict[str, str] = {
        # 1. Zero‑width & BOM
        "zero_width": r"[\u200B\u200C\u200D\u2060\u180E\uFEFF]",
        # 2. BiDi control & isolates
        "bidi_control": r"[\u202A-\u202E\u2066-\u2069]",
        # 3. Directional marks (LRM/RLM)
        "directional_marks": r"[\u200E\u200F]",
        # 4. Exotic spaces
        "exotic_spaces": r"[\u2000-\u200A\u202F\u205F\u3000]",
        # 5. Non‑printing control chars (C0 & C1, minus TAB/CR/LF)
        "control_chars": r"[\x00-\x08\x0B\x0C\x0E-\x1F\u007F-\u009F]",
        # 6. Combining diacritics
        "combining_diacritics": r"[\u0300-\u036F]",
        # 7. Invisible operators
        "invisible_operators": r"[\u2061-\u2063]",
        # 8. Mongolian and other separators
        "separators": r"[\u180B-\u180D]",
        # 9. Variation selectors
        "variation_selectors": r"[\uFE00-\uFE0F]",
        # 10. Full-width/half-width character forms
        "width_forms": r"[\uFF00-\uFFEF]",
        # 11. Inter-word spaces and markers
        "word_spacers": r"[\u00B7\u2022\u2023\u2043\u204C\u204D\u2219\u25E6\u2043\u2219\u22C5\u30FB]",
        # 12. Special character sequences
        "spacing_patterns": r"·|\s·\s|\sCR\s|\sLF\s",
        # 13. End-of-line markers — ONLY standalone "CR"/"LF" markers
        # (word boundary on both sides), so we don't mangle CRM/SCRUM/CRUD/LFS.
        "line_markers": r"(?<!\S)(?:CR|LF)(?!\S)",
    }

    # Build one compiled regex with named groups so we know what we hit.
    _MASTER_PATTERN: Pattern[str] = re.compile(
        "|".join(f"(?P<{name}>{pat})" for name, pat in _REGEXES.items())
    )

    # ---------------------------------------------------------------------
    #  Character mappings for humanization
    # ---------------------------------------------------------------------
    _HUMANIZE_MAPPINGS: Dict[str, str] = {
        # Smart quotes to straight quotes
        # Mapped explicitly (U+201C/U+201D/U+2018/U+2019) for curly-quote removal.
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "«": '"',
        "»": '"',
        "„": '"',
        # Dashes to hyphen
        "—": "-",  # em dash
        "–": "-",  # en dash
        "‒": "-",  # figure dash
        "―": "-",  # horizontal bar
        "‐": "-",  # hyphen
        # Ellipsis
        "…": "...",
        # Other common typographical replacements
        "•": "*",  # bullet
        "·": "*",  # middle dot
        "′": "'",  # prime
        "″": '"',  # double prime
        "×": "x",  # multiplication sign
        "ε": "e",  # Greek epsilon to Latin e
        "а": "a",  # Cyrillic a to Latin a
        # More homoglyphs
        "ӓ": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
    }

    # Compile the humanization pattern once
    _HUMANIZE_PATTERN: Pattern[str] = re.compile(
        "|".join(re.escape(char) for char in _HUMANIZE_MAPPINGS.keys())
    )

    # ------------------------------------------------------------------
    #  Input normalization — a NARROW char layer for RAW user input
    #  (CV/JD). Strips invisibles and maps homoglyph spoof characters to
    #  Latin ONLY inside mixed-script words. Kept separate from strip()/
    #  clean(), whose whitespace-collapse and AI-tell heuristics target
    #  MODEL OUTPUT and corrupt real resumes if run over input.
    # ------------------------------------------------------------------

    #: Truly invisible / zero-width / control characters removed from input. Built from
    #: _REGEXES.items() (the outermost iterable, evaluated in class scope) so the class
    #: attribute is visible inside the comprehension.
    _INVISIBLE_INPUT_PATTERN: Pattern[str] = re.compile(
        "|".join(
            pattern
            for name, pattern in _REGEXES.items()
            if name
            in {
                "zero_width",
                "bidi_control",
                "directional_marks",
                "invisible_operators",
                "separators",
                "variation_selectors",
                "control_chars",
            }
        )
    )

    #: Homoglyph spoof characters (Cyrillic/Greek look-alikes) → Latin. Applied only
    #: inside mixed-script words so legitimately non-Latin text is preserved.
    _HOMOGLYPH_MAP: dict[str, str] = {
        "ε": "e",
        "а": "a",
        "ӓ": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
    }

    # ------------------------------------------------------------------
    #  Mechanical AI-tell layer (English-only). Removal is whole-phrase,
    #  case-insensitive.
    #
    #  humanize() overlap — curly quotes and em/en-dashes are already owned
    #  by humanize() (em-dash→"-" gives MAX_EM_DASH_DENSITY=0). remove_ai_tells()
    #  owns ONLY emoji and the ban-list phrases below; it MUST NOT re-implement
    #  quote or dash handling. Title-Case→sentence-case is a JUDGMENT tell handled
    #  as prompt guidance in CV/cover-letter generation, NOT in this gate.
    # ------------------------------------------------------------------

    #: Invariant (verified by tests, not enforced at runtime): post-pass em-dash
    #: count equals this (English-only); satisfied by humanize().
    MAX_EM_DASH_DENSITY: int = 0

    #: Invariant (verified by tests, not enforced at runtime): post-pass detect()
    #: returns this many watermark markers.
    WATERMARK_COUNT_AFTER_PASS: int = 0

    #: Chatbot residue removed whole-phrase, case-insensitive. "Let me know if"
    #: is stored WITHOUT a trailing ellipsis: humanize() rewrites …→... upstream and
    #: models emit "..." anyway, so a literal U+2026 here would never match live text.
    CHATBOT_RESIDUE_BANLIST: tuple[str, ...] = (
        "I hope this helps",
        "Let me know if",
        "Of course!",
        "Certainly!",
    )

    #: Stock inflated phrases removed whole-phrase, case-insensitive.
    STOCK_PHRASE_BANLIST: tuple[str, ...] = (
        "stands as a testament",
        "in the evolving landscape",
        "plays a crucial role",
        "marks a pivotal moment",
    )

    #: Single precompiled, word-boundary-anchored, case-insensitive alternation over
    #: BOTH ban-lists, matching the file's _HUMANIZE_PATTERN/_MASTER_PATTERN
    #: convention (one pass, compiled once). The (?<!\w)…(?!\w) anchors stop a phrase
    #: from matching INSIDE a real word — e.g. "displays a crucial role" must NOT lose
    #: "plays a crucial role". \b…\b is wrong here: a phrase ending in punctuation
    #: ("Of course!") has no word boundary after "!". Built with the ban-list tuples as
    #: the outermost iterable so they resolve in class scope (cf. _INVISIBLE_INPUT_PATTERN).
    _BAN_PATTERN: Pattern[str] = re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(phrase)
            for phrase in CHATBOT_RESIDUE_BANLIST + STOCK_PHRASE_BANLIST
        )
        + r")(?!\w)",
        re.IGNORECASE,
    )

    #: Emoji removed from body and headings. Variation selectors are already
    #: handled by the invisibles layer, so they are not repeated here. The
    #: clock/watch/media block is ENUMERATED (⌚ U+231A, ⌛ U+231B, and the media+clock
    #: run U+23E9–U+23FA incl. ⏰/⏱/⏲/⏳), NOT the whole U+2300–U+23FF Misc-Technical
    #: block — that block also holds legitimate glyphs (⌘ ⌀ ⌥ ⌨ ⏎ ⏏) a CV may use.
    _EMOJI_PATTERN: Pattern[str] = re.compile(
        "["
        "\U0001f300-\U0001faff"
        "\U0000231a\U0000231b\U000023e9-\U000023fa"  # clock/watch + media controls ONLY
        "\U00002600-\U000027bf"
        "\U0001f000-\U0001f0ff"
        "\U00002b00-\U00002bff"
        "\U0001f1e6-\U0001f1ff"
        "]+"
    )

    @classmethod
    def detect(cls, text: str) -> Counter[str]:
        """Return counts of each watermark category found in *text*."""
        counts: Counter[str] = Counter()
        
        # Check for CR/LF markers
        cr_lf_markers = re.findall(r'\s(?:CR|LF)\s', text)
        if cr_lf_markers:
            counts["cr_lf_markers"] = len(cr_lf_markers)
            
        # Check for dots between words
        dot_spacers = re.findall(r'\S\s+[·\.\u00B7]\s+\S', text)
        if dot_spacers:
            counts["dot_spacers"] = len(dot_spacers)
        
        # Check for special patterns
        if re.search(cls._REGEXES["spacing_patterns"], text):
            counts["spacing_patterns"] = len(re.findall(cls._REGEXES["spacing_patterns"], text))
        
        # Detect individual characters
        for match in cls._MASTER_PATTERN.finditer(text):
            for name in cls._REGEXES:
                # Skip patterns; they are already handled above
                if name in ["spacing_patterns", "line_markers"]:
                    continue
                    
                if match.group(name):
                    counts[name] += 1
                    break

        # Special check for repeating space patterns
        # Check for spaces between every character
        consistent_spaces = re.findall(r"\S\s\S", text)
        if len(consistent_spaces) > 10:  # More than 10 words with spaces between them
            # Check the ratio of such patterns to the text length
            if len(consistent_spaces) / len(text) > 0.1:  # 10% of the text has such patterns
                counts["consistent_spacing_patterns"] = len(consistent_spaces)
        
        # Special check for spaces between EVERY letter (very characteristic of some LLMs)
        letter_spaces = re.findall(r'\w\s\w\s\w\s\w', text)
        if letter_spaces:
            counts["letter_spacing"] = len(letter_spaces)
                
        return counts

    @classmethod
    def strip(cls, text: str, *, collapse_whitespace: bool = True) -> str:
        """Remove watermark characters and optionally tidy whitespace."""
        # 1. Remove ONLY standalone CR/LF markers (with a whitespace boundary on both
        # sides), not the letters "CR"/"LF" inside words — otherwise CRM/SCRUM/CRUD/LFS get mangled.
        cleaned = re.sub(r'(?<!\S)(?:CR|LF)(?!\S)', ' ', text)
        
        # 2. Handle dots between words (special case from the screenshot)
        # Replace dots between words with spaces
        cleaned = re.sub(r'(\w)·(\w)', r'\1 \2', cleaned)  # Dot without spaces - add a space
        cleaned = re.sub(r'(\S)\s*·\s*(\S)', r'\1 \2', cleaned)  # Keep a space between words

        # 3. Remove patterns with dots between words
        cleaned = re.sub(r'(\S)\s+[·\.\u00B7]\s+(\S)', r'\1 \2', cleaned)
        
        # 4. Remove sequence patterns
        cleaned = re.sub(cls._REGEXES["spacing_patterns"], " ", cleaned)
        
        # 5. Remove individual marker characters
        cleaned = cls._MASTER_PATTERN.sub("", cleaned)
        
        # 6. Special handling for inter-word dots and characters
        cleaned = re.sub(r'(\w)\s+[·\.\u00B7]\s+(\w)', r'\1 \2', cleaned)
        
        # 7. Middle-dot (U+00B7) watermark wedged between two letters -> remove it.
        # Narrowed to the middle-dot watermark ONLY. The old class also held an
        # ASCII '.', so this rule ate ordinary periods inside abbreviations,
        # degrees, and emails on non-adversarial prose: "Ph.D."->"PhD.", "e.g."->"eg.",
        # "ada@example.com"->"ada@examplecom". A real period between letters is
        # legitimate text and must survive verbatim (third instance of the substring-
        # heuristic class, after the ban-list and the CR/LF marker). Middle-dots
        # between word chars are already collapsed to a space by rules 2/3 above.
        cleaned = re.sub(r'([a-zA-Z])[·\u00B7]([a-zA-Z])', r'\1\2', cleaned)
        
        # 8. Handle cases where a watermark sits between every WORD
        # New approach: more selective dot removal
        cleaned = re.sub(r'([a-zA-Z0-9]):·', r'\1: ', cleaned)  # After a colon
        cleaned = re.sub(r'([a-zA-Z0-9]),·', r'\1, ', cleaned)  # After a comma

        # 9. Handle cases with spaces between all letters
        if re.search(r'\w\s\w\s\w\s\w', cleaned):  # Watermark sign - spaces between all characters
            cleaned = re.sub(r'(\w)\s(\w)', r'\1\2', cleaned)
        
        # 10. Normalize space sequences
        if collapse_whitespace:
            # Replace multiple consecutive spaces/tabs with a single space
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            # Trim spaces around newlines
            cleaned = re.sub(r" *\n *", "\n", cleaned)
        
        return cleaned.strip()

    @classmethod
    def humanize(cls, text: str) -> str:
        """Convert typographical characters to plain 'human' equivalents.
        
        This converts fancy quotes, dashes, and other typographical characters
        that LLMs might use to their simpler equivalents that humans typically
        type directly.
        """
        def replace_char(match):
            return cls._HUMANIZE_MAPPINGS[match.group(0)]
        
        return cls._HUMANIZE_PATTERN.sub(replace_char, text)

    # Call-order contract — cleanup runs BEFORE keyword
    # matching, so ATS coverage is computed on cleaned output and homoglyphs never
    # undercount (match basis = match(clean(text))). Shared-pass contract —
    # clean() = strip + humanize ONLY; the mechanical layer (remove_ai_tells) is
    # applied SEPARATELY by consumers and is NOT composed into clean(). clean() is the
    # single string->string entry point reused by BOTH the CV and the cover letter;
    # idempotent and fail-loud (no swallowed exceptions). Nested traversal of the
    # assembled bodies + the remove_ai_tells call site are owned by the CV and
    # cover-letter generators.
    @classmethod
    def clean(
        cls,
        text: str,
        *,
        collapse_whitespace: bool = True,
        humanize: bool = True,
    ) -> str:
        """High‑level helper to clean and optionally humanize text.

        Parameters
        ----------
        text : str
            Input string to inspect.
        collapse_whitespace : bool, default True
            If *True*, post‑processes the result so that multiple spaces created
            by removal are collapsed and leading/trailing whitespace trimmed.
        humanize : bool, default True
            If *True*, converts typographical characters to plain text
            equivalents after removing watermarks.
        """
        cleaned = cls.strip(text, collapse_whitespace=collapse_whitespace)

        if humanize:
            cleaned = cls.humanize(cleaned)

        return cleaned

    @classmethod
    def _demangle_word(cls, match: re.Match[str]) -> str:
        """Map homoglyph spoof chars to Latin, but only inside a mixed-script word."""
        word = match.group(0)
        # An all-non-Latin run (e.g. a Cyrillic name) is preserved verbatim.
        if not re.search(r"[A-Za-z]", word):
            return word
        # A pure-Latin run has nothing to map.
        if not any(char in cls._HOMOGLYPH_MAP for char in word):
            return word
        return "".join(cls._HOMOGLYPH_MAP.get(char, char) for char in word)

    @classmethod
    def normalize_input(cls, text: str) -> str:
        """Normalize RAW user input (CV/JD) for the extract pass.

        Char layer ONLY: strip zero-width/invisible characters and map homoglyph
        spoof characters (Cyrillic/Greek look-alikes) to Latin — but only inside
        mixed-script words, so legitimately non-Latin text (e.g. a Cyrillic name)
        is preserved verbatim (truth-preserving). Deliberately NARROWER than
        ``clean()``: no whitespace collapse and no mechanical AI-tell stripping,
        both of which corrupt real resumes and are OUTPUT-only concerns.
        """
        stripped = cls._INVISIBLE_INPUT_PATTERN.sub("", text)
        return re.sub(r"\w+", cls._demangle_word, stripped)

    @classmethod
    def remove_ai_tells(cls, text: str, *, english: bool = True) -> str:
        """Remove mechanical AI-tells from MODEL OUTPUT.

        Owns ONLY: emoji (language-agnostic), plus — English-only — chatbot residue
        and stock inflated phrases (``CHATBOT_RESIDUE_BANLIST`` /
        ``STOCK_PHRASE_BANLIST``, whole-phrase, word-boundary-anchored,
        case-insensitive).

        English gate: when ``english`` is False only the language-specific
        tells (the ban-list phrases) are skipped, so legitimate non-English phrasing is
        preserved. Emoji are stripped on every run — they are not an English-specific
        tell. The punctuation/space normalization at the end is language-neutral
        mechanics and runs UNCONDITIONALLY, so a removed tell never leaves a NEW tell
        in either path. Curly quotes and em/en-dashes are NOT handled here — they are
        owned by ``humanize``. Title-Case→sentence case is NOT done here either: a regex
        cannot distinguish an AI heading from a real job title / proper noun / acronym,
        so it is downstream prompt guidance, not a deterministic rule (truth-preserving).

        Fail loud: no try/except wraps the body — an internal encoding/regex error
        propagates rather than returning un-cleaned text.
        """
        # Emoji → a single SPACE (not ""), so two words flanking an emoji are not fused
        # into one token (truth-preserving for ATS); the final collapse tidies the gap.
        cleaned = cls._EMOJI_PATTERN.sub(" ", text)

        if english:
            # One precompiled, word-boundary-anchored pass (_BAN_PATTERN) — a ban
            # phrase never matches inside a real word ("displays a crucial role" keeps
            # its text).
            cleaned = cls._BAN_PATTERN.sub("", cleaned)

        # Punctuation/space normalization — language-neutral, so it runs for EVERY path.
        # It repairs only the gap a removal leaves, never punctuation elsewhere:
        #   1. a punctuation pair stranded ACROSS A SPACE (", ," / ": ;") collapses to a
        #      single mark. The required space is what distinguishes a removal artifact
        #      from legitimate adjacent punctuation ("std::vector", "...") — which, being
        #      space-free, is left untouched.
        #   2. orphaned punctuation is pulled back onto the preceding word (" ," / " .").
        cleaned = re.sub(r"([,;:])[ \t]+[,;:]", r"\1 ", cleaned)
        cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)

        # Collapse the gaps left by removed emoji/phrases and drop a leading orphan
        # comma, per line (newlines kept).
        lines = cleaned.split("\n")
        return "\n".join(
            re.sub(r"^[ \t]*,[ \t]*", "", re.sub(r"[ \t]{2,}", " ", line)).strip()
            for line in lines
        )

    # Title-Case→sentence-case recasing was REMOVED here (truth-
    # preservation) — it now lives as prompt guidance in CV/cover-letter generation,
    # not in this mechanical gate.

    # ------------------------------------------------------------------
    #  Convenience instance interface
    # ------------------------------------------------------------------
    def __call__(self, text: str, **kwargs):  # type: ignore[override]
        """Allow an instance to be used as a callable transformer."""
        return self.clean(text, **kwargs)


__all__ = ["TextPreprocessing"]

# Example of using the TextPreprocessing
if __name__ == "__main__":
    # Real text from an LLM with possible watermarks
    llm_text = """UX Design: Crafting Seamless Digital Experiences

User Experience (UX) design is the process of creating products that provide meaningful, intuitive, and satisfying experiences for users. It goes beyond aesthetics—focusing on how a product feels, how it functions, and how easily users can achieve their goals.

Great UX design is rooted in empathy. It starts with understanding the users—their behaviors, needs, and frustrations. Through research, prototyping, and testing, UX designers refine digital interactions to ensure clarity, efficiency, and enjoyment.

From mobile apps to websites, every touchpoint matters. A smooth onboarding flow, clear navigation, fast load times, and accessible interfaces can turn a casual visitor into a loyal user.

In short, UX design is not just about making things look good—it's about making them work beautifully."""
    
    # Create text with dots between words (as in the screenshot)
    text_with_dots = "UX·Design:·Crafting·Seamless·Digital·Experiences"
    
    # Detect watermarks
    watermarks = TextPreprocessing.detect(llm_text)
    
    print("===== LLM text analysis =====")
    if watermarks:
        print("\n🔍 Watermarks detected:")
        for category, count in watermarks.items():
            print(f"  • {category}: {count}")
    else:
        print("\n✓ No watermarks detected")

    # Test text with dots
    print("\n===== Test of text with dots between words =====")
    print(f"Original: {text_with_dots}")
    print(f"Processed: {TextPreprocessing.clean(text_with_dots)}")

    # Simple usage example - just pass the text
    processed_text = TextPreprocessing.clean(llm_text)
    print("\n===== Processed text (cleaned and humanized by default) =====")
    print(processed_text)

    # Example of usage as a callable object - even simpler
    cleaner = TextPreprocessing()
    short_example = "Test of 'smart' quotes — and dashes…"
    print(f"\nOriginal text: {short_example}")
    print(f"Processed: {cleaner(short_example)}")
