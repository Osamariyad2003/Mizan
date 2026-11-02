"""Arabic text handling.

Two unrelated problems, both of which make retrieval fail silently rather than
loudly:

1. **Spelling variation.** The same word is written several valid ways --
   alef with or without hamza, ta marbuta or ha, with or without diacritics,
   Arabic-Indic or Western digits -- and the definite article and attached
   prepositions are part of the word ("al-qanun", "wa-l-qanun", "li-l-madda").
   Exact-token keyword search treats every variant as a different word.
   :func:`normalize_for_search` and :func:`light_stem` fold them together. They
   are applied identically to documents and queries, so they only need to be
   *consistent*, not linguistically perfect.

2. **Visual-order PDF text.** Some PDFs store Arabic glyphs in display order.
   Plain extraction then loses the spaces between words; layout extraction
   keeps the spaces but returns each line reversed. :func:`visual_to_logical`
   undoes the reversal. The loader only reaches for it when plain extraction is
   measurably broken, so well-formed PDFs are untouched.

Code points are written as escapes throughout: several are combining marks or
invisible, and a literal copy of them in source is impossible to review.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic, Arabic Supplement, and the two presentation-form blocks.
_ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_ARABIC_WORD = re.compile(r"[\u0600-\u06FF]+")
_PRESENTATION_FORM = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")

# Harakat (U+064B-065F), superscript alef (U+0670), honorific marks
# (U+0610-061A) and Quranic annotation marks (U+06D6-06ED). Stripped *before*
# tokenizing: they are combining marks, which a \w-based tokenizer treats as
# word breaks -- splitting a vowelled word into fragments.
_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)
_TATWEEL = "\u0640"  # kashida: a stretching character with no meaning

_ALEF = "\u0627"
_FOLD = str.maketrans(
    {
        "\u0623": _ALEF,  # alef with hamza above
        "\u0625": _ALEF,  # alef with hamza below
        "\u0622": _ALEF,  # alef with madda
        "\u0671": _ALEF,  # alef wasla
        "\u0649": "\u064A",  # alef maqsura -> ya
        "\u0629": "\u0647",  # ta marbuta -> ha
        **{chr(0x0660 + d): str(d) for d in range(10)},  # Arabic-Indic digits
        **{chr(0x06F0 + d): str(d) for d in range(10)},  # Extended (Persian) digits
    }
)

# Definite article, alone or fused with a preposition or conjunction, longest
# first: wa-bi-al, wa-ka-al, wa-li-l, wa-al, fa-al, bi-al, ka-al, li-l, al.
# Applied after folding, so they only need to match folded text.
#
# A bare leading waw is deliberately *not* stripped: it is as often a root
# letter (wizara "ministry", wathiqa "document", wazir "minister") as the
# conjunction "and", and stripping it would break the match between the
# definite and indefinite forms of exactly those words.
_ARTICLE_PREFIXES = (
    "\u0648\u0628\u0627\u0644",  # wbal
    "\u0648\u0643\u0627\u0644",  # wkal
    "\u0648\u0644\u0644",        # wll
    "\u0648\u0627\u0644",        # wal
    "\u0641\u0627\u0644",        # fal
    "\u0628\u0627\u0644",        # bal
    "\u0643\u0627\u0644",        # kal
    "\u0644\u0644",              # ll
    "\u0627\u0644",              # al
)
_MIN_STEM = 2  # keeps two-letter legal terms such as haqq ("right") intact


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_CHAR.search(text))


def normalize_for_search(text: str) -> str:
    """Fold spelling variants so equivalent Arabic words compare equal.

    Returns non-Arabic text unchanged, so it is safe on mixed-language input.
    """
    if not has_arabic(text):
        return text
    if _PRESENTATION_FORM.search(text):
        # PDF glyph codes -> the base letters people actually type.
        text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    return text.translate(_FOLD)


def light_stem(token: str) -> str:
    """Strip the definite article and fused prefixes from one folded token."""
    for prefix in _ARTICLE_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= _MIN_STEM:
            return token[len(prefix):]
    return token


# Runs that keep left-to-right order even inside a visual-order Arabic line:
# numbers in either digit set and Latin words, with internal separators.
_LTR_UNIT = r"[A-Za-z0-9\u0660-\u0669\u06F0-\u06F9]+"
_LTR_RUN = re.compile(_LTR_UNIT + r"(?:[.,:/%\-]" + _LTR_UNIT + r")*")


def visual_to_logical(line: str) -> str:
    """Reverse a visual-order Arabic line back into reading order.

    Reversing the whole line restores the Arabic but also reverses embedded
    numbers and Latin words, which were already in the right order -- so each
    of those runs is flipped back afterwards ("21" stays "12"). Lines with no
    Arabic are returned untouched.
    """
    if not has_arabic(line):
        return line
    reversed_line = line[::-1]
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], reversed_line)


# --------------------------------------------------------------------------
# Presentation forms and word-order reversal (e.g. PDFs exported by Qistas)
# --------------------------------------------------------------------------

# Initial-form ghain. An initial form cannot end a word, so at a word end it is
# a mis-mapped glyph: some fonts decode the tanween-fatha-on-alef ending
# ("wafqan", "ilzamiyyan") to this code point. Must be fixed *before* folding,
# which erases the positional information that makes it detectable.
_GHAIN_INITIAL_AT_WORD_END = re.compile("\uFECF(?=[\\s.,:\u060C\u061B\u061F)]|$)")


def fold_presentation_forms(text: str) -> str:
    """Replace PDF glyph codes (U+FB50-FDFF, U+FE70-FEFF) with base letters.

    Only those characters are touched; running NFKC over the whole text would
    also rewrite unrelated characters such as fractions and superscripts.
    """
    if not _PRESENTATION_FORM.search(text):
        return text
    text = _GHAIN_INITIAL_AT_WORD_END.sub(_ALEF, text)
    return _PRESENTATION_FORM.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)


_SENTENCE_PUNCT = ".:\u060C\u061B"  # . : Arabic comma, Arabic semicolon


def word_order_reversed(text: str, min_lines: int = 5) -> bool:
    """Detect text whose *words* are in reverse order on each line.

    In normal Arabic text, sentence punctuation ends lines; in word-reversed
    extraction it starts them (".bil-hadith ... al-mutadarrir"). Whole lines
    are used rather than individual tokens because reversed clause numbering
    ("1. tathbut" -> "tathbut. 1") puts a period at a token end and would
    otherwise masquerade as normal text.
    """
    starts = ends = 0
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 3 or not has_arabic(s):
            continue
        starts += s[0] in _SENTENCE_PUNCT
        ends += s[-1] in _SENTENCE_PUNCT
    return starts >= min_lines and starts > 2 * ends


# One unit that must survive reversal intact: a number or Latin run with its
# internal separators (2010-04-15, 3.5, 12/b) or a run of Arabic letters.
_UNIT = re.compile(
    _LTR_UNIT + r"(?:[.,:/%\-]" + _LTR_UNIT + r")*"
    r"|[\u0600-\u06FF]+"
    r"|\s+"
    r"|."
)
_MIRRORED = str.maketrans("()[]{}<>\u00AB\u00BB", ")(][}{><\u00BB\u00AB")
# After reversal, punctuation that belonged after a word can land before the
# next one ("1 .tathbut"); no Arabic typography puts it there, so move it back.
_DETACHED_PUNCT = re.compile("(?<=\\S) ([.:\u060C\u061B])(?=[^\\s\\d.:\u060C\u061B])")


def reverse_word_order(line: str) -> str:
    """Put a word-reversed Arabic line back into reading order.

    Words keep their letters; only their sequence is reversed. Numbers and
    dates are treated as single units, brackets are mirrored, and punctuation
    stranded on the wrong side of a space is reattached.
    """
    if not has_arabic(line):
        return line
    units = _UNIT.findall(line.strip())
    rebuilt = "".join(reversed(units)).translate(_MIRRORED)
    return _DETACHED_PUNCT.sub(r"\1 ", rebuilt)


# A bold word some PDFs draw twice comes out fused with itself ("al-aqar" x2
# with no space). Three-letter minimum halves: genuine Arabic reduplication
# (zalzala, waswasa) repeats only two letters, so it is never touched.
_DOUBLED_WORD = re.compile(r"(?<![\u0600-\u06FF])([\u0600-\u06FF]{3,})\1(?![\u0600-\u06FF])")


def undouble_words(text: str) -> str:
    """Collapse Arabic words fused with an identical copy of themselves."""
    return _DOUBLED_WORD.sub(r"\1", text)


# Jordanian (and most Arab) legislation names itself in Article 1:
# "yusamma hadha al-qanun / al-nizam / al-ta'limat ( <short title> )".
_NAMING_CLAUSE = re.compile(
    "\u064A\u0633\u0645\u0649\\s+\u0647\u0630\u0627\\s+"
    "(?:\u0627\u0644\u0642\u0627\u0646\u0648\u0646|\u0627\u0644\u0646\u0638\u0627\u0645"
    "|\u0627\u0644\u062A\u0639\u0644\u064A\u0645\u0627\u062A)"
    "\\s*\\(\\s*([^()\\n]{4,90}?)\\s*\\)"
)


def official_short_title(text: str, search_chars: int = 20000) -> str | None:
    """The instrument's own short title from its naming clause, if present.

    Far more reliable than PDF metadata, which export tools often fill with
    their own name -- every file from one legal database titled identically.
    """
    match = _NAMING_CLAUSE.search(text[:search_chars])
    return " ".join(match.group(1).split()) if match else None


_LAM_ALEF = frozenset({chr(0x0644), chr(0x0627)})


def merge_word_breaks(letters_source: str, breaks_source: str) -> str | None:
    """Combine one extraction's letters with another's word breaks.

    For visual-order PDFs the two pypdf modes are each half right: plain mode
    has every letter in reading order but no spaces; layout mode has the spaces
    but, after :func:`visual_to_logical`, every lam-alef ligature comes out
    swapped -- the ligature is one glyph that decodes to two letters already in
    reading order, so reversing the line flips it.

    Returns ``None`` unless the two agree letter-for-letter apart from those
    swaps, so a mismatch can never produce shifted, misaligned text.
    """
    letters = [c for c in letters_source if not c.isspace()]
    layout_letters = [c for c in breaks_source if not c.isspace()]
    if len(letters) != len(layout_letters):
        return None
    for a, b in zip(letters, layout_letters):
        if a != b and {a, b} != _LAM_ALEF:
            return None

    out: list[str] = []
    it = iter(letters)
    for ch in breaks_source:
        out.append(ch if ch.isspace() else next(it))
    return "".join(out)


def average_word_length(text: str) -> float:
    words = _ARABIC_WORD.findall(text)
    return sum(map(len, words)) / len(words) if words else 0.0


def spacing_looks_broken(text: str, min_words: int = 5, max_avg_len: float = 9.0) -> bool:
    """Detect extraction that has glued Arabic words together.

    Arabic words average roughly 4-6 letters. A page averaging well above that
    is almost certainly run-together text rather than unusually long words.
    """
    if len(_ARABIC_WORD.findall(text)) < min_words:
        return False
    return average_word_length(text) > max_avg_len
