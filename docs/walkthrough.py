"""Framework for the line-by-line code walkthrough PDF.

Each module is described as a list of blocks:

    (first_line, last_line, "what these lines do")

The renderer prints the real source with line numbers, so the document cannot
quote code that does not exist, and :func:`check_coverage` refuses to build
unless the blocks tile the file exactly - no gaps, no overlaps, first block
starting at line 1 and last ending at the final line. That is what makes the
claim "every line is explained" checkable rather than aspirational.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepTogether, PageBreak, Preformatted

from make_pdf import BODY, BOLD, MONO, S, p

ROOT = Path(__file__).resolve().parent.parent
MAX_CHARS = 112  # columns that fit the text width at this font size

LISTING = ParagraphStyle(
    "listing", fontName=MONO, fontSize=6.9, leading=8.7,
    textColor=colors.HexColor("#22211f"), backColor=colors.HexColor("#f7f6f2"),
    borderPadding=5, spaceBefore=3, spaceAfter=3,
)
NOTE = ParagraphStyle(
    "note", fontName=BODY, fontSize=9.1, leading=13, textColor=colors.HexColor("#1b1a18"),
    leftIndent=4, spaceAfter=9,
)
MODULE_TITLE = ParagraphStyle(
    "module", fontName=BOLD, fontSize=15, leading=19,
    textColor=colors.HexColor("#2f6f4f"), spaceBefore=4, spaceAfter=3, keepWithNext=1,
)


def source_lines(module: str) -> list[str]:
    return (ROOT / module).read_text(encoding="utf-8").splitlines()


def check_coverage(module: str, blocks: list[tuple[int, int, str]]) -> None:
    """Fail loudly if the annotations do not cover the file exactly."""
    lines = source_lines(module)
    expected = 1
    for first, last, _ in blocks:
        if first != expected:
            raise SystemExit(
                f"{module}: line {expected} is not covered (next block starts at {first}). "
                "Did the file change after the notes were written?"
            )
        if last < first:
            raise SystemExit(f"{module}: block {first}-{last} is backwards")
        expected = last + 1
    if expected - 1 != len(lines):
        raise SystemExit(
            f"{module}: notes cover {expected - 1} lines but the file has {len(lines)}"
        )


def _listing(lines: list[str], first: int) -> Preformatted:
    """Source with line numbers, wrapping anything too wide to fit."""
    out = []
    for offset, line in enumerate(lines):
        number = f"{first + offset:3d}  "
        text = line.replace("\t", "    ")
        room = MAX_CHARS - len(number)
        if len(text) <= room:
            out.append(number + text)
            continue
        indent = len(text) - len(text.lstrip())
        out.append(number + text[:room])
        rest = text[room:]
        while rest:
            out.append(" " * (len(number) + indent + 4) + rest[:room - indent - 4])
            rest = rest[room - indent - 4:]
    return Preformatted("\n".join(out), LISTING)


def module_section(module: str, summary: str, blocks: list[tuple[int, int, str]]) -> list:
    """One module: a heading, a one-paragraph summary, then annotated blocks."""
    check_coverage(module, blocks)
    lines = source_lines(module)
    flow: list = [
        KeepTogether([p(module, "h1"), p(f"{len(lines)} lines. {summary}", "body")])
    ]
    for first, last, note in blocks:
        chunk = lines[first - 1:last]
        flow.append(KeepTogether([_listing(chunk, first), p(note, "body")])
                    if len(chunk) <= 26 else _listing(chunk, first))
        if len(chunk) > 26:
            flow.append(p(note, "body"))
    flow.append(PageBreak())
    return flow
