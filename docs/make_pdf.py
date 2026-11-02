"""Generate the technical documentation PDF.

    python docs/make_pdf.py            # -> docs/RAG-system-explained.pdf

Numbers that describe the running system (chunk counts, file sizes, versions)
are read from a snapshot produced by docs/snapshot.py, so the document cannot
drift from the code without someone noticing.

Arabic needs two steps before it can go into a PDF: `arabic_reshaper` picks the
right contextual letter shapes, and the bidi algorithm reorders the string into
visual (right-to-left) order. ReportLab draws glyphs left to right and does
neither.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arabic_reshaper import reshape
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "RAG-system-explained.pdf"
SNAPSHOT = json.loads((HERE / "snapshot.json").read_text(encoding="utf-8"))

# --- palette ---------------------------------------------------------------
INK = colors.HexColor("#1b1a18")
MUTED = colors.HexColor("#6b6862")
ACCENT = colors.HexColor("#2f6f4f")
RULE = colors.HexColor("#d8d4cc")
PANEL = colors.HexColor("#f4f2ed")
CODEBG = colors.HexColor("#f7f6f2")

FONTS = Path("C:/Windows/Fonts")


def register_fonts() -> tuple[str, str, str]:
    """Body font must carry Arabic glyphs; code font only needs ASCII."""
    body, bold, mono = "Helvetica", "Helvetica-Bold", "Courier"
    candidates = [("Body", "tahoma.ttf"), ("Body-Bold", "tahomabd.ttf")]
    if all((FONTS / name).exists() for _, name in candidates):
        for face, name in candidates:
            pdfmetrics.registerFont(TTFont(face, str(FONTS / name)))
        # Without a family mapping, <b> inside a Paragraph is silently ignored
        # for TrueType fonts - the markup parses, and nothing changes.
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                      italic="Body", boldItalic="Body-Bold")
        body, bold = "Body", "Body-Bold"
    if (FONTS / "consola.ttf").exists():
        pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
        bold_mono = FONTS / "consolab.ttf"
        if bold_mono.exists():
            pdfmetrics.registerFont(TTFont("Mono-Bold", str(bold_mono)))
            pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold",
                                          italic="Mono", boldItalic="Mono-Bold")
        mono = "Mono"
    return body, bold, mono


BODY, BOLD, MONO = register_fonts()


def ar(text: str) -> str:
    """Arabic for the PDF: contextual shaping, then visual (RTL) order."""
    return get_display(reshape(text))


# --- styles ----------------------------------------------------------------
_base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=27, leading=32, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=13, leading=18, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=16, leading=20, textColor=ACCENT,
                         spaceBefore=18, spaceAfter=7, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=11, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.6, leading=14.2, textColor=INK,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=9.6, leading=14, textColor=INK,
                             leftIndent=11, bulletIndent=2, spaceAfter=3),
    "small": ParagraphStyle("small", fontName=BODY, fontSize=8.4, leading=12, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8.6, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8.6, leading=12, textColor=INK),
    "cellar": ParagraphStyle("cellar", fontName=BODY, fontSize=10, leading=15, textColor=INK,
                             alignment=TA_RIGHT),
    "caption": ParagraphStyle("caption", fontName=BODY, fontSize=8.2, leading=11, textColor=MUTED,
                              spaceBefore=3, spaceAfter=9),
    "toc": ParagraphStyle("toc", fontName=BODY, fontSize=10, leading=17, textColor=INK),
    "cover_fact": ParagraphStyle("cover_fact", fontName=BODY, fontSize=9.5, leading=15, textColor=INK),
}
CODE = ParagraphStyle("code", fontName=MONO, fontSize=7.7, leading=10.2, textColor=INK,
                      backColor=CODEBG, borderPadding=6, spaceBefore=4, spaceAfter=8)
DIAGRAM = ParagraphStyle("diagram", fontName=MONO, fontSize=7.0, leading=9.4, textColor=INK,
                         spaceBefore=4, spaceAfter=6)


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list:
    return [Paragraph(t, S["bullet"], bulletText="\u2022") for t in items]


def code(text: str, style: ParagraphStyle = CODE) -> Preformatted:
    return Preformatted(text.strip("\n"), style)


def table(rows: list[list], widths: list[float], header: bool = True,
          align_right_first: bool = False) -> Table:
    data = []
    for r, row in enumerate(rows):
        cells = []
        for value in row:
            if isinstance(value, Paragraph):
                cells.append(value)
            else:
                style = "cellb" if (header and r == 0) else "cell"
                cells.append(Paragraph(str(value), S[style]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), PANEL),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(style))
    return t


def panel(text: str) -> Table:
    """A callout box for the things most worth remembering."""
    t = Table([[Paragraph(text, S["cell"])]], colWidths=[165 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


class Doc(BaseDocTemplate):
    """Two templates: a bare cover, then pages with a rule and page numbers."""

    def __init__(self, path: str):
        super().__init__(path, pagesize=A4, title="RAG system - technical documentation",
                         author="Generated by docs/make_pdf.py", leftMargin=22 * mm,
                         rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=18 * mm)
        self.headings: list[tuple[str, int]] = []
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame]),
            PageTemplate(id="main", frames=[frame], onPage=self._chrome),
        ])

    def afterFlowable(self, flowable):
        """Record where each section heading landed, for the contents page."""
        if isinstance(flowable, Paragraph) and flowable.style.name == "h1":
            self.headings.append((flowable.getPlainText(), self.page))

    def _chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY, 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, A4[1] - 13 * mm,
                          "RAG system over Jordanian legislation - technical documentation")
        canvas.drawRightString(A4[0] - self.rightMargin, 11 * mm, str(canvas.getPageNumber()))
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(self.leftMargin, A4[1] - 15 * mm, A4[0] - self.rightMargin, A4[1] - 15 * mm)
        canvas.restoreState()


def build(story_factory) -> None:
    """Build twice: once to learn the page numbers, once to print them.

    A contents page cannot know where sections land until the document has been
    laid out, so the first pass is thrown away apart from its headings.
    """
    import tempfile

    scratch = Path(tempfile.gettempdir()) / "rag-doc-pass1.pdf"
    first = Doc(str(scratch))
    first.build(story_factory(None))

    pages = {title: page for title, page in first.headings}
    Doc(str(OUT)).build(story_factory(pages))
    scratch.unlink(missing_ok=True)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    from content import story  # noqa: E402  (content.py builds the flowables)

    sys.exit(build(story))
