"""Article structure of legislation.

Laws are cited and asked about by article ("what does Article 61 of the Labour
Law say?"), so the article -- not the page, and not an arbitrary 500-token
window -- is the natural unit for retrieval and for citations. This module
finds article boundaries in extracted text.

Three heading styles are recognised, all common in Arab legislation:

* ``- <al-madda> 61 <title> <effective-date label> 2019-05-16`` -- legal-database
  exports, where the heading line also carries a topic and an effective date.
* ``<al-madda> 7`` alone on a line -- Word and most official PDFs.
* ``<al-madda> 4- <text...>`` -- heading and first sentence on one line.

A line that merely *mentions* an article ("under Article 70 of this law") must
not start a new one. The guard is numbering: articles run in increasing order,
so a candidate is accepted only if it continues the sequence (a small forward
jump is allowed for repealed articles, and a restart at 1 for an annexed
regulation). Mid-text references almost never happen to continue it.

Code points are written as escapes (see arabic.py for why).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MADDA = "\u0627\u0644\u0645\u0627\u062F\u0629"  # al-madda, "article"
_BIS = "\u0645\u0643\u0631\u0631[\u0629\u0647]?"  # mukarrar(a), "bis"
_EFFECTIVE = "\u062A\u0627\u0631\u064A\u062E \u0627\u0644\u0633\u0631\u064A\u0627\u0646"  # effective date
_FROM = "\u0645\u0646"  # "min", as in "Article 70 *of* this law"

# "- al-madda 61 <topic> <effective> 2019-05-16"
_DB_HEADING = re.compile(
    rf"^-\s*{_MADDA}\s+(\d+)\s*((?:{_BIS})?)\s*(.*?)\s*(?:{_EFFECTIVE}\s+(\d{{4}}-\d{{2}}-\d{{2}}))?\s*$"
)
# "al-madda 7" / "al-madda (7)" alone, optionally "bis" and trailing punctuation
_BARE_HEADING = re.compile(
    rf"^{_MADDA}\s*\(?\s*(\d+)\s*\)?\s*((?:{_BIS})?)\s*[-\u2013:.]?\s*$"
)
# "al-madda 4- first sentence..." (dash or colon right after the number)
_INLINE_HEADING = re.compile(
    rf"^{_MADDA}\s*\(?\s*(\d+)\s*\)?\s*((?:{_BIS})?)\s*[-\u2013:]\s*(\S.*)$"
)
# Part / chapter / section lines that give an article its context.
_DIVISION = re.compile(
    "^(?:\u0627\u0644\u0643\u062A\u0627\u0628|\u0627\u0644\u0628\u0627\u0628"
    "|\u0627\u0644\u0641\u0635\u0644|\u0627\u0644\u0641\u0631\u0639|\u0627\u0644\u0642\u0633\u0645"
    "|\u0627\u0644\u062C\u0632\u0621)\\s+\\S"
)

MAX_FORWARD_JUMP = 50  # repealed or renumbered articles leave gaps
MIN_ARTICLES = 5       # fewer than this and the text is not treated as legislation


@dataclass
class Article:
    number: int
    label: str          # e.g. "61" or "42 bis" in the document's own wording
    start: int          # offset of the heading line
    body_start: int     # offset just after the heading (and its division line)
    end: int            # offset where the next article (or the text) begins
    topic: str = ""     # the database heading's topic, if any
    division: str = ""  # enclosing part / chapter, if known
    effective: str = ""  # effective date from the heading, if any

    @property
    def citation(self) -> str:
        return f"{_MADDA} {self.label}"


def _accept(number: int, bis: bool, previous: int | None) -> bool:
    if previous is None or number == 1:
        return True
    if bis and number == previous:
        return True
    return previous < number <= previous + MAX_FORWARD_JUMP


def find_articles(text: str) -> list[Article]:
    """Locate article headings; empty if the text does not look like a law."""
    articles: list[Article] = []
    lines = text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    division = ""
    previous: int | None = None
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        match_db = _DB_HEADING.match(line)
        match_bare = None if match_db else _BARE_HEADING.match(line)
        match_inline = None if (match_db or match_bare) else _INLINE_HEADING.match(line)
        match = match_db or match_bare or match_inline

        if match_inline and match_inline.group(3).startswith(_FROM + " "):
            match = None  # "Article 70 of this law ..." -- a reference, not a heading

        if match:
            number = int(match.group(1))
            bis = bool(match.group(2))
            if _accept(number, bis, previous):
                start = offsets[i] + (len(raw) - len(raw.lstrip()))
                body_start = offsets[i] + len(raw) + 1
                topic = match_db.group(3).strip() if match_db else ""
                effective = (match_db.group(4) or "") if match_db else ""
                if match_inline:
                    # Body begins on the heading line itself, after the dash.
                    body_start = offsets[i] + raw.index(match_inline.group(3))
                # A database heading is followed by its part name on one line.
                if match_db and i + 1 < len(lines) and _DIVISION.match(lines[i + 1].strip()):
                    division = lines[i + 1].strip()
                    body_start = offsets[i + 1] + len(lines[i + 1]) + 1
                    i += 1
                label = str(number) + (" " + match.group(2) if bis else "")
                articles.append(
                    Article(
                        number=number, label=label, start=start,
                        body_start=min(body_start, len(text)), end=len(text),
                        topic=topic, division=division, effective=effective,
                    )
                )
                previous = number
                i += 1
                continue

        if _DIVISION.match(line) and len(line) <= 80:
            # "Part Four" -- often followed by its title on the next line.
            title = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if title and len(title) <= 60 and not _BARE_HEADING.match(title) and not _DB_HEADING.match(title):
                division = f"{line}: {title}"
            else:
                division = line
        i += 1

    for current, following in zip(articles, articles[1:]):
        current.end = following.start
    if len(articles) < MIN_ARTICLES:
        return []
    return articles
