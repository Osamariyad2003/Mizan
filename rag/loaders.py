"""Document loaders.

Each loader returns a :class:`Document` whose ``text`` is the full extracted
plain text plus ``page_spans`` -- (char_start, char_end, page_label) triples that
let a chunk report which page it came from, which is what makes PDF citations
useful instead of merely decorative.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .arabic import (
    average_word_length,
    fold_presentation_forms,
    merge_word_breaks,
    official_short_title,
    reverse_word_order,
    spacing_looks_broken,
    undouble_words,
    visual_to_logical,
    word_order_reversed,
)

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".sh", ".sql", ".yaml", ".yml", ".toml", ".ini",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {
    ".pdf", ".docx", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm",
}

# Directories that are never worth indexing even when file extensions match.
SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".rag_index", "catboost_info", ".idea",
}
MAX_FILE_BYTES = 32 * 1024 * 1024

# pypdf logs a warning for every malformed cross-reference entry it recovers
# from ("Ignoring wrong pointing object"); common in exported PDFs, harmless,
# and it buries the ingest output.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)
    # (char_start, char_end, page_label); empty for single-page formats.
    page_spans: list[tuple[int, int, str]] = field(default_factory=list)

    def page_for_offset(self, offset: int) -> str | None:
        for start, end, label in self.page_spans:
            if start <= offset < end:
                return label
        return None

    def pages_for_range(self, start: int, end: int) -> list[str]:
        hits = [
            label
            for p_start, p_end, label in self.page_spans
            if p_start < end and start < p_end
        ]
        seen: set[str] = set()
        return [p for p in hits if not (p in seen or seen.add(p))]


def doc_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _normalize(text: str) -> str:
    """Collapse the whitespace damage typical of PDF and HTML extraction."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ").replace("\uFEFF", "")
    # Rejoin words hyphenated across a line break: "inter-\nnational".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Per-format loaders
# --------------------------------------------------------------------------

def load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        doc_id=doc_id_for(path),
        source=str(path),
        title=path.name,
        text=_normalize(raw),
        metadata={"format": path.suffix.lstrip(".") or "txt"},
    )


def load_pdf(path: Path) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Reading PDFs requires: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    repaired_pages: list[int] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:  # a single malformed page must not kill the document
            page_text = ""
        # Glyph codes -> real letters first: every check below measures words.
        page_text = _strip_print_artifacts(fold_presentation_forms(page_text))
        repaired = _repair_arabic_spacing(page, page_text)
        if repaired is not None:
            page_text = repaired
            repaired_pages.append(page_no)
        pages.append((page_no, page_text))

    # Decided once per document: a file comes from one producer, and a single
    # page can hold too few sentences to judge on its own.
    reversed_words = word_order_reversed("\n".join(text for _, text in pages))

    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for page_no, page_text in pages:
        if reversed_words:
            plain_lines = [reverse_word_order(line) for line in page_text.splitlines()]
            layout_lines = _layout_lines(reader.pages[page_no - 1])
            page_text = "\n".join(
                _order_like_layout(layout_lines, plain_lines) if layout_lines else plain_lines
            )
        page_text = _normalize(undouble_words(page_text))
        if not page_text:
            continue
        block = page_text + "\n\n"
        parts.append(block)
        spans.append((cursor, cursor + len(block), str(page_no)))
        cursor += len(block)

    info = getattr(reader, "metadata", None)
    title = (getattr(info, "title", None) or "").strip() if info else ""
    metadata: dict = {"format": "pdf", "pages": len(reader.pages)}
    if repaired_pages:
        metadata["arabic_spacing_repaired_pages"] = repaired_pages
    if reversed_words:
        metadata["arabic_word_order_repaired"] = True
    return Document(
        doc_id=doc_id_for(path),
        source=str(path),
        title=title or path.name,
        text="".join(parts).strip(),
        metadata=metadata,
        page_spans=spans,
    )


def _layout_lines(page) -> list[str]:
    """A page's lines in visual top-to-bottom order, each in reading order."""
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return []
    # Reverse *before* folding: a lam-alef ligature is one glyph code here and
    # must stay one character while the line is reversed.
    lines = (fold_presentation_forms(visual_to_logical(l.strip())) for l in layout.splitlines())
    return [l for l in _strip_print_artifacts("\n".join(lines)).splitlines() if l.strip()]


_WORDS = re.compile(r"[\w\u0600-\u06FF]+")


def _order_like_layout(layout_lines: list[str], plain_lines: list[str]) -> list[str]:
    """Take line *order* from layout extraction and line *text* from plain.

    In some legal-database PDFs, plain extraction emits every article heading
    on a page first and every article body after them, so no body can be tied
    to its article number. Layout extraction keeps each heading directly above
    its own text, but decodes a few broken glyphs into longer junk than plain
    extraction does. Each layout line is therefore swapped for its plain twin
    (matched by shared words) wherever one exists.

    Heading lines stay in their layout form, whose word order the article
    splitter relies on.
    """
    pool = [(set(_WORDS.findall(line)), line) for line in plain_lines if line.strip()]
    used = [False] * len(pool)
    out: list[str] = []
    for line in layout_lines:
        if _LAYOUT_HEADING.match(line):
            out.append(line)
            continue
        words = set(_WORDS.findall(line))
        best, best_score = -1, 0.0
        for i, (plain_words, _) in enumerate(pool):
            if used[i] or not words or not plain_words:
                continue
            score = len(words & plain_words) / len(words | plain_words)
            if score > best_score:
                best, best_score = i, score
        if best >= 0 and best_score >= 0.6:
            used[best] = True
            out.append(pool[best][1])
        else:
            out.append(line)
    return out


# "- al-madda N ..." as legal-database exports print an article heading.
_LAYOUT_HEADING = re.compile(r"^-\s*\u0627\u0644\u0645\u0627\u062F\u0629\s+\d+")


# Browser print footers ("https://site/print/...?a=1&b=2" plus a timestamp) and
# "page N of M" counters. They repeat on every page, so they add nothing a
# reader would search for while diluting every chunk they land in.
_URL_QUERY_NOISE = re.compile(r"https?|://|//:|www\.|\.com")
_PAGE_COUNTER = re.compile(
    r"^.{0,12}?[\d\u0660-\u0669]{1,4}\s+(?:\u0645\u0646|of)\s+[\d\u0660-\u0669]{1,4}\s*$",
    re.IGNORECASE,
)


def _strip_print_artifacts(text: str) -> str:
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if _URL_QUERY_NOISE.search(s) and s.count("=") + s.count("&") >= 4:
            continue  # print-footer URL with its query string
        if len(s) <= 25 and _PAGE_COUNTER.match(s):
            continue
        kept.append(line)
    return "\n".join(kept)


def _repair_arabic_spacing(page, plain_text: str) -> str | None:
    """Re-extract a page whose Arabic words came out glued together.

    Some PDFs store Arabic glyphs in display order. Plain extraction then drops
    the spaces between words; layout extraction keeps them but emits every line
    reversed. Reverse the layout lines back into reading order, then take the
    *letters* from plain mode and only the *word breaks* from layout, which
    also undoes the lam-alef ligatures the line reversal flipped.

    Only used when plain extraction is measurably broken and the repair is a
    clear improvement -- a well-formed page is never touched.
    """
    if not spacing_looks_broken(plain_text):
        return None
    try:
        layout = fold_presentation_forms(page.extract_text(extraction_mode="layout") or "")
    except Exception:
        return None
    candidate = "\n".join(visual_to_logical(line.strip()) for line in layout.splitlines())
    if average_word_length(candidate) >= average_word_length(plain_text) * 0.75:
        return None
    # Prefer the exact merge; if the extractions disagree beyond ligature
    # swaps, the reversed layout text is still far better than glued words.
    return merge_word_breaks(plain_text, candidate) or candidate


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DC = "{http://purl.org/dc/elements/1.1/}"
# A paragraph this short with no sentence punctuation is treated as a heading
# ("Article 5", "Chapter Two") and kept on the same chunk as the text under it.
_HEADING_MAX_CHARS = 40
_SENTENCE_END = (".", ":", "\u061F", "?", "!", "\u060C", "\u061B")


def _docx_paragraph_text(p) -> str:
    out: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == _W + "t":
            out.append(node.text or "")
        elif tag == _W + "tab":
            out.append("\t")
        elif tag in (_W + "br", _W + "cr"):
            out.append("\n")
    return "".join(out).strip()


def _docx_blocks(element):
    """Yield paragraph and table text in document order.

    Descends through content controls and similar wrappers, but treats a table
    as one unit so its rows are not read as unrelated paragraphs.
    """
    for child in element:
        tag = child.tag
        if tag == _W + "p":
            text = _docx_paragraph_text(child)
            if text:
                yield "p", text
        elif tag == _W + "tbl":
            rows = []
            for tr in child.iter(_W + "tr"):
                cells = [
                    " ".join(filter(None, (_docx_paragraph_text(p) for p in tc.iter(_W + "p"))))
                    for tc in tr.iter(_W + "tc")
                ]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                yield "tbl", "\n".join(rows)
        elif tag != _W + "sectPr":
            yield from _docx_blocks(child)


def load_docx(path: Path) -> Document:
    """Word documents, read directly from the .docx XML -- no extra dependency.

    Word stores text in reading order, so none of the PDF repairs are needed.
    Footnotes and endnotes are appended, since in legal and academic documents
    they often carry the amendment history or the source.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "Not a valid .docx file. If it is an old binary .doc, open it in Word "
            "and save it as .docx."
        ) from exc

    with archive:
        body = ET.fromstring(archive.read("word/document.xml")).find(_W + "body")
        blocks = list(_docx_blocks(body)) if body is not None else []

        notes: list[str] = []
        for part in ("word/footnotes.xml", "word/endnotes.xml"):
            if part in archive.namelist():
                root = ET.fromstring(archive.read(part))
                for note in root:
                    if note.get(_W + "type") in ("separator", "continuationSeparator"):
                        continue
                    text = " ".join(filter(None, (_docx_paragraph_text(p) for p in note.iter(_W + "p"))))
                    if text:
                        notes.append(text)

        title = ""
        if "docProps/core.xml" in archive.namelist():
            node = ET.fromstring(archive.read("docProps/core.xml")).find(_DC + "title")
            title = (node.text or "").strip() if node is not None else ""

    # Blank line between blocks (the chunker's strongest boundary), but a
    # heading is glued to the block after it with a single newline.
    parts: list[str] = []
    for kind, text in blocks:
        is_heading = (
            kind == "p"
            and len(text) <= _HEADING_MAX_CHARS
            and not text.endswith(_SENTENCE_END)
        )
        parts.append(text + ("\n" if is_heading else "\n\n"))
    text = "".join(parts)
    if notes:
        text += "\n\nNotes:\n\n" + "\n\n".join(notes)

    return Document(
        doc_id=doc_id_for(path),
        source=str(path),
        title=title or path.name,
        text=_normalize(text),
        metadata={"format": "docx", "paragraphs": len(blocks), "notes": len(notes)},
    )


def load_csv(path: Path, max_rows: int = 2000) -> Document:
    """Render tabular data as one "col: value" record per block.

    Row-per-block keeps a record intact inside a single chunk, which matters far
    more for retrieval than preserving the visual table layout.
    """
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    blocks: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            fields = [f"{k}: {v}" for k, v in row.items() if k and v not in (None, "")]
            if fields:
                blocks.append(f"[row {i + 1}] " + " | ".join(fields))
            total = i + 1
    header = f"Tabular file {path.name} ({total} rows indexed)"
    return Document(
        doc_id=doc_id_for(path),
        source=str(path),
        title=path.name,
        text=header + "\n\n" + "\n\n".join(blocks),
        metadata={
            "format": "csv",
            "rows_indexed": total,
            "truncated": total >= max_rows,
        },
    )


def load_json(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        blocks = []
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rendered = json.dumps(json.loads(line), ensure_ascii=False)
            except json.JSONDecodeError:
                rendered = line
            blocks.append(f"[record {i + 1}] {rendered}")
        text = "\n\n".join(blocks)
    else:
        try:
            text = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = raw
    return Document(
        doc_id=doc_id_for(path),
        source=str(path),
        title=path.name,
        text=_normalize(text),
        metadata={"format": path.suffix.lstrip(".")},
    )


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": chr(34), "&#39;": chr(39), "&apos;": chr(39),
}


def load_html(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title_match = _TITLE_RE.search(raw)
    body = _SCRIPT_RE.sub(" ", raw)
    body = re.sub(r"</(p|div|h[1-6]|li|tr|section)>", "\n", body, flags=re.IGNORECASE)
    body = _TAG_RE.sub(" ", body)
    for entity, char in _HTML_ENTITIES.items():
        body = body.replace(entity, char)
    return Document(
        doc_id=doc_id_for(path),
        source=str(path),
        title=(title_match.group(1).strip() if title_match else path.name),
        text=_normalize(body),
        metadata={"format": "html"},
    )


LOADERS: dict[str, Callable[[Path], Document]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".csv": load_csv,
    ".tsv": load_csv,
    ".json": load_json,
    ".jsonl": load_json,
    ".html": load_html,
    ".htm": load_html,
}


def _best_title(doc: Document, path: Path) -> str:
    """Pick the title that citations will show.

    1. A legal instrument's own naming clause, if present.
    2. Embedded metadata, but only if it says something: export tools often
       stamp every file with their own one-word name.
    3. The filename, which the user controls.
    """
    official = official_short_title(doc.text)
    if official:
        return official
    if doc.title and doc.title != path.name and len(doc.title.split()) >= 3:
        return doc.title
    return path.stem.replace("_", " ").strip()


# Bump whenever extraction output changes, so cached text from an older loader
# is never reused.
EXTRACTOR_VERSION = 4


def _extract(path: Path, cache_dir: Path | None) -> Document:
    """Run the format's loader, reusing a cached result for identical bytes.

    Layout-mode PDF extraction dominates ingest time, and its output depends
    only on the file's bytes -- so re-ingesting after a chunking change, or
    ingesting the same file into a second index, need not repeat it.
    """
    loader = LOADERS.get(path.suffix.lower(), load_text)
    if cache_dir is None:
        return loader(path)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    cached = cache_dir / f"{digest}-v{EXTRACTOR_VERSION}.json"
    if cached.exists():
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
            data["page_spans"] = [tuple(span) for span in data.get("page_spans", [])]
            doc = Document(**data)
            # Identity comes from where the file is now, not where it was: the
            # same bytes may have been cached under another name.
            if doc.title == Path(doc.source).name:
                doc.title = path.name
            doc.doc_id, doc.source = doc_id_for(path), str(path)
            return doc
        except (OSError, ValueError, TypeError):
            pass  # unreadable cache entry: fall through and re-extract

    doc = loader(path)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc.__dict__, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cached)
    except OSError:
        pass  # caching is an optimisation; never fail an ingest over it
    return doc


def load_document(path: str | Path, cache_dir: str | Path | None = None) -> Document:
    path = Path(path)
    doc = _extract(path, Path(cache_dir) if cache_dir else None)
    if doc.metadata.get("format") in ("pdf", "docx"):
        doc.title = _best_title(doc, path)
    stat = path.stat()
    doc.metadata.setdefault("bytes", stat.st_size)
    doc.metadata.setdefault("mtime", stat.st_mtime)
    doc.metadata["content_hash"] = hashlib.sha256(
        doc.text.encode("utf-8")
    ).hexdigest()[:16]
    return doc


def discover(paths: Iterable[str | Path], recursive: bool = True) -> Iterator[Path]:
    """Yield indexable files from a mix of file and directory paths."""
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
            continue
        if not p.is_dir():
            continue
        walker = p.rglob("*") if recursive else p.glob("*")
        for child in walker:
            if not child.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in child.parts):
                continue
            if child.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if child.name.startswith("~$"):  # Word/Excel lock file, not a document
                continue
            try:
                if child.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield child
