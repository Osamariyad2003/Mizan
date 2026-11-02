"""Token-aware, structure-respecting chunking.

Text is first broken into *atoms* along the strongest available boundary
(blank line, then line, then sentence, then word). Atoms are packed greedily
into chunks up to ``chunk_tokens``, and each new chunk re-opens with the tail of
the previous one so a fact split across a boundary is still retrievable whole.

Every chunk keeps its ``(char_start, char_end)`` offsets into the parent
document, which is what lets a citation name a page and what lets the answer be
checked against the original file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, Iterable

from .config import Config, DEFAULT_CONFIG
from .legal import find_articles
from .loaders import Document

_MADDA = "\u0627\u0644\u0645\u0627\u062F\u0629"  # al-madda, "article"

_PARA_RE = re.compile(r"\n\s*\n")
_LINE_RE = re.compile(r"\n")
# Sentence end for Latin plus Arabic full stop / question mark.
_SENT_RE = re.compile(r"(?<=[.!?\u061F\u06D4])\s+")
_WORD_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _encoder():
    """Return a token encoder, or None if tiktoken is unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - optional dependency
        return None


def count_tokens(text: str) -> int:
    """Token count, with a ~4-chars-per-token fallback when tiktoken is absent.

    This is only used for sizing chunks, so an approximation is acceptable; the
    Anthropic token count of the final prompt is a separate concern.
    """
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    pages: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Context prepended when the chunk is embedded, keyword-indexed and shown to
    # the model -- e.g. "<law title> -- Article 61". A passage lifted out of a
    # 500-page law rarely names the law itself; the header puts that back.
    header: str = ""

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text

    @property
    def location(self) -> str:
        """Human-readable provenance: ``Labour Law -- Article 61 (p.22)``,
        ``report.pdf p.3`` or ``notes.md #2``."""
        name = self.title or self.source
        pages = ""
        if self.pages:
            pages = self.pages[0] if len(self.pages) == 1 else f"{self.pages[0]}-{self.pages[-1]}"
        article = self.metadata.get("article")
        if article:
            where = f"{name} \u2014 {_MADDA} {article}"
            return f"{where} (p.{pages})" if pages else where
        if pages:
            return f"{name} p.{pages}"
        return f"{name} #{self.ordinal + 1}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


def _split_offsets(text: str, start: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Split ``text`` on ``pattern``, returning absolute (start, end) spans."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            spans.append((start + cursor, start + match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((start + cursor, start + len(text)))
    return spans


def _atomize(text: str, max_tokens: int) -> list[tuple[int, int]]:
    """Break text into spans that each fit within ``max_tokens`` where possible."""
    levels = [_PARA_RE, _LINE_RE, _SENT_RE, _WORD_RE]

    def recurse(start: int, end: int, level: int) -> list[tuple[int, int]]:
        segment = text[start:end]
        if not segment.strip():
            return []
        if count_tokens(segment) <= max_tokens or level >= len(levels):
            if count_tokens(segment) <= max_tokens:
                return [(start, end)]
            # Nothing left to split on (e.g. one enormous token-dense blob):
            # cut on a character estimate so a chunk can never overflow.
            approx_chars = max(1, max_tokens * 3)
            return [
                (s, min(s + approx_chars, end))
                for s in range(start, end, approx_chars)
            ]
        pieces: list[tuple[int, int]] = []
        for s, e in _split_offsets(segment, start, levels[level]):
            pieces.extend(recurse(s, e, level + 1))
        return pieces

    return recurse(0, len(text), 0)


def chunk_document(doc: Document, config: Config = DEFAULT_CONFIG) -> list[Chunk]:
    """Split a document into offset-tracked chunks.

    Legislation is split one article per chunk (long articles are windowed
    within their own bounds); anything else gets overlapping token windows.
    """
    if not doc.text.strip():
        return []
    articles = find_articles(doc.text) if config.article_chunks else []
    if articles:
        return _article_chunks(doc, articles, config)
    header = doc.title if config.context_headers else ""
    return _window_chunks(doc, config, 0, len(doc.text), header=header)


def _article_chunks(doc: Document, articles: list, config: Config) -> list[Chunk]:
    """Chunks aligned to article boundaries.

    Consecutive short articles are packed together up to
    ``article_pack_tokens``; an article is never split across chunks unless it
    alone exceeds ``chunk_tokens``. Packing matters because many articles are a
    single sentence ("The King is the head of state"), and a chunk that small
    carries too little context to be matched by meaning. Each article keeps its
    own heading inside the chunk text, so attribution survives packing.
    """
    chunks: list[Chunk] = []
    fmt = doc.metadata.get("format", "txt")

    # Preamble before Article 1 (title page, enacting formula).
    if doc.text[: articles[0].start].strip():
        chunks.extend(
            _window_chunks(
                doc, config, 0, articles[0].start,
                header=doc.title if config.context_headers else "",
            )
        )

    def emit(group: list) -> None:
        first, last = group[0], group[-1]
        start, end = first.start, last.end
        body = doc.text[start:end]
        label = first.label if len(group) == 1 else f"{first.label}-{last.label}"
        topics = list(dict.fromkeys(a.topic for a in group if a.topic))[:3]
        parts = [doc.title, f"{_MADDA} {label}", " / ".join(topics)]
        header = " \u2014 ".join(p for p in parts if p) if config.context_headers else ""
        meta = {
            "format": fmt,
            "article": label,
            "article_number": first.number,
            "articles": [a.label for a in group],
            **({"topic": " / ".join(topics)} if topics else {}),
            **({"division": last.division} if last.division else {}),
            **({"effective": first.effective} if first.effective and len(group) == 1 else {}),
        }
        tokens = count_tokens(body)
        if tokens <= config.chunk_tokens:
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{ordinal:05d}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    ordinal=ordinal,
                    text=body.strip(),
                    char_start=start,
                    char_end=end,
                    token_count=tokens,
                    pages=doc.pages_for_range(start, end),
                    metadata=meta,
                    header=header,
                )
            )
        else:
            # One long article: overlapping windows, all carrying its header.
            chunks.extend(
                _window_chunks(
                    doc, config, start, end, header=header,
                    ordinal_offset=len(chunks), metadata=meta,
                )
            )

    group: list = []
    group_tokens = 0
    for art in articles:
        if not doc.text[art.start : art.end].strip():
            continue
        tokens = count_tokens(doc.text[art.start : art.end])
        if group and group_tokens + tokens > config.article_pack_tokens:
            emit(group)
            group, group_tokens = [], 0
        group.append(art)
        group_tokens += tokens
    if group:
        emit(group)
    return chunks


def _window_chunks(
    doc: Document,
    config: Config,
    span_start: int,
    span_end: int,
    header: str = "",
    ordinal_offset: int = 0,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Overlapping token windows over ``doc.text[span_start:span_end]``."""
    text = doc.text
    atoms = [
        (span_start + s, span_start + e)
        for s, e in _atomize(text[span_start:span_end], config.chunk_tokens)
    ]
    if not atoms:
        return []

    sized = [(s, e, count_tokens(text[s:e])) for s, e in atoms]
    chunks: list[Chunk] = []
    buffer: list[tuple[int, int, int]] = []
    buffer_tokens = 0

    def overlap_tail() -> tuple[list[tuple[int, int, int]], int]:
        """The tail of the current buffer to re-open the next chunk with."""
        tail: list[tuple[int, int, int]] = []
        tokens = 0
        for atom in reversed(buffer):
            if tokens + atom[2] > config.chunk_overlap_tokens:
                break
            tail.insert(0, atom)
            tokens += atom[2]
        if tail:
            return tail, tokens

        # No whole atom fits the overlap budget -- which is the common case for
        # dense prose, where one paragraph is larger than the overlap. Take a
        # partial tail of the last atom instead, otherwise overlap silently
        # becomes zero and facts on a chunk boundary get split in half.
        start, end, tokens_in_atom = buffer[-1]
        if tokens_in_atom <= 0:
            return [], 0
        chars_per_token = max(1.0, (end - start) / tokens_in_atom)
        tail_chars = int(config.chunk_overlap_tokens * chars_per_token)
        cut = max(start, end - tail_chars)
        if cut <= start:  # would duplicate the whole atom: no progress
            return [], 0
        # Prefer a word boundary so the overlap does not start mid-token.
        space = text.find(" ", cut, min(end, cut + 40))
        if space != -1:
            cut = space + 1
        if cut >= end:
            return [], 0
        return [(cut, end, count_tokens(text[cut:end]))], count_tokens(text[cut:end])

    def flush(final: bool = False) -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        start, end = buffer[0][0], buffer[-1][1]
        body = text[start:end].strip()

        # Too small to stand on its own: carry it into the next chunk rather
        # than emitting a fragment or dropping the content.
        if len(body) < config.min_chunk_chars and not final and chunks:
            return

        if body:
            ordinal = ordinal_offset + len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{ordinal:05d}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    ordinal=ordinal,
                    text=body,
                    char_start=start,
                    char_end=end,
                    token_count=buffer_tokens,
                    pages=doc.pages_for_range(start, end),
                    metadata=dict(metadata) if metadata else {"format": doc.metadata.get("format", "txt")},
                    header=header,
                )
            )
        if final:
            buffer, buffer_tokens = [], 0
        else:
            buffer, buffer_tokens = overlap_tail()

    for atom in sized:
        if buffer and buffer_tokens + atom[2] > config.chunk_tokens:
            flush()
        buffer.append(atom)
        buffer_tokens += atom[2]

    # Emit the remainder, unless the overlap tail is all that is left and the
    # previous chunk already covers it.
    if buffer:
        tail_end = buffer[-1][1]
        if not chunks or chunks[-1].char_end < tail_end:
            flush(final=True)

    return chunks


def chunk_documents(
    docs: Iterable[Document], config: Config = DEFAULT_CONFIG
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, config))
    return out


def content_hash(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
