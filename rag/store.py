"""Persistent hybrid index: dense vectors + BM25 + chunk metadata.

Layout on disk (``.rag_index/`` by default)::

    chunks.jsonl     one chunk per line, with offsets and provenance
    embeddings.npy   float32 [n_chunks, dim], L2-normalised, row i <-> line i
    meta.json        embedder identity, chunking fingerprint, per-source state

Deliberately file-based. A dedicated vector database earns its operational cost
somewhere north of a million chunks; below that, an in-memory matrix and an
exact dot product are faster than an approximate index, exactly accurate, and
trivial to back up or diff. The retrieval interface is narrow enough that
swapping in FAISS or pgvector later means reimplementing
:meth:`VectorStore.search_dense` and nothing else.

The matrix is read into RAM rather than memory-mapped. Every query multiplies
the whole matrix, so a mapping saves no I/O -- it only adds a file handle that
Windows then refuses to let the next atomic save replace.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk, chunk_document
from .config import Config, DEFAULT_CONFIG
from .loaders import Document

log = logging.getLogger("rag.store")

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"


@dataclass
class IngestReport:
    added_docs: int = 0
    skipped_docs: int = 0
    updated_docs: int = 0
    added_chunks: int = 0
    removed_chunks: int = 0
    failures: list[tuple[str, str]] = None  # type: ignore[assignment]
    duplicates: list[tuple[str, str]] = None  # type: ignore[assignment]  (skipped, kept)
    pruned: list[str] = None  # type: ignore[assignment]  sources whose file is gone

    def __post_init__(self) -> None:
        if self.pruned is None:
            self.pruned = []
        if self.failures is None:
            self.failures = []
        if self.duplicates is None:
            self.duplicates = []

    def summary(self) -> str:
        parts = [
            f"{self.added_docs} new",
            f"{self.updated_docs} updated",
            f"{self.skipped_docs} unchanged",
            f"+{self.added_chunks} chunks",
        ]
        if self.removed_chunks:
            parts.append(f"-{self.removed_chunks} stale chunks")
        if self.pruned:
            parts.append(f"{len(self.pruned)} deleted file(s) removed")
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} duplicate(s) skipped")
        if self.failures:
            parts.append(f"{len(self.failures)} failed")
        return ", ".join(parts)


class VectorStore:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.dir = Path(config.index_dir)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        # source path -> {content_hash, mtime, title, chunks}
        self.sources: dict[str, dict[str, Any]] = {}
        self.embedder_name: str | None = None
        self.fingerprint: dict[str, Any] = {}
        self._bm25: BM25Index | None = None
        self._vectors_dirty = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            # Rebuilt from chunk text rather than persisted: it costs well under
            # a second for tens of thousands of chunks and can never go stale
            # against the chunks actually in the index.
            self._bm25 = BM25Index([c.indexed_text for c in self.chunks])
        return self._bm25

    def _invalidate(self) -> None:
        self._bm25 = None

    @classmethod
    def load(cls, config: Config = DEFAULT_CONFIG) -> "VectorStore":
        store = cls(config)
        meta_path = store.dir / META_FILE
        if not meta_path.exists():
            return store

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store.sources = meta.get("sources", {})
        store.embedder_name = meta.get("embedder")
        store.fingerprint = meta.get("fingerprint", {})

        chunks_path = store.dir / CHUNKS_FILE
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as fh:
                store.chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]

        emb_path = store.dir / EMBEDDINGS_FILE
        if emb_path.exists():
            # Read fully into RAM: see the module docstring on why not mmap.
            embeddings = np.load(emb_path)
            if embeddings.shape[0] != len(store.chunks):
                # A crash between the two writes; keyword search still works,
                # and the next ingest rebuilds the vectors.
                embeddings = None
            store.embeddings = embeddings
        return store

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_chunks = self.dir / (CHUNKS_FILE + ".tmp")
        with tmp_chunks.open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        tmp_chunks.replace(self.dir / CHUNKS_FILE)

        emb_path = self.dir / EMBEDDINGS_FILE
        if self.embeddings is not None and len(self.chunks):
            # Rewrite only when the vectors actually changed: re-running ingest
            # on an unchanged corpus should not rewrite hundreds of megabytes.
            if self._vectors_dirty or not emb_path.exists():
                # Must keep the .npy suffix: np.save appends one otherwise.
                tmp_emb = self.dir / "embeddings.tmp.npy"
                np.save(tmp_emb, np.asarray(self.embeddings, dtype=np.float32))
                tmp_emb.replace(emb_path)
        elif emb_path.exists():
            emb_path.unlink()  # dense vectors were dropped; do not leave a stale file
        self._vectors_dirty = False

        meta = {
            "embedder": self.embedder_name,
            "fingerprint": self.fingerprint or self.config.index_fingerprint(),
            "chunk_count": len(self.chunks),
            "dim": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
            "sources": self.sources,
        }
        tmp_meta = self.dir / (META_FILE + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_meta.replace(self.dir / META_FILE)

    def clear(self) -> None:
        self.embeddings = None
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.chunks = []
        self.sources = {}
        self._vectors_dirty = False
        self._invalidate()

    # -- mutation ----------------------------------------------------------

    def remove_source(self, source: str) -> int:
        """Drop every chunk belonging to ``source``. Returns chunks removed."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            if self.embeddings is not None and len(self.embeddings):
                self.embeddings = np.asarray(self.embeddings)[keep]
                self._vectors_dirty = True
            self._invalidate()
        self.sources.pop(source, None)
        return removed

    def needs_reindex(self, doc: Document) -> bool:
        known = self.sources.get(doc.source)
        if known is None:
            return True
        return known.get("content_hash") != doc.metadata.get("content_hash")

    def add_documents(
        self,
        docs: Iterable[Document],
        embedder: Any | None = None,
        force: bool = False,
        progress: Any | None = None,
    ) -> IngestReport:
        """Chunk, embed and index documents, skipping unchanged sources."""
        report = IngestReport()
        pending: list[Chunk] = []

        # Identical content under another filename (a file downloaded twice) is
        # skipped: indexing it would return every passage twice, spending half
        # of top_k on repeats and citing two "sources" that are one document.
        seen_content = {
            meta["content_hash"]: source
            for source, meta in self.sources.items()
            if meta.get("content_hash")
        }

        for doc in docs:
            content = doc.metadata.get("content_hash")
            twin = seen_content.get(content) if content else None
            if twin and twin != doc.source:
                report.duplicates.append((doc.source, twin))
                if progress:
                    progress(doc.source, f"identical to {Path(twin).name} - skipped")
                continue

            if not force and not self.needs_reindex(doc):
                report.skipped_docs += 1
                if progress:
                    progress(doc.source, "unchanged")
                continue

            is_update = doc.source in self.sources
            if is_update:
                report.removed_chunks += self.remove_source(doc.source)
                report.updated_docs += 1
            else:
                report.added_docs += 1

            doc_chunks = chunk_document(doc, self.config)
            pending.extend(doc_chunks)
            if content:
                seen_content[content] = doc.source
            self.sources[doc.source] = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content_hash": doc.metadata.get("content_hash"),
                "mtime": doc.metadata.get("mtime"),
                "format": doc.metadata.get("format"),
                "chunks": len(doc_chunks),
            }
            repairs = []
            spacing = doc.metadata.get("arabic_spacing_repaired_pages")
            if spacing:
                self.sources[doc.source]["arabic_spacing_repaired_pages"] = spacing
                repairs.append(f"word spacing on {len(spacing)} page(s)")
            if doc.metadata.get("arabic_word_order_repaired"):
                self.sources[doc.source]["arabic_word_order_repaired"] = True
                repairs.append("reversed word order")
            if progress:
                note = f" [repaired Arabic {', '.join(repairs)}]" if repairs else ""
                title = f" — {doc.title}" if doc.title and doc.title != Path(doc.source).stem else ""
                progress(doc.source, f"{len(doc_chunks)} chunks{title}{note}")

        if not pending:
            return report

        new_vectors: np.ndarray | None = None
        if embedder is not None:
            new_vectors = embedder.encode_passages([c.indexed_text for c in pending])
            self.embedder_name = getattr(embedder, "name", None)
            self.fingerprint = self.config.index_fingerprint()
        elif self.embeddings is not None:
            # Adding chunks with no embedder would leave the matrix covering
            # only part of the corpus. Rather than carry a half-dense index --
            # which silently hides the new content from dense search and
            # misaligns row indices on the next delete -- drop to BM25-only and
            # say so. Re-running ingest with dense enabled rebuilds it.
            log.warning(
                "Indexing without an embedder: discarding %d existing dense "
                "vectors so the index stays consistent. Re-run ingest with "
                "dense retrieval enabled to rebuild them.",
                len(np.asarray(self.embeddings)),
            )
            self.embeddings = None
            self.embedder_name = None
            self._vectors_dirty = True

        self.chunks.extend(pending)
        report.added_chunks = len(pending)

        if new_vectors is not None:
            if self.embeddings is None or len(self.embeddings) == 0:
                self.embeddings = new_vectors
            else:
                existing = np.asarray(self.embeddings)
                if existing.shape[1] != new_vectors.shape[1]:
                    raise ValueError(
                        f"Embedding dimension changed ({existing.shape[1]} -> "
                        f"{new_vectors.shape[1]}). Re-index from scratch: rag reset"
                    )
                self.embeddings = np.vstack([existing, new_vectors])
            self._vectors_dirty = True

        self._invalidate()
        return report

    # -- retrieval ---------------------------------------------------------

    def search_dense(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Exact cosine search (vectors are L2-normalised, so: a dot product)."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        matrix = np.asarray(self.embeddings)
        scores = matrix @ query_vector.astype(np.float32).ravel()
        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []
        # argpartition keeps this O(n) instead of sorting the whole corpus.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def search_lexical(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, top_k)

    def stats(self) -> dict[str, Any]:
        by_format: dict[str, int] = {}
        for chunk in self.chunks:
            fmt = chunk.metadata.get("format", "txt")
            by_format[fmt] = by_format.get(fmt, 0) + 1
        tokens = sum(c.token_count for c in self.chunks)
        return {
            "index_dir": str(self.dir),
            "sources": len(self.sources),
            "chunks": len(self.chunks),
            "tokens": tokens,
            "avg_chunk_tokens": (tokens // len(self.chunks)) if self.chunks else 0,
            "dense": self.embeddings is not None,
            "embedder": self.embedder_name,
            "dim": int(np.asarray(self.embeddings).shape[1]) if self.embeddings is not None else None,
            "by_format": by_format,
        }

    def check_fingerprint(self, keys: Sequence[str] | None = None) -> str | None:
        """Return a warning if the index was built with different settings.

        Pass ``keys`` to limit the check. At query time only the embedding
        model matters -- chunk size is fixed at ingest and cannot go stale for
        a search -- so querying checks ``("embed_model",)`` alone.
        """
        if not self.fingerprint:
            return None
        current = self.config.index_fingerprint()
        drifted = {
            k: (v, current.get(k))
            for k, v in self.fingerprint.items()
            if current.get(k) != v and (keys is None or k in keys)
        }
        if not drifted:
            return None
        details = "; ".join(f"{k}: indexed with {old!r}, now {new!r}" for k, (old, new) in drifted.items())
        return f"Index settings changed ({details}). Re-run ingest with --force."
