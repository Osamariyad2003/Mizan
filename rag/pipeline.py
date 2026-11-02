"""The facade: ingest documents, search them, ask questions about them.

    from rag import RAGPipeline

    rag = RAGPipeline()
    rag.ingest(["./docs"])
    answer = rag.ask("What is the deployment procedure?")
    print(answer.text)
    for source in answer.cited_sources:
        print(" -", source.location)
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .chunking import Chunk
from .config import Config, DEFAULT_CONFIG
from .generate import Answer, generate_answer, stream_answer
from .loaders import discover, load_document
from .retriever import ChunkFilter, RetrievedChunk, Retriever
from .store import IngestReport, VectorStore

log = logging.getLogger("rag")


class RAGPipeline:
    def __init__(
        self,
        config: Config | None = None,
        load: bool = True,
        adopt_index_settings: bool = True,
    ):
        # Copy the shared default so adopting index settings never leaks into
        # other pipelines in the same process.
        self.config = config if config is not None else replace(DEFAULT_CONFIG)
        self.store = VectorStore.load(self.config) if load else VectorStore(self.config)
        self._embedder: Any | None = None
        self._embedder_failed = False
        if load and adopt_index_settings:
            self._adopt_index_chunking()

    def _adopt_index_chunking(self) -> None:
        """Keep chunking consistent with how the existing index was built.

        Without this, an index built with RAG_CHUNK_TOKENS=1200 would get 512-
        token chunks for every file added later unless the variable was set
        again each time. A value the caller chose explicitly is left alone.
        """
        stored = self.store.fingerprint
        for field_name, env_var in (
            ("chunk_tokens", "RAG_CHUNK_TOKENS"),
            ("chunk_overlap_tokens", "RAG_CHUNK_OVERLAP"),
        ):
            if field_name not in stored or env_var in os.environ:
                continue
            default = Config.__dataclass_fields__[field_name].default
            if getattr(self.config, field_name) == default:
                setattr(self.config, field_name, stored[field_name])

    # -- components --------------------------------------------------------

    @property
    def embedder(self) -> Any | None:
        """Load embedding weights on first use; ``None`` means keyword-only.

        Dense retrieval is a quality upgrade, not a hard requirement, so a
        missing model or absent torch degrades to BM25 with a warning instead of
        taking the whole system down.
        """
        if self._embedder is None and not self._embedder_failed:
            if not self.config.dense_enabled():
                self._embedder_failed = True
                return None
            try:
                from .embeddings import get_embedder

                self._embedder = get_embedder(self.config)
            except Exception as exc:
                self._embedder_failed = True
                log.warning("Dense retrieval unavailable (%s); using BM25 only.", exc)
        return self._embedder

    @property
    def retriever(self) -> Retriever:
        return Retriever(self.store, self.embedder, self.config)

    # -- ingestion ---------------------------------------------------------

    def ingest(
        self,
        paths: Sequence[str | Path],
        recursive: bool = True,
        force: bool = False,
        progress: Callable[[str, str], None] | None = None,
    ) -> IngestReport:
        """Index files and directories. Unchanged files are skipped.

        Files that were indexed from one of these directories but no longer
        exist there (deleted or renamed) are removed from the index first --
        otherwise their passages keep appearing in answers, and a renamed file
        would be mistaken for a duplicate of its own former name.
        """
        report = IngestReport()
        report.pruned = self._prune_missing(paths)
        documents = []
        for path in discover(paths, recursive=recursive):
            try:
                documents.append(load_document(path, cache_dir=self.config.extract_cache_dir()))
            except Exception as exc:
                report.failures.append((str(path), str(exc)))
                log.warning("Could not read %s: %s", path, exc)

        sub_report = self.store.add_documents(
            documents, embedder=self.embedder, force=force, progress=progress
        )
        sub_report.failures.extend(report.failures)
        sub_report.pruned = report.pruned
        sub_report.removed_chunks += sum(self._pruned_chunk_counts.values())
        self.store.save()
        return sub_report

    def _prune_missing(self, paths: Sequence[str | Path]) -> list[str]:
        """Drop indexed files under ``paths`` that no longer exist on disk.

        Scoped to the directories being ingested, so ingesting one folder never
        touches sources from another, and non-file sources (``ingest_text``)
        are never pruned.
        """
        self._pruned_chunk_counts: dict[str, int] = {}
        roots = [Path(p).resolve() for p in paths if Path(p).is_dir()]
        if not roots:
            return []
        for source in list(self.store.sources):
            candidate = Path(source)
            if "://" in source or candidate.exists():
                continue
            resolved = candidate.resolve()
            if any(resolved.is_relative_to(root) for root in roots):
                self._pruned_chunk_counts[source] = self.store.remove_source(source)
        return list(self._pruned_chunk_counts)

    def ingest_text(self, text: str, source: str, title: str | None = None) -> IngestReport:
        """Index a string directly -- for content that never lands on disk."""
        import hashlib

        from .loaders import Document

        doc = Document(
            doc_id=hashlib.sha1(source.encode("utf-8")).hexdigest()[:16],
            source=source,
            title=title or source,
            text=text,
            metadata={
                "format": "text",
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            },
        )
        report = self.store.add_documents([doc], embedder=self.embedder, force=True)
        self.store.save()
        return report

    def remove(self, source: str) -> int:
        removed = self.store.remove_source(source)
        self.store.save()
        return removed

    def reset(self) -> None:
        self.store.clear()

    # -- query -------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        chunk_filter: ChunkFilter | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve without generating -- no API key needed."""
        if self.store.size == 0:
            # Nothing to search: return before loading embedding weights, which
            # takes seconds and would be pure waste on an empty index.
            return []
        return self.retriever.retrieve(query, top_k=top_k, chunk_filter=chunk_filter, rerank=rerank)

    def context_for(
        self,
        question: str,
        top_k: int | None = None,
        chunk_filter: ChunkFilter | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """The passages to put in front of the model, filled to a token budget.

        A budget rather than a fixed count, because chunk sizes vary tenfold --
        one article can be a sentence, a page window a thousand tokens. Measured
        on the legal test set, filling ~4,000 tokens with the best-ranked chunks
        puts the answering passage in front of the model far more often than a
        fixed six. The top hit is always included, whatever its size.
        """
        cap = top_k or self.config.top_k
        hits = self.search(question, top_k=cap, chunk_filter=chunk_filter, rerank=rerank)
        budget = self.config.context_tokens
        chosen: list[RetrievedChunk] = []
        used = 0
        for hit in hits:
            if chosen and used + hit.chunk.token_count > budget:
                break
            chosen.append(hit)
            used += hit.chunk.token_count
        return chosen

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        chunk_filter: ChunkFilter | None = None,
        history: Sequence[dict[str, Any]] | None = None,
        rerank: bool | None = None,
    ) -> Answer:
        hits = self.context_for(question, top_k=top_k, chunk_filter=chunk_filter, rerank=rerank)
        return generate_answer(question, hits, self.config, history=history)

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        chunk_filter: ChunkFilter | None = None,
        history: Sequence[dict[str, Any]] | None = None,
        rerank: bool | None = None,
    ) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Return the sources up front, then a generator of answer text.

        Sources first so a UI can render citations while the answer streams.
        """
        hits = self.context_for(question, top_k=top_k, chunk_filter=chunk_filter, rerank=rerank)
        return hits, stream_answer(question, hits, self.config, history=history)

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        from .llm import credentials_present, default_model, resolve_provider

        stats = self.store.stats()
        stats["warning"] = self.store.check_fingerprint()
        keys = credentials_present()
        stats["keys_found"] = sorted(name for name, ok in keys.items() if ok)
        # Where the key came from, so "it is set but not picked up" is visible.
        from .config import ENV_FILENAME, LOADED_ENV

        stats["env_file_keys"] = sorted(LOADED_ENV)
        stats["env_filename"] = ENV_FILENAME
        try:
            provider = resolve_provider(self.config)
            stats["provider"] = provider
            stats["answer_model"] = self.config.model or default_model(provider)
        except Exception:
            # No credentials, or an unknown provider name: retrieval still works.
            stats["provider"] = None
            stats["answer_model"] = None
        return stats

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.store.chunks if c.chunk_id == chunk_id), None)
