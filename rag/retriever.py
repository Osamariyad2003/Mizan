"""Hybrid retrieval: dense + BM25, fused, then diversified.

The pipeline is:

1. **Recall** -- take ``candidate_k`` from each retriever independently.
2. **Fuse** -- combine with Reciprocal Rank Fusion. RRF uses *ranks*, not raw
   scores, which is what makes it safe: a cosine similarity and a BM25 score
   live on incomparable scales, and any attempt to weight them directly needs
   per-corpus calibration that goes stale.
3. **Diversify** -- Maximal Marginal Relevance drops near-duplicate chunks, a
   real problem with overlapping windows over repetitive documents.
4. **Rerank** (optional) -- a cross-encoder (local, default) or the configured
   LLM provider rescores the survivors for relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .chunking import Chunk
from .config import Config, DEFAULT_CONFIG
from .store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    index: int = -1
    rank: int = 0  # final position after fusion, MMR and any reranking

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def location(self) -> str:
        return self.chunk.location

    def why(self) -> str:
        """Short explanation of how this chunk was found -- useful when tuning."""
        bits = []
        if self.dense_rank is not None:
            bits.append(f"dense #{self.dense_rank + 1} ({self.dense_score:.3f})")
        if self.lexical_rank is not None:
            bits.append(f"bm25 #{self.lexical_rank + 1} ({self.lexical_score:.2f})")
        if self.rerank_score is not None:
            bits.append(f"rerank {self.rerank_score:.2f}")
        return ", ".join(bits) or "n/a"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "location": self.location,
            "rank": self.rank,
            "score": round(self.score, 6),
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "rerank_score": self.rerank_score,
            "text": self.chunk.text,
        }


ChunkFilter = Callable[[Chunk], bool]


def make_filter(source: str | None = None, fmt: str | None = None) -> ChunkFilter | None:
    """Build a chunk predicate from the common CLI-level filters."""
    if not source and not fmt:
        return None

    def predicate(chunk: Chunk) -> bool:
        if source and source.lower() not in chunk.source.lower():
            return False
        if fmt and chunk.metadata.get("format") != fmt:
            return False
        return True

    return predicate


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """RRF: score(d) = sum over lists of weight / (k + rank(d)).

    ``k`` damps the contribution of top ranks so that a single retriever cannot
    dominate the fusion on the strength of one confident hit. ``weights`` lets a
    corpus that measurably favours one retriever say so -- equal weights are the
    right default, not the right answer for every corpus.
    """
    fused: dict[int, float] = {}
    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx] if weights and list_idx < len(weights) else 1.0
        for rank, (doc_idx, _score) in enumerate(ranked):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank + 1)
    return fused


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] so relevance and redundancy are comparable."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def mmr_select(
    vectors: np.ndarray,
    candidate_ids: Sequence[int],
    relevance_scores: Sequence[float],
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance over candidate row indices.

    Relevance comes from the *fused* ranking rather than raw cosine, so the
    first pick is always the top hybrid hit and MMR only breaks up redundancy
    below it. Both terms are normalised to [0, 1], which is what makes
    ``lambda_mult`` mean what it looks like it means.
    """
    if len(candidate_ids) <= top_k:
        return list(candidate_ids)

    cand = np.asarray(candidate_ids)
    mat = np.asarray(vectors)[cand]
    relevance = _minmax(np.asarray(relevance_scores, dtype=np.float32))

    selected: list[int] = []
    remaining = list(range(len(cand)))
    while remaining and len(selected) < top_k:
        if not selected:
            best = int(np.argmax(relevance[remaining]))
            selected.append(remaining.pop(best))
            continue
        chosen = mat[selected]
        # Redundancy = similarity to the most similar already-selected chunk.
        redundancy = np.clip((mat[remaining] @ chosen.T).max(axis=1), 0.0, 1.0)
        mmr = lambda_mult * relevance[remaining] - (1 - lambda_mult) * redundancy
        best = int(np.argmax(mmr))
        selected.append(remaining.pop(best))

    return [int(cand[i]) for i in selected]


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Any | None = None,
        config: Config = DEFAULT_CONFIG,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        chunk_filter: ChunkFilter | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.config
        top_k = top_k or cfg.top_k
        if self.store.size == 0:
            return []

        candidate_k = max(cfg.candidate_k, top_k * 4)

        # A filter must be applied before truncation, or it silently shrinks the
        # result set: widen recall, then filter, then cut to top_k.
        widen = candidate_k * 3 if chunk_filter else candidate_k

        dense_hits: list[tuple[int, float]] = []
        query_vector: np.ndarray | None = None
        if self.embedder is not None and self.store.embeddings is not None:
            query_vector = self.embedder.encode_queries([query])[0]
            dense_hits = self.store.search_dense(query_vector, widen)

        lexical_hits = self.store.search_lexical(query, widen)

        if chunk_filter is not None:
            dense_hits = [h for h in dense_hits if chunk_filter(self.store.chunks[h[0]])]
            lexical_hits = [h for h in lexical_hits if chunk_filter(self.store.chunks[h[0]])]

        dense_hits = dense_hits[:candidate_k]
        lexical_hits = lexical_hits[:candidate_k]
        if not dense_hits and not lexical_hits:
            return []

        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits)}
        dense_score = {idx: s for idx, s in dense_hits}
        lexical_rank = {idx: r for r, (idx, _) in enumerate(lexical_hits)}
        lexical_score = {idx: s for idx, s in lexical_hits}

        lists: list[Sequence[tuple[int, float]]] = []
        weights: list[float] = []
        if dense_hits:
            lists.append(dense_hits)
            weights.append(cfg.dense_weight)
        if lexical_hits:
            lists.append(lexical_hits)
            weights.append(cfg.lexical_weight)

        fused = reciprocal_rank_fusion(lists, k=cfg.rrf_k, weights=weights)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        candidate_ids = [idx for idx, _ in ordered]

        if cfg.use_mmr and self.store.embeddings is not None and len(candidate_ids) > top_k:
            pool = candidate_ids[: max(top_k * 4, top_k)]
            selected = mmr_select(
                self.store.embeddings,
                pool,
                [fused[i] for i in pool],
                top_k,
                cfg.mmr_lambda,
            )
            # Keep anything MMR dropped as the tail, so top_k is always filled.
            tail = [i for i in candidate_ids if i not in set(selected)]
            candidate_ids = selected + tail

        should_rerank = cfg.rerank if rerank is None else rerank
        # A reranker can only promote what it is shown, so give it a deeper pool.
        pool_size = max(top_k * 3, cfg.rerank_candidates if should_rerank else 0)

        results = [
            RetrievedChunk(
                chunk=self.store.chunks[idx],
                score=fused[idx],
                dense_score=dense_score.get(idx),
                lexical_score=lexical_score.get(idx),
                dense_rank=dense_rank.get(idx),
                lexical_rank=lexical_rank.get(idx),
                index=idx,
                rank=rank,
            )
            for rank, idx in enumerate(candidate_ids[: pool_size])
        ]

        if should_rerank and results:
            if cfg.reranker in ("llm", "claude"):  # "claude" kept as an alias
                from .rerank import llm_rerank

                results = llm_rerank(query, results, config=cfg)
            else:
                from .crossencoder import local_rerank

                results = local_rerank(query, results, config=cfg)

        results = results[:top_k]
        if cfg.min_score > 0:
            results = [r for r in results if r.score >= cfg.min_score]
        for rank, hit in enumerate(results):
            hit.rank = rank
        return results
