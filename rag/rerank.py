"""Optional LLM reranking.

Hybrid fusion is good at *recall* -- getting the right chunk into the top 30 --
and only approximately right about *order*. A reranker reads the query and each
candidate together and scores actual relevance, which is what the fused ranking
can only approximate.

It costs one API call per query on whichever provider is configured, so the
local cross-encoder in ``crossencoder.py`` is the default; this one is for when
you would rather spend an API call than CPU time.
"""

from __future__ import annotations

import re
from typing import Sequence

from .config import Config, DEFAULT_CONFIG
from .llm import get_backend
from .retriever import RetrievedChunk

RERANK_SYSTEM = """You score how well each excerpt answers a question.

Output one line per excerpt, exactly: ID=SCORE
SCORE is 0-10, where 10 means the excerpt directly answers the question, 5 means
it is related background, and 0 means it is irrelevant. Judge only whether the
excerpt helps answer *this* question. Output nothing else."""

_SCORE_RE = re.compile(r"^\s*(\d{1,3})\s*=\s*(\d{1,2}(?:\.\d+)?)\s*$", re.MULTILINE)
_SNIPPET_CHARS = 1200


def llm_rerank(
    query: str,
    candidates: Sequence[RetrievedChunk],
    config: Config = DEFAULT_CONFIG,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Re-order candidates by model-judged relevance.

    On any failure the original fused order is returned unchanged -- a reranker
    that can break retrieval is worse than no reranker.
    """
    if len(candidates) <= 1:
        return list(candidates)

    pool = list(candidates[: top_n or len(candidates)])
    excerpts = "\n\n".join(
        f'<excerpt id="{i}">\n{hit.chunk.indexed_text[:_SNIPPET_CHARS]}\n</excerpt>'
        for i, hit in enumerate(pool, start=1)
    )
    prompt = f"<question>{query}</question>\n\n{excerpts}"

    # Relevance judgement is a cheap discrimination task, not a reasoning one.
    completion = get_backend(config).complete(
        [RERANK_SYSTEM],
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        effort="low",
    )
    text = completion.text
    scores: dict[int, float] = {}
    for match in _SCORE_RE.finditer(text):
        idx, score = int(match.group(1)), float(match.group(2))
        if 1 <= idx <= len(pool):
            scores[idx] = score

    if not scores:  # unparseable response -- keep the fused order
        return list(candidates)

    for i, hit in enumerate(pool, start=1):
        hit.rerank_score = scores.get(i)

    # Unscored candidates sink below scored ones but keep their relative order.
    reranked = sorted(
        pool,
        key=lambda h: (h.rerank_score if h.rerank_score is not None else -1.0, h.score),
        reverse=True,
    )
    return reranked + list(candidates[len(pool):])
