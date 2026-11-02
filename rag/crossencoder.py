"""Local cross-encoder reranking.

Retrieval scores the question and each passage *separately* -- an embedding per
side, or keyword overlap -- which is fast enough to search thousands of chunks
but cannot weigh how well a passage actually answers the question. A
cross-encoder reads the pair *together* and scores relevance directly. It is too
slow to run over the whole corpus, so it only reorders the few dozen candidates
that fusion has already found.

Runs locally: no API key, no per-query cost, and it works offline once the
weights are downloaded. The default model is multilingual and handles Arabic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import Config, DEFAULT_CONFIG


@lru_cache(maxsize=4)
def _load(model_name: str, device: str, quantize: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:  # pragma: no cover - cosmetic only
        pass

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if quantize and device == "cpu":
        # int8 weights for the linear layers: several times faster on CPU,
        # where a cross-encoder is otherwise the slowest step in a query.
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.to(device)
    return tokenizer, model, device


class CrossEncoderReranker:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.name = config.reranker_model
        self.tokenizer, self.model, self.device = _load(
            self.name, config.embed_device, config.reranker_quantize
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to ``query``; higher is better."""
        if not passages:
            return []
        import torch

        # Batch passages of similar length together: a batch is padded to its
        # longest member, and legal articles range from one line to a page.
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        scores = [0.0] * len(passages)
        batch = max(1, self.config.reranker_batch_size)
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            window = [passages[i] for i in ids]
            encoded = self.tokenizer(
                [query] * len(window),
                window,
                padding=True,
                truncation="only_second",  # never truncate the question
                max_length=self.config.reranker_max_tokens,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            # One relevance logit per pair; some models emit [irrelevant, relevant].
            column = logits[:, -1] if logits.shape[-1] > 1 else logits.view(-1)
            for i, value in zip(ids, column.float().cpu().tolist()):
                scores[i] = value
        return scores


def local_rerank(query: str, candidates: Sequence, config: Config = DEFAULT_CONFIG) -> list:
    """Reorder retrieved chunks by cross-encoder relevance.

    Only the first ``rerank_candidates`` are scored; any beyond that keep their
    fused order after the reranked ones.
    """
    if len(candidates) <= 1:
        return list(candidates)
    pool = list(candidates[: config.rerank_candidates])
    reranker = CrossEncoderReranker(config)
    scores = reranker.score(query, [hit.chunk.indexed_text for hit in pool])
    for hit, score in zip(pool, scores):
        hit.rerank_score = score
    pool.sort(key=lambda hit: hit.rerank_score, reverse=True)
    return pool + list(candidates[len(pool):])
