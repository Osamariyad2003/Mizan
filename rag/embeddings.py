"""Dense embedding backends.

Two backends are supported:

* **local** (default) -- a sentence-embedding transformer run through
  ``transformers`` + ``torch`` with mean pooling and L2 normalisation. No API
  key, no network at query time once the weights are cached.
* **voyage** -- Voyage AI, used when ``VOYAGE_API_KEY`` is set. Anthropic does
  not serve an embeddings endpoint, so Voyage is the hosted option that pairs
  with Claude; for Gemini, Google's own embedding models are the hosted option.

Both return L2-normalised ``float32`` vectors, so cosine similarity is a plain
dot product and the rest of the system does not care which backend produced
them.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

import numpy as np

from .config import Config, DEFAULT_CONFIG


class Embedder(Protocol):
    name: str
    dim: int

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (mat / norms).astype(np.float32)


class LocalEmbedder:
    """Mean-pooled transformer encoder running on the local machine."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Dense retrieval needs torch and transformers "
                "(pip install torch transformers), or run with RAG_DENSE=0 "
                "for keyword-only search."
            ) from exc

        # Model-loading chatter would otherwise interleave with CLI output.
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:  # pragma: no cover - cosmetic only
            pass

        self._torch = torch
        self.config = config
        self.name = config.embed_model

        device = config.embed_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config.embed_model)
        self.model = AutoModel.from_pretrained(config.embed_model)
        self.model.eval()
        self.model.to(device)
        self.dim = int(self.model.config.hidden_size)

        # e5/gte-family models are trained with asymmetric query/passage
        # prefixes; applying them to a model that was not is actively harmful.
        lowered = config.embed_model.lower()
        # BGE models are trained to be read from the [CLS] token; e5 and most
        # sentence-transformers models from the mean of all tokens. Using the
        # wrong one quietly degrades retrieval rather than failing.
        self._pooling = config.embed_pooling
        if self._pooling == "auto":
            self._pooling = "cls" if "bge" in lowered else "mean"
        self._use_prefix = "e5" in lowered
        self._query_prefix = config.embed_query_prefix if self._use_prefix else ""
        self._passage_prefix = config.embed_passage_prefix if self._use_prefix else ""

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        torch = self._torch
        # Batch texts of similar length: each batch is padded to its longest
        # member, and a corpus of legal articles mixes one-liners with pages.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
        out: list[np.ndarray] = []
        batch = max(1, self.config.embed_batch_size)
        for i in range(0, len(order), batch):
            window = [prefix + (texts[j] or " ") for j in order[i : i + batch]]
            encoded = self.tokenizer(
                window,
                padding=True,
                truncation=True,
                max_length=self.config.embed_max_tokens,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
            if self._pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled.float().cpu().numpy())
        stacked = np.vstack(out)
        restored = np.empty_like(stacked)
        restored[order] = stacked  # back to the caller's order
        return _l2_normalize(restored)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._query_prefix)


class VoyageEmbedder:
    """Hosted embeddings via Voyage AI (used when VOYAGE_API_KEY is set)."""

    def __init__(self, config: Config = DEFAULT_CONFIG, model: str | None = None):
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "VOYAGE_API_KEY is set but the client is missing: pip install voyageai"
            ) from exc
        self._client = voyageai.Client()
        self.name = model or os.environ.get("VOYAGE_MODEL", "voyage-3")
        self.config = config
        self.dim = 1024  # confirmed on first call

    def _encode(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        batch = max(1, min(self.config.embed_batch_size * 4, 128))
        for i in range(0, len(texts), batch):
            result = self._client.embed(
                list(texts[i : i + batch]), model=self.name, input_type=input_type
            )
            vectors.extend(result.embeddings)
        mat = np.asarray(vectors, dtype=np.float32)
        self.dim = mat.shape[1]
        return _l2_normalize(mat)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")


def get_embedder(config: Config = DEFAULT_CONFIG) -> Embedder:
    """Pick a backend: Voyage when a key is present, otherwise local weights."""
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbedder(config)
    return LocalEmbedder(config)
