"""Central configuration for the RAG system.

Settings come from a ``.env`` file next to the project (or in the working
directory), from environment variables, or from a ``Config`` passed in code --
in reverse order of precedence, so a real environment variable always wins over
the file. Defaults are chosen to work offline for everything except answer
generation, which needs a Gemini or Anthropic key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ENV_FILENAME = ".env"


def load_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=value`` lines into the environment, without overriding it.

    A deliberately small parser rather than a dependency: comments, blank
    lines, an optional ``export`` prefix and quoted values are all that a file
    holding a couple of API keys needs. An empty value is skipped, so a
    half-filled template cannot look like a configured key.
    """
    loaded: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return loaded
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"").strip()
        # A real environment variable wins, so `set KEY=... && python -m rag`
        # overrides the file for one run.
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_env_files() -> dict[str, str]:
    """Load ``.env`` from the working directory and from the project root."""
    project_root = Path(__file__).resolve().parent.parent
    loaded: dict[str, str] = {}
    for directory in dict.fromkeys([Path.cwd(), project_root]):
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            loaded.update(load_env_file(candidate))
    return loaded


# Must run before the Config defaults below are evaluated: they read
# os.environ at class-definition time.
LOADED_ENV = load_env_files()


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- storage -----------------------------------------------------------
    index_dir: Path = field(
        default_factory=lambda: Path(_env_str("RAG_INDEX_DIR", ".rag_index"))
    )

    # --- chunking ----------------------------------------------------------
    # Sized so that `top_k` chunks comfortably fit in context alongside the
    # prompt while still carrying enough surrounding text to be self-contained.
    chunk_tokens: int = _env_int("RAG_CHUNK_TOKENS", 512)
    chunk_overlap_tokens: int = _env_int("RAG_CHUNK_OVERLAP", 64)
    min_chunk_chars: int = _env_int("RAG_MIN_CHUNK_CHARS", 80)
    # Legislation: one chunk per article instead of fixed windows.
    article_chunks: bool = _env_bool("RAG_ARTICLE_CHUNKS", True)
    # Pack consecutive short articles up to this many tokens (0 = one per chunk).
    # Measured on the legal test set, packing lowered retrieval quality at every
    # context budget, so articles are indexed individually by default.
    article_pack_tokens: int = _env_int("RAG_ARTICLE_PACK_TOKENS", 0)
    # Prepend "<title> -- <article>" to every chunk before indexing it.
    context_headers: bool = _env_bool("RAG_CONTEXT_HEADERS", True)

    # --- embeddings --------------------------------------------------------
    # BAAI/bge-m3: 1024 dims, multilingual, strong on Arabic. Measured on the
    # legal test set it beat multilingual-e5-small and -e5-large both before and
    # after reranking (answer ranked first 80% vs 76% / 71%). The price is a
    # 2.2 GB download and slower indexing on CPU; set RAG_EMBED_MODEL to
    # intfloat/multilingual-e5-small for a fast, light alternative.
    embed_model: str = _env_str("RAG_EMBED_MODEL", "BAAI/bge-m3")
    embed_batch_size: int = _env_int("RAG_EMBED_BATCH", 16)
    embed_max_tokens: int = _env_int("RAG_EMBED_MAX_TOKENS", 512)
    embed_device: str = _env_str("RAG_EMBED_DEVICE", "auto")  # auto | cpu | cuda
    embed_pooling: str = _env_str("RAG_EMBED_POOLING", "auto")  # auto | mean | cls
    # e5 models are trained with asymmetric prefixes; harmless for other models
    # only if disabled, so it is tied to the model name at runtime.
    embed_query_prefix: str = _env_str("RAG_QUERY_PREFIX", "query: ")
    embed_passage_prefix: str = _env_str("RAG_PASSAGE_PREFIX", "passage: ")

    # --- retrieval ---------------------------------------------------------
    # Most passages to consider for an answer; context_tokens then decides how
    # many of them actually fit.
    top_k: int = _env_int("RAG_TOP_K", 12)
    context_tokens: int = _env_int("RAG_CONTEXT_TOKENS", 4000)
    candidate_k: int = _env_int("RAG_CANDIDATE_K", 30)  # per retriever, pre-fusion
    rrf_k: int = _env_int("RAG_RRF_K", 60)  # reciprocal-rank-fusion smoothing
    # Relative trust in each retriever during fusion. Equal weights are the
    # neutral prior; measure with eval/evaluate.py before moving them. Corpora
    # of paraphrasable prose usually want dense above 1.0, corpora full of
    # identifiers and error codes want lexical up.
    dense_weight: float = _env_float("RAG_DENSE_WEIGHT", 1.0)
    lexical_weight: float = _env_float("RAG_LEXICAL_WEIGHT", 1.0)
    use_mmr: bool = _env_bool("RAG_USE_MMR", True)
    mmr_lambda: float = _env_float("RAG_MMR_LAMBDA", 0.7)  # 1.0 = pure relevance
    min_score: float = _env_float("RAG_MIN_SCORE", 0.0)
    # Reorder the fused candidates with a model that reads question and
    # passage together. "local" is an offline cross-encoder; "llm" spends one
    # API call per query on the configured provider.
    # On by default: on the legal test set it lifted "right passage ranked
    # first" from 47% to 76%, for ~2 s per question on CPU.
    rerank: bool = _env_bool("RAG_RERANK", True)
    reranker: str = _env_str("RAG_RERANKER", "local")  # local | llm
    reranker_model: str = _env_str("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_candidates: int = _env_int("RAG_RERANK_CANDIDATES", 20)
    reranker_batch_size: int = _env_int("RAG_RERANKER_BATCH", 8)
    reranker_max_tokens: int = _env_int("RAG_RERANKER_MAX_TOKENS", 512)
    reranker_quantize: bool = _env_bool("RAG_RERANKER_INT8", True)  # int8 on CPU

    # --- generation --------------------------------------------------------
    # Which hosted model writes the answers. "auto" picks whichever provider has
    # credentials, preferring Gemini (it has a free tier). Retrieval is local
    # either way, so switching provider needs no re-indexing.
    llm_provider: str = _env_str("RAG_LLM_PROVIDER", "auto")  # auto | gemini | claude | openrouter
    # Empty means the provider's default (see llm.py): claude-opus-5 / gemini-3.8-flash.
    model: str = _env_str("RAG_MODEL", "")
    max_tokens: int = _env_int("RAG_MAX_TOKENS", 16000)
    effort: str = _env_str("RAG_EFFORT", "medium")  # low|medium|high|xhigh|max
    stream: bool = _env_bool("RAG_STREAM", True)
    cache_context: bool = _env_bool("RAG_CACHE_CONTEXT", True)
    # Transient provider failures (503 overload, 429 rate limit) are retried
    # with a doubling backoff; 4xx errors are not.
    retry_attempts: int = _env_int("RAG_RETRY_ATTEMPTS", 2)
    retry_initial_seconds: float = _env_float("RAG_RETRY_SECONDS", 1.5)

    def extract_cache_dir(self) -> Path | None:
        """Where extracted text is cached (RAG_EXTRACT_CACHE; "off" disables)."""
        override = os.environ.get("RAG_EXTRACT_CACHE")
        if override and override.lower() in {"off", "0", "none", "false"}:
            return None
        return Path(override) if override else Path(self.index_dir) / "extracted"

    def dense_enabled(self) -> bool:
        return _env_bool("RAG_DENSE", True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["index_dir"] = str(self.index_dir)
        return d

    # Fingerprint of everything that, if changed, makes an existing index stale.
    def index_fingerprint(self) -> dict[str, Any]:
        return {
            "embed_model": self.embed_model,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "embed_max_tokens": self.embed_max_tokens,
            "article_chunks": self.article_chunks,
            "article_pack_tokens": self.article_pack_tokens,
            "context_headers": self.context_headers,
        }


DEFAULT_CONFIG = Config()
