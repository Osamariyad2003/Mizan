"""Retrieval evaluation harness.

Retrieval quality is the part of a RAG system that actually moves answer
quality, and it is measurable without touching the generator. This compares the
three retrieval modes on the same questions:

    python eval/evaluate.py --dataset eval/questions.jsonl

Dataset format, one JSON object per line::

    {"question": "How do I roll back a deploy?",
     "expected_source": "deploy.md",          # substring match on the path
     "expected_text": "ship rollback"}        # optional substring in the chunk

A question counts as a hit when a retrieved chunk matches *either* expectation,
so a dataset can be written from memory without knowing the chunk boundaries.

``--generate`` builds a starter dataset by asking the configured model (Gemini
or Claude) to write questions from sampled chunks (needs an API key). Synthetic questions are a starting point, not
a substitute for real user queries -- they are drawn from the same text the
retriever indexes, which flatters the retriever.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.arabic import normalize_for_search  # noqa: E402
from rag.config import Config  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import RetrievedChunk  # noqa: E402


def _fold(text: str) -> str:
    """Compare text the way search does: Arabic variants folded, spacing collapsed."""
    return " ".join(normalize_for_search(text).lower().split())


@dataclass
class Case:
    question: str
    expected_source: str | None = None
    expected_text: str | None = None
    kind: str = "all"
    match_all: bool = False  # require source AND text, not either

    def matches(self, hit: RetrievedChunk) -> bool:
        source_ok = bool(self.expected_source) and (
            self.expected_source.lower() in hit.chunk.source.lower()
        )
        text_ok = bool(self.expected_text) and _fold(self.expected_text) in _fold(hit.text)
        if self.match_all:
            return (source_ok or not self.expected_source) and (text_ok or not self.expected_text)
        return source_ok or text_ok

    def is_usable(self) -> bool:
        return bool(self.expected_source or self.expected_text)


def load_dataset(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON ({exc})")
        case = Case(
            question=row["question"],
            expected_source=row.get("expected_source"),
            expected_text=row.get("expected_text"),
            kind=row.get("kind", "all"),
            match_all=row.get("match") == "all",
        )
        if not case.is_usable():
            raise SystemExit(
                f"{path}:{line_no}: needs expected_source or expected_text to be scorable"
            )
        cases.append(case)
    return cases


@dataclass
class Metrics:
    mode: str
    n: int = 0
    hits_at_1: int = 0
    hits_at_k: int = 0
    reciprocal_rank_total: float = 0.0
    seconds: float = 0.0
    misses: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.misses is None:
            self.misses = []

    @property
    def ms_per_query(self) -> float:
        return 1000 * self.seconds / self.n if self.n else 0.0

    def record(self, hits: Sequence[RetrievedChunk], case: Case, seconds: float = 0.0) -> None:
        self.n += 1
        self.seconds += seconds
        ranks = [i for i, hit in enumerate(hits) if case.matches(hit)]
        if not ranks:
            self.misses.append(case.question)
            return
        first = ranks[0]
        self.hits_at_k += 1
        if first == 0:
            self.hits_at_1 += 1
        self.reciprocal_rank_total += 1.0 / (first + 1)

    @property
    def precision_at_1(self) -> float:
        return self.hits_at_1 / self.n if self.n else 0.0

    @property
    def recall_at_k(self) -> float:
        return self.hits_at_k / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank_total / self.n if self.n else 0.0


MODES = {
    # label: (dense, lexical-only override, rerank)
    "bm25 only": {"dense": False, "rerank": False},
    "dense only": {"dense": True, "lexical": False, "rerank": False},
    "hybrid": {"dense": True, "rerank": False},
    "hybrid + rerank": {"dense": True, "rerank": True},
}


def _retrieve(rag: RAGPipeline, case: Case, mode: dict[str, Any], top_k: int) -> list[RetrievedChunk]:
    """Run one retrieval mode by temporarily disabling a retriever."""
    store = rag.store
    saved_embeddings = store.embeddings
    saved_bm25 = store._bm25

    try:
        if not mode.get("dense", True):
            store.embeddings = None
        if mode.get("lexical", True) is False:
            # Empty BM25 index -> dense-only fusion.
            from rag.bm25 import BM25Index

            store._bm25 = BM25Index([])
        return rag.search(case.question, top_k=top_k, rerank=mode.get("rerank", False))
    finally:
        store.embeddings = saved_embeddings
        store._bm25 = saved_bm25


def evaluate(
    rag: RAGPipeline, cases: Sequence[Case], top_k: int, modes: Sequence[str],
    by_kind: bool = True,
) -> list[Metrics]:
    """Score each mode overall, and per question kind (ar / en / article)."""
    results = []
    kinds = sorted({c.kind for c in cases}) if by_kind else []
    # Build the lexical index and load the embedder before timing anything:
    # both are one-off startup costs, not per-query latency. (_retrieve swaps
    # the BM25 index out and back, so it must exist before the first swap.)
    _ = rag.store.bm25
    if rag.embedder is not None:
        rag.embedder.encode_queries(["warm-up"])
    for label in modes:
        mode = MODES[label]
        if mode.get("dense", True) and rag.store.embeddings is None:
            continue  # index has no vectors; nothing to compare
        overall = Metrics(mode=label)
        per_kind = {k: Metrics(mode=f"  {label} [{k}]") for k in kinds}
        for case in cases:
            start = time.perf_counter()
            hits = _retrieve(rag, case, mode, top_k)
            elapsed = time.perf_counter() - start
            overall.record(hits, case, elapsed)
            if case.kind in per_kind:
                per_kind[case.kind].record(hits, case, elapsed)
        results.append(overall)
        if len(per_kind) > 1:
            results.extend(per_kind.values())
    return results


def sweep_weights(
    rag: RAGPipeline, cases: Sequence[Case], top_k: int
) -> list[Metrics]:
    """Try a few dense:lexical fusion ratios and report each.

    Which retriever to trust is a property of the corpus, not a universal
    constant, so it is worth measuring rather than assuming.
    """
    ratios = [(1.0, 2.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 0.0)]
    original = (rag.config.dense_weight, rag.config.lexical_weight)
    results = []
    try:
        for dense_w, lex_w in ratios:
            rag.config.dense_weight = dense_w
            rag.config.lexical_weight = lex_w
            metrics = Metrics(mode=f"dense {dense_w:g} : bm25 {lex_w:g}")
            for case in cases:
                metrics.record(rag.search(case.question, top_k=top_k), case)
            results.append(metrics)
    finally:
        rag.config.dense_weight, rag.config.lexical_weight = original
    return results


def print_report(results: Sequence[Metrics], top_k: int) -> None:
    width = max(len(r.mode) for r in results) + 2
    print(f"\n{'mode'.ljust(width)}{'P@1':>8}{f'recall@{top_k}':>12}{'MRR':>8}{'n':>6}{'ms/q':>8}")
    print("-" * (width + 42))
    for r in results:
        print(
            f"{r.mode.ljust(width)}"
            f"{r.precision_at_1:>8.2f}{r.recall_at_k:>12.2f}{r.mrr:>8.3f}{r.n:>6}{r.ms_per_query:>8.0f}"
        )
    best = max(
        (r for r in results if not r.mode.startswith(" ")),
        key=lambda r: (r.recall_at_k, r.mrr),
        default=None,
    )
    if best:
        print(f"\nBest recall@{top_k}: {best.mode}")
    print(
        "\nP@1     the top chunk was relevant\n"
        f"recall@{top_k} a relevant chunk appeared anywhere in the top {top_k}\n"
        "MRR     1/rank of the first relevant chunk, averaged"
    )


GENERATE_SYSTEM = """You write evaluation questions for a document retrieval system.

Given an excerpt, write ONE specific question that the excerpt answers and that
someone would plausibly ask. The question must be answerable from this excerpt
alone and must not quote it verbatim. Output only the question."""


def generate_dataset(rag: RAGPipeline, out_path: Path, n: int, seed: int = 0) -> None:
    """Draft an eval dataset by asking the configured model to write questions."""
    from rag.llm import get_backend

    chunks = [c for c in rag.store.chunks if len(c.text) > 300]
    if not chunks:
        raise SystemExit("Index has no chunks substantial enough to build questions from.")

    random.seed(seed)
    sample = random.sample(chunks, min(n, len(chunks)))
    backend = get_backend(rag.config)

    rows = []
    for i, chunk in enumerate(sample, 1):
        question = backend.complete(
            [GENERATE_SYSTEM],
            [{"role": "user", "content": chunk.text[:4000]}],
            max_tokens=300,
            effort="low",
        ).text.strip()
        if not question:
            continue
        rows.append(
            {
                "question": question,
                "expected_source": Path(chunk.source).name,
                "expected_text": chunk.text[:60].strip(),
            }
        )
        print(f"  [{i}/{len(sample)}] {question[:72]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} cases to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("eval/laws_questions.jsonl"))
    parser.add_argument("--show-misses", action="store_true",
                        help="List the questions with no correct passage in the top-k.")
    parser.add_argument("--index", type=str, default=None, help="Index directory.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", nargs="*", default=["bm25 only", "dense only", "hybrid"],
                        choices=list(MODES), help="Retrieval modes to compare.")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep dense:bm25 fusion weights instead of comparing modes.")
    parser.add_argument("--generate", type=int, metavar="N",
                        help="Generate N synthetic cases into --dataset and exit (uses the API).")
    args = parser.parse_args()

    config = Config()
    if args.index:
        config.index_dir = Path(args.index)
    rag = RAGPipeline(config)

    if rag.store.size == 0:
        raise SystemExit(f"Index at {config.index_dir} is empty. Run: python -m rag ingest <path>")

    if args.generate:
        generate_dataset(rag, args.dataset, args.generate)
        return

    if not args.dataset.exists():
        raise SystemExit(
            f"No dataset at {args.dataset}. Create one, or generate a draft:\n"
            f"  python eval/evaluate.py --generate 20 --dataset {args.dataset}"
        )

    cases = load_dataset(args.dataset)
    print(f"Evaluating {len(cases)} questions against {rag.store.size} chunks...")

    if args.sweep:
        if rag.store.embeddings is None:
            raise SystemExit("Weight sweep needs a dense index; nothing to weigh against BM25.")
        print_report(sweep_weights(rag, cases, args.top_k), args.top_k)
        print("\nSet the winner with RAG_DENSE_WEIGHT / RAG_LEXICAL_WEIGHT.")
        return

    results = evaluate(rag, cases, args.top_k, args.modes)
    if not results:
        raise SystemExit("No modes could run (index has no dense vectors?).")
    print_report(results, args.top_k)
    if args.show_misses:
        for r in results:
            if r.misses and not r.mode.startswith(" "):
                print(f"\nMissed by {r.mode}:")
                for question in r.misses:
                    print(f"  - {question}")


if __name__ == "__main__":
    main()
