"""Collect the facts the documentation quotes, so they cannot go stale.

    python docs/snapshot.py      # -> docs/snapshot.json

Run this before docs/make_pdf.py after changing the code or re-indexing.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import Config  # noqa: E402
from rag.store import VectorStore  # noqa: E402

PACKAGES = (
    "torch", "transformers", "pypdf", "numpy", "tiktoken", "fastapi",
    "typer", "rich", "google-genai", "anthropic", "pytest", "reportlab",
)


def collect() -> dict:
    snap: dict = {}
    snap["modules"] = {
        path.name: sum(1 for _ in path.open(encoding="utf-8"))
        for path in sorted((ROOT / "rag").glob("*.py"))
    }
    snap["tests"] = {
        path.name: sum(1 for line in path.open(encoding="utf-8") if line.startswith("def test_"))
        for path in sorted((ROOT / "tests").glob("*.py"))
    }
    snap["web_lines"] = sum(1 for _ in (ROOT / "rag/web/index.html").open(encoding="utf-8"))
    snap["eval_lines"] = sum(1 for _ in (ROOT / "eval/evaluate.py").open(encoding="utf-8"))

    store = VectorStore.load(Config())
    stats = store.stats()
    snap["index"] = {
        key: stats[key]
        for key in ("sources", "chunks", "tokens", "avg_chunk_tokens", "embedder", "dim", "by_format")
    }
    index_dir = pathlib.Path(Config().index_dir)
    snap["index_files"] = {f.name: f.stat().st_size for f in sorted(index_dir.iterdir()) if f.is_file()}
    snap["extract_cache_files"] = len(list((index_dir / "extracted").glob("*.json")))
    snap["documents"] = {
        pathlib.Path(source).name: {
            "title": meta.get("title"),
            "chunks": meta.get("chunks"),
            "format": meta.get("format"),
            "repaired_order": bool(meta.get("arabic_word_order_repaired")),
        }
        for source, meta in store.sources.items()
    }
    snap["article_chunks"] = sum(1 for c in store.chunks if c.metadata.get("article"))
    snap["config"] = {k: str(v) for k, v in Config().to_dict().items()}
    snap["versions"] = {}
    for package in PACKAGES:
        try:
            snap["versions"][package] = md.version(package)
        except Exception:
            snap["versions"][package] = "not installed"
    snap["python"] = sys.version.split()[0]
    return snap


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent / "snapshot.json"
    out.write_text(json.dumps(collect(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")
