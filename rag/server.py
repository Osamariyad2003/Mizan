"""HTTP API.

    python -m rag serve --port 8000

    GET  /                 the test page
    GET  /health
    GET  /stats
    POST /search       {"query": "...", "top_k": 5}
    POST /ask          {"question": "...", "history": [...]}
    POST /ask/stream   same body, Server-Sent Events
    POST /ingest       {"paths": [...]}   (disabled unless RAG_ALLOW_INGEST=1)

The pipeline is loaded once at startup: embedding weights take seconds to load
and the whole index is read into memory, so per-request construction would
dominate latency.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Config
from .pipeline import RAGPipeline
from .retriever import make_filter

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(Config())
    return _pipeline


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load model weights at startup, not on the first user request: the
    # embedder and the reranker together take several seconds to load.
    rag = get_pipeline()
    _ = rag.embedder
    if rag.store.size:
        rag.search("warm-up", top_k=1)  # also builds BM25 and loads the reranker
    yield


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Hybrid retrieval over local documents, answered by Gemini or Claude.",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    source: str | None = None
    format: str | None = None
    # None means "use the configured default" rather than "off".
    rerank: bool | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # None lets context_tokens decide how many passages fit, which is the
    # measured-better behaviour; a number caps it.
    top_k: int | None = Field(None, ge=1, le=50)
    source: str | None = None
    format: str | None = None
    rerank: bool | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    force: bool = False


PAGE = Path(__file__).parent / "web" / "index.html"
WEB_DIR = PAGE.parent


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The test page: ask questions, see which articles answered them."""
    return FileResponse(PAGE, media_type="text/html")


@app.get("/manifest.webmanifest", include_in_schema=False)
def web_manifest() -> FileResponse:
    """Metadata that makes the browser client installable as a web app."""
    return FileResponse(WEB_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> FileResponse:
    """Offline shell; API calls intentionally remain network-first."""
    return FileResponse(
        WEB_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/app-icon.svg", include_in_schema=False)
def app_icon() -> FileResponse:
    return FileResponse(WEB_DIR / "app-icon.svg", media_type="image/svg+xml")


@app.get("/examples", include_in_schema=False)
def examples(limit: int = 4) -> list[str]:
    """A few questions to click on the test page.

    Taken from the evaluation set when there is one, so the page opens with
    questions that are known to have answers in the indexed documents.
    """
    dataset = Path(__file__).resolve().parent.parent / "eval" / "laws_questions.jsonl"
    if not dataset.exists():
        return []
    by_kind: dict[str, list[str]] = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_kind.setdefault(row.get("kind", "all"), []).append(row["question"])
    # One of each kind first, so the page shows the range (Arabic, English,
    # article lookup) rather than four variations of the same thing.
    picked: list[str] = []
    for kind in sorted(by_kind):
        picked.extend(by_kind[kind][:1])
    for kind in sorted(by_kind):
        picked.extend(q for q in by_kind[kind][1:3] if q not in picked)
    return picked[:limit]


@app.get("/health")
def health() -> dict[str, Any]:
    rag = get_pipeline()
    return {"status": "ok", "chunks": rag.store.size, "dense": rag.store.embeddings is not None}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return get_pipeline().stats()


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    rag = get_pipeline()
    hits = rag.search(
        req.query,
        top_k=req.top_k,
        chunk_filter=make_filter(req.source, req.format),
        rerank=req.rerank,
    )
    return {"query": req.query, "results": [h.to_dict() for h in hits]}


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    rag = get_pipeline()
    try:
        answer = rag.ask(
            req.question,
            top_k=req.top_k,
            chunk_filter=make_filter(req.source, req.format),
            history=req.history,
            rerank=req.rerank,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return answer.to_dict()


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE: a `sources` event first, then `delta` events, then `done`."""
    rag = get_pipeline()
    hits, stream = rag.ask_stream(
        req.question,
        top_k=req.top_k,
        chunk_filter=make_filter(req.source, req.format),
        history=req.history,
        rerank=req.rerank,
    )

    def events() -> Iterator[str]:
        payload = [h.to_dict() for h in hits]
        yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        try:
            for delta in stream:
                yield f"event: delta\ndata: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    """Index paths on the server's filesystem.

    Off by default: it reads arbitrary server-side paths, which is a real
    capability to hand to an HTTP caller. Enable deliberately with
    ``RAG_ALLOW_INGEST=1`` behind whatever authentication you already run.
    """
    if os.environ.get("RAG_ALLOW_INGEST", "").lower() not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=403,
            detail="Ingestion over HTTP is disabled. Set RAG_ALLOW_INGEST=1 to enable.",
        )
    rag = get_pipeline()
    report = rag.ingest(req.paths, force=req.force)
    return {
        "summary": report.summary(),
        "added_chunks": report.added_chunks,
        "failures": report.failures,
        "chunks": rag.store.size,
    }
