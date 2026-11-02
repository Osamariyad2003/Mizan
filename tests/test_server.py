"""HTTP API and the test page.

Run with: python -m pytest tests -q

A pipeline with a fake embedder is injected into the server module, so these
exercise real routing and real JSON with no model weights and no network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag.server as server
from rag.config import Config
from rag.pipeline import RAGPipeline

from test_rag import FakeEmbedder


@pytest.fixture
def client(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "labour.md").write_text(
        "Article 61. Every worker is entitled to fourteen days of paid annual leave.",
        encoding="utf-8",
    )
    (docs / "penal.md").write_text(
        "Article 401. Theft is taking movable property of another without consent.",
        encoding="utf-8",
    )
    config = Config(index_dir=tmp_path / "index", rerank=False)
    config.llm_provider = "auto"
    config.model = ""
    pipeline = RAGPipeline(config, load=False)
    pipeline._embedder = FakeEmbedder()
    pipeline.ingest([str(docs)])

    monkeypatch.setattr(server, "_pipeline", pipeline)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(server.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def test_root_serves_the_test_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>ميزان | البحث القانوني الأردني</title>" in response.text


def test_page_is_self_contained(client):
    """No CDN: the page must work offline, like the rest of the system."""
    html = client.get("/").text
    assert "src=\"http" not in html and "href=\"http" not in html.replace('href="https://aistudio', "")


def test_web_app_assets_are_served(client):
    assert client.get("/manifest.webmanifest").status_code == 200
    assert client.get("/manifest.webmanifest").json()["display"] == "standalone"
    assert client.get("/sw.js").status_code == 200
    assert client.get("/app-icon.svg").status_code == 200


def test_examples_come_from_the_evaluation_set(client):
    examples = client.get("/examples").json()
    assert 0 < len(examples) <= 4
    assert all(isinstance(q, str) and q.strip() for q in examples)


def test_examples_are_capped(client):
    assert len(client.get("/examples?limit=2").json()) == 2


# --------------------------------------------------------------------------
# endpoints the page calls
# --------------------------------------------------------------------------

def test_stats_reports_index_and_provider_fields(client):
    stats = client.get("/stats").json()
    assert stats["chunks"] > 0 and stats["sources"] == 2
    # No keys in this environment: retrieval works, answers are unavailable.
    assert stats["provider"] is None and stats["keys_found"] == []


def test_health_is_cheap_and_true(client):
    assert client.get("/health").json()["status"] == "ok"


def test_search_returns_ranked_passages(client):
    body = client.post("/search", json={"query": "annual leave", "top_k": 2}).json()
    assert body["query"] == "annual leave"
    assert 0 < len(body["results"]) <= 2
    top = body["results"][0]
    assert "labour.md" in top["source"]
    assert {"location", "rank", "score", "text"} <= top.keys()


def test_search_honours_the_source_filter(client):
    body = client.post("/search", json={"query": "article", "source": "penal"}).json()
    assert body["results"] and all("penal.md" in r["source"] for r in body["results"])


def test_ask_without_credentials_explains_what_to_set(client):
    response = client.post("/ask", json={"question": "how much leave?"})
    assert response.status_code == 502
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_ask_accepts_a_request_without_top_k(client):
    """top_k is optional: the token budget decides how many passages fit."""
    response = client.post("/ask", json={"question": "how much leave?"})
    assert response.status_code == 502  # reached generation, not a validation error


def test_ask_rejects_an_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_stream_sends_sources_then_an_error_event(client):
    with client.stream("POST", "/ask/stream", json={"question": "how much leave?"}) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "event: sources" in text
    assert "event: error" in text and "GEMINI_API_KEY" in text
    assert text.index("event: sources") < text.index("event: error")


def test_ingest_over_http_stays_disabled(client):
    response = client.post("/ingest", json={"paths": ["."]})
    assert response.status_code == 403 and "RAG_ALLOW_INGEST" in response.json()["detail"]


@pytest.mark.parametrize("top_k", [0, -1, 51, 1.5, "many"])
def test_search_rejects_invalid_top_k_boundaries(client, top_k):
    assert client.post("/search", json={"query": "article", "top_k": top_k}).status_code == 422


def test_search_rejects_wrong_content_type(client):
    response = client.post("/search", content="query=article", headers={"content-type": "text/plain"})
    assert response.status_code == 422


def test_history_rejects_non_object_entries_before_generation(client):
    response = client.post("/ask", json={"question": "q", "history": ["not-an-object"]})
    assert response.status_code == 422


def test_concurrent_searches_are_deterministic(client):
    from concurrent.futures import ThreadPoolExecutor

    def search_once(_):
        response = client.post("/search", json={"query": "annual leave", "top_k": 2})
        return response.status_code, [row["source"] for row in response.json()["results"]]

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(search_once, range(24)))
    assert all(outcome == outcomes[0] for outcome in outcomes)


def test_service_worker_is_not_cached_by_http_clients(client):
    response = client.get("/sw.js")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["service-worker-allowed"] == "/"


def test_ui_contains_initial_loading_empty_error_and_success_state_logic(client):
    html = client.get("/").text
    for marker in ("spinner", "لم نجد مقاطع مطابقة", 'type==="error"', 'type==="success"'):
        assert marker in html


@pytest.mark.xfail(strict=True, reason="SearchRequest accepts whitespace-only queries")
def test_search_rejects_whitespace_only_query(client):
    assert client.post("/search", json={"query": "   "}).status_code == 422


@pytest.mark.xfail(strict=True, reason="Pydantic request models silently ignore unknown fields")
def test_search_rejects_unknown_fields(client):
    assert client.post("/search", json={"query": "article", "admin": True}).status_code == 422


@pytest.mark.xfail(strict=True, reason="AskRequest has no history item/count/size limits")
def test_ask_rejects_unbounded_history(client):
    history = [{"role": "user", "content": "x"} for _ in range(1001)]
    assert client.post("/ask", json={"question": "q", "history": history}).status_code == 422
