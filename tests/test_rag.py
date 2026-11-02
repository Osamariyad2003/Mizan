"""Test suite. Run with: python -m pytest tests -q

Uses a deterministic hash-based fake embedder rather than real model weights:
the tests then exercise the plumbing (persistence, fusion, filtering, citation
parsing) in under a second and without a network. Embedding *quality* is a
different question, and that is what eval/evaluate.py is for.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rag.bm25 import BM25Index, tokenize
from rag.chunking import chunk_document, count_tokens
from rag.config import Config
from rag.generate import format_context, parse_citations
from rag.loaders import Document, load_document
from rag.pipeline import RAGPipeline
from rag.retriever import RetrievedChunk, Retriever, make_filter, mmr_select, reciprocal_rank_fusion
from rag.store import VectorStore


class FakeEmbedder:
    """Deterministic pseudo-embeddings: same text -> same vector."""

    name = "fake-embedder"
    dim = 32

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer((digest * 2)[: self.dim], dtype=np.uint8).astype(np.float32)
        raw -= 127.5  # centre so vectors are not all in one orthant
        norm = float(np.linalg.norm(raw)) or 1.0
        return (raw / norm).astype(np.float32)

    def encode_passages(self, texts):
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    encode_queries = encode_passages


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(index_dir=tmp_path / "index", chunk_tokens=64, chunk_overlap_tokens=12)


@pytest.fixture
def corpus_dir(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text(
        "# Deployment\n\n"
        "Deploys run through the release pipeline. To roll back, run "
        "`ship rollback --to PREVIOUS` within thirty minutes.\n\n"
        "After thirty minutes the artifact is garbage collected and a rollback "
        "requires a fresh build from the tagged commit.\n",
        encoding="utf-8",
    )
    (docs / "oncall.md").write_text(
        "# On-call\n\nPager escalation goes to the secondary after ten minutes. "
        "Error code E_QUOTA_EXCEEDED means the tenant hit its write quota.\n",
        encoding="utf-8",
    )
    (docs / "ignored.bin").write_bytes(b"\x00\x01binary")
    return docs


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_chunk_offsets_map_back_to_source(config):
    doc = Document("d1", "s.txt", "s.txt", "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12)))
    chunks = chunk_document(doc, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end].strip() == chunk.text


def test_chunks_respect_token_budget(config):
    doc = Document("d1", "s.txt", "s.txt", " ".join(f"token{i}" for i in range(4000)))
    for chunk in chunk_document(doc, config):
        # Small slack: an atom is never split below the word level.
        assert chunk.token_count <= config.chunk_tokens * 1.35


def test_consecutive_chunks_overlap(config):
    doc = Document("d1", "s.txt", "s.txt", "\n\n".join(f"Para {i}. " + "alpha beta " * 30 for i in range(8)))
    chunks = chunk_document(doc, config)
    overlaps = sum(1 for a, b in zip(chunks, chunks[1:]) if b.char_start < a.char_end)
    assert overlaps >= len(chunks) // 2


def test_empty_and_tiny_documents(config):
    assert chunk_document(Document("d", "s", "t", "   \n  "), config) == []
    assert len(chunk_document(Document("d", "s", "t", "one short line"), config)) == 1


def test_unsplittable_blob_is_still_bounded(config):
    chunks = chunk_document(Document("d", "s", "t", "x" * 50_000), config)
    assert chunks
    assert max(c.token_count for c in chunks) <= config.chunk_tokens * 1.35


def test_overlap_survives_paragraphs_larger_than_the_overlap_budget(config):
    """Regression: whole-atom-only overlap silently produced zero overlap."""
    doc = Document("d1", "s.txt", "s.txt", "\n\n".join(f"Para {i}. " + "alpha beta " * 30 for i in range(8)))
    chunks = chunk_document(doc, config)
    overlaps = sum(1 for a, b in zip(chunks, chunks[1:]) if b.char_start < a.char_end)
    assert overlaps >= len(chunks) // 2


def test_degenerate_config_still_terminates(tmp_path):
    """Overlap >= chunk size must not loop forever or lose content."""
    cfg = Config(index_dir=tmp_path, chunk_tokens=32, chunk_overlap_tokens=32)
    doc = Document("d1", "s.txt", "s.txt", " ".join(f"word{i}" for i in range(600)))
    chunks = chunk_document(doc, cfg)
    assert chunks
    assert chunks[-1].char_end >= len(doc.text.rstrip()) - 1


def test_no_content_is_lost_between_chunks(config):
    doc = Document("d1", "s.txt", "s.txt", "\n\n".join(f"Sentence number {i} with filler words." for i in range(60)))
    chunks = chunk_document(doc, config)
    covered = "".join(doc.text[c.char_start : c.char_end] for c in chunks)
    for i in (0, 17, 59):
        assert f"Sentence number {i} " in covered


def test_pdf_pages_become_chunk_locations(config):
    doc = Document(
        "d", "r.pdf", "r.pdf",
        "Page one content here.\n\nPage two content here.",
        page_spans=[(0, 24, "1"), (24, 46, "2")],
    )
    chunks = chunk_document(doc, config)
    assert any("p." in c.location for c in chunks)


def test_count_tokens_is_positive():
    assert count_tokens("hello world") > 0
    assert count_tokens("") >= 0


# --------------------------------------------------------------------------
# lexical retrieval
# --------------------------------------------------------------------------

def test_bm25_ranks_exact_identifier_first():
    corpus = [
        "General notes about quotas and limits in the platform.",
        "Error code E_QUOTA_EXCEEDED means the tenant hit its write quota.",
        "Unrelated text about deployment pipelines.",
    ]
    assert BM25Index(corpus).search("E_QUOTA_EXCEEDED", top_k=3)[0][0] == 1


def test_bm25_handles_non_latin_script():
    idx = BM25Index(["hello world", "برنامج التخطيط الاستراتيجي"])
    assert idx.search("التخطيط", top_k=1)[0][0] == 1


def test_bm25_empty_cases():
    assert BM25Index([]).search("x") == []
    assert BM25Index(["the and of"]).search("the of") == []


def test_tokenizer_splits_on_punctuation_and_case():
    assert tokenize("Hello, WORLD! foo_bar", drop_stopwords=False) == ["hello", "world", "foo", "bar"]


# --------------------------------------------------------------------------
# fusion and diversification
# --------------------------------------------------------------------------

def test_rrf_rewards_agreement_between_retrievers():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    lexical = [(3, 5.0), (1, 4.0), (9, 1.0)]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    # 1 is top-1 in one list and top-2 in the other: it should beat 2 and 9,
    # each of which appears in only one list.
    assert fused[1] > fused[2] and fused[1] > fused[9]


def test_rrf_on_single_list_preserves_order():
    fused = reciprocal_rank_fusion([[(5, 1.0), (6, 0.5)]], k=60)
    assert fused[5] > fused[6]


def test_rrf_weights_shift_the_balance():
    dense = [(1, 0.9)]
    lexical = [(2, 5.0)]
    assert reciprocal_rank_fusion([dense, lexical], weights=[3.0, 1.0])[1] > \
           reciprocal_rank_fusion([dense, lexical], weights=[3.0, 1.0])[2]
    assert reciprocal_rank_fusion([dense, lexical], weights=[1.0, 3.0])[2] > \
           reciprocal_rank_fusion([dense, lexical], weights=[1.0, 3.0])[1]


def test_zero_lexical_weight_makes_retrieval_dense_only(config, corpus_dir):
    rag = RAGPipeline(config, load=False)
    rag._embedder = FakeEmbedder()
    rag.ingest([str(corpus_dir)])
    config.lexical_weight = 0.0
    hits = rag.search("rollback", top_k=3)
    # Lexical hits contribute nothing to the fused score.
    assert all(h.score > 0 for h in hits)
    assert any(h.dense_rank is not None for h in hits)


def test_mmr_prefers_the_top_hit_then_diversifies():
    vectors = np.array(
        [
            [1.0, 0.0],   # 0
            [0.99, 0.14], # 1 - near-duplicate of 0
            [0.0, 1.0],   # 2 - orthogonal
        ],
        dtype=np.float32,
    )
    picked = mmr_select(vectors, [0, 1, 2], [1.0, 0.9, 0.5], top_k=2, lambda_mult=0.5)
    assert picked[0] == 0
    assert picked[1] == 2  # diversity beats the near-duplicate


def test_mmr_is_a_noop_when_candidates_fit():
    assert mmr_select(np.eye(3, dtype=np.float32), [0, 1], [1.0, 0.5], top_k=5) == [0, 1]


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

def test_store_roundtrip_preserves_chunks_and_vectors(config, corpus_dir):
    store = VectorStore(config)
    docs = [load_document(p) for p in sorted(corpus_dir.glob("*.md"))]
    store.add_documents(docs, embedder=FakeEmbedder())
    store.save()

    reloaded = VectorStore.load(config)
    assert reloaded.size == store.size
    assert reloaded.embeddings is not None
    assert np.allclose(np.asarray(reloaded.embeddings), np.asarray(store.embeddings))
    assert reloaded.chunks[0].text == store.chunks[0].text


def test_unchanged_documents_are_skipped(config, corpus_dir):
    store = VectorStore(config)
    docs = [load_document(p) for p in sorted(corpus_dir.glob("*.md"))]
    first = store.add_documents(docs, embedder=FakeEmbedder())
    assert first.added_docs == 2

    again = store.add_documents([load_document(p) for p in sorted(corpus_dir.glob("*.md"))], embedder=FakeEmbedder())
    assert again.skipped_docs == 2
    assert again.added_chunks == 0


def test_edited_document_replaces_its_old_chunks(config, corpus_dir):
    store = VectorStore(config)
    target = corpus_dir / "deploy.md"
    store.add_documents([load_document(target)], embedder=FakeEmbedder())
    before = store.size

    target.write_text("Completely different content about billing invoices.", encoding="utf-8")
    report = store.add_documents([load_document(target)], embedder=FakeEmbedder())

    assert report.updated_docs == 1
    assert report.removed_chunks == before
    assert all("billing" in c.text or "invoice" in c.text for c in store.chunks)
    assert len(store.chunks) == len(np.asarray(store.embeddings))


def test_remove_source_keeps_vectors_aligned(config, corpus_dir):
    store = VectorStore(config)
    store.add_documents([load_document(p) for p in sorted(corpus_dir.glob("*.md"))], embedder=FakeEmbedder())
    store.remove_source(str(corpus_dir / "deploy.md"))
    assert store.size == len(np.asarray(store.embeddings))
    assert all("deploy.md" not in c.source for c in store.chunks)


def test_fingerprint_detects_setting_drift(config, corpus_dir):
    store = VectorStore(config)
    store.add_documents([load_document(corpus_dir / "oncall.md")], embedder=FakeEmbedder())
    store.save()

    drifted = Config(index_dir=config.index_dir, chunk_tokens=999)
    warning = VectorStore.load(drifted).check_fingerprint()
    assert warning and "chunk_tokens" in warning


def test_dense_search_without_embeddings_returns_nothing(config):
    assert VectorStore(config).search_dense(np.ones(4, dtype=np.float32), 5) == []


def test_indexing_without_an_embedder_never_leaves_a_half_dense_index(config, corpus_dir):
    """Chunk count and vector rows must stay equal, or deletes misindex."""
    store = VectorStore(config)
    store.add_documents([load_document(corpus_dir / "deploy.md")], embedder=FakeEmbedder())
    assert store.embeddings is not None

    store.add_documents([load_document(corpus_dir / "oncall.md")], embedder=None)
    assert store.embeddings is None  # dropped rather than left partial
    store.remove_source(str(corpus_dir / "deploy.md"))  # must not raise
    assert store.size > 0


def test_chunks_and_vectors_stay_aligned_through_edits(config, corpus_dir):
    store = VectorStore(config)
    embedder = FakeEmbedder()
    store.add_documents([load_document(p) for p in sorted(corpus_dir.glob("*.md"))], embedder=embedder)

    (corpus_dir / "oncall.md").write_text("Rewritten on-call notes, much shorter.", encoding="utf-8")
    store.add_documents([load_document(corpus_dir / "oncall.md")], embedder=embedder)
    store.remove_source(str(corpus_dir / "deploy.md"))
    store.save()

    reloaded = VectorStore.load(config)
    assert reloaded.size == len(np.asarray(reloaded.embeddings))


# --------------------------------------------------------------------------
# retrieval end to end
# --------------------------------------------------------------------------

def _pipeline(config, corpus_dir) -> RAGPipeline:
    rag = RAGPipeline(config, load=False)
    rag._embedder = FakeEmbedder()
    rag.ingest([str(corpus_dir)])
    return rag


def test_ingest_indexes_supported_files_only(config, corpus_dir):
    rag = _pipeline(config, corpus_dir)
    assert rag.store.size > 0
    assert all(not c.source.endswith(".bin") for c in rag.store.chunks)


def test_keyword_query_finds_the_right_document(config, corpus_dir):
    rag = _pipeline(config, corpus_dir)
    hits = rag.search("E_QUOTA_EXCEEDED", top_k=3)
    assert hits and "oncall.md" in hits[0].chunk.source


def test_source_filter_restricts_results(config, corpus_dir):
    rag = _pipeline(config, corpus_dir)
    hits = rag.search("rollback", top_k=5, chunk_filter=make_filter(source="oncall"))
    assert hits
    assert all("oncall" in h.chunk.source for h in hits)


def test_format_filter_restricts_results(config, corpus_dir):
    rag = _pipeline(config, corpus_dir)
    assert rag.search("rollback", top_k=5, chunk_filter=make_filter(fmt="pdf")) == []


def test_retrieval_reports_which_retriever_fired(config, corpus_dir):
    rag = _pipeline(config, corpus_dir)
    hit = rag.search("rollback", top_k=1)[0]
    assert hit.why() != "n/a"
    assert hit.rank == 0


def test_search_on_empty_index_is_empty(config):
    assert RAGPipeline(config, load=False).search("anything") == []


def test_bm25_only_mode_still_retrieves(config, corpus_dir):
    rag = RAGPipeline(config, load=False)
    rag._embedder = FakeEmbedder()
    rag.ingest([str(corpus_dir)])
    # Simulate an index built without dense vectors.
    rag.store.embeddings = None
    rag._embedder = None
    rag._embedder_failed = True
    hits = Retriever(rag.store, None, config).retrieve("rollback", top_k=3)
    assert hits and all(h.dense_score is None for h in hits)


def test_ingest_text_indexes_a_string(config):
    rag = RAGPipeline(config, load=False)
    rag._embedder = FakeEmbedder()
    rag.ingest_text("The maintenance window is Sunday 02:00 UTC.", source="memo://ops")
    assert rag.search("maintenance window", top_k=1)[0].chunk.source == "memo://ops"


# --------------------------------------------------------------------------
# prompt assembly and citations
# --------------------------------------------------------------------------

def _hit(text: str, source: str = "a.md") -> RetrievedChunk:
    from rag.chunking import Chunk

    return RetrievedChunk(
        chunk=Chunk("c1", "d1", source, source, 0, text, 0, len(text), 10),
        score=1.0,
    )


def test_context_is_numbered_and_attributed():
    rendered = format_context([_hit("alpha"), _hit("beta", "b.md")])
    assert 'id="1"' in rendered and 'id="2"' in rendered
    assert "b.md" in rendered and "<context>" in rendered


def test_parse_citations_in_order_of_first_use():
    assert parse_citations("Yes [2], and also [1][2].", 3) == [2, 1]


def test_parse_citations_ignores_out_of_range_markers():
    assert parse_citations("See [9] and [1].", 2) == [1]


def test_parse_citations_on_uncited_answer():
    assert parse_citations("I could not find that in the documents.", 3) == []


def test_answer_without_sources_does_not_call_the_api(config):
    from rag.generate import generate_answer

    answer = generate_answer("anything?", [], config)
    assert answer.sources == []
    assert not answer.grounded


# --------------------------------------------------------------------------
# reranking (stubbed client -- no network)
# --------------------------------------------------------------------------

class _StubBackend:
    """Stands in for a provider backend; returns canned reranker output."""

    name = "stub"
    model = "stub-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, system_parts, messages, max_tokens=None, effort=None):
        from rag.llm import Completion

        self.calls += 1
        return Completion(text=self.reply, model=self.model)


def test_rerank_reorders_by_model_score(monkeypatch, config):
    import rag.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "get_backend", lambda _cfg: _StubBackend("1=2\n2=9\n3=5"))
    candidates = [_hit("first"), _hit("second"), _hit("third")]
    ranked = rerank_mod.llm_rerank("q", candidates, config)
    assert [h.text for h in ranked] == ["second", "third", "first"]
    assert ranked[0].rerank_score == 9.0


def test_rerank_falls_back_to_fused_order_when_unparseable(monkeypatch, config):
    import rag.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "get_backend", lambda _cfg: _StubBackend("I cannot score these."))
    candidates = [_hit("first"), _hit("second")]
    assert [h.text for h in rerank_mod.llm_rerank("q", candidates, config)] == ["first", "second"]


def test_rerank_handles_partial_scores(monkeypatch, config):
    import rag.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "get_backend", lambda _cfg: _StubBackend("2=7"))
    candidates = [_hit("first"), _hit("second")]
    ranked = rerank_mod.llm_rerank("q", candidates, config)
    assert ranked[0].text == "second"  # scored beats unscored
    assert ranked[1].rerank_score is None


def test_rerank_is_a_noop_on_a_single_candidate(config):
    from rag.rerank import llm_rerank

    single = [_hit("only")]
    assert llm_rerank("q", single, config) == single
