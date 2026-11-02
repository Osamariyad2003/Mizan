"""Multi-source ingestion: Word files, PDF text repair, duplicates, settings.

Run with: python -m pytest tests -q
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rag.arabic import fold_presentation_forms, reverse_word_order, word_order_reversed
from rag.config import Config
from rag.loaders import discover, load_document
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

from test_rag import FakeEmbedder

# --------------------------------------------------------------------------
# presentation forms (glyph codes some PDF exporters emit instead of letters)
# --------------------------------------------------------------------------

LAM_INITIAL, FEH_INITIAL, QAF_MEDIAL = chr(0xFEDF), chr(0xFED3), chr(0xFED8)
GHAIN_INITIAL, LAM_ALEF = chr(0xFECF), chr(0xFEFB)


def test_glyph_codes_fold_to_letters():
    assert fold_presentation_forms(LAM_INITIAL) == "ل"
    assert fold_presentation_forms(LAM_ALEF) == "لا"


def test_misdecoded_tanween_ending_is_repaired():
    """'wafqan' exported as waw + feh + qaf + *initial ghain*."""
    broken = "و" + FEH_INITIAL + QAF_MEDIAL + GHAIN_INITIAL + " لاحكام"
    assert fold_presentation_forms(broken) == "وفقا لاحكام"


def test_genuine_word_initial_ghain_is_kept():
    room = GHAIN_INITIAL + "رفة"  # ghurfa, 'room' -- ghain really starts it
    assert fold_presentation_forms(room) == "غرفة"


def test_text_without_glyph_codes_is_untouched():
    text = "½ café ² المادة"
    assert fold_presentation_forms(text) == text


# --------------------------------------------------------------------------
# word-order reversal
# --------------------------------------------------------------------------

def test_reversed_sentence_is_restored():
    assert reverse_word_order(".للضرر تعرض شخص أي :المتضرر") == "المتضرر: أي شخص تعرض للضرر."


def test_reversed_clause_number_is_reattached():
    """'1. tathbut' comes out as 'tathbut. 1' in word-reversed extraction."""
    assert reverse_word_order(".لذلك تثبت. 1") == "1. تثبت لذلك."


def test_dates_survive_word_reversal():
    assert reverse_word_order("2010-04-15 السريان تاريخ") == "تاريخ السريان 2010-04-15"


def test_brackets_are_mirrored():
    assert reverse_word_order("حيثما ( المركبة ) بكلمة") == "بكلمة ( المركبة ) حيثما"


def test_english_lines_are_not_reversed():
    assert reverse_word_order("Section 4 applies.") == "Section 4 applies."


def test_detector_flags_word_reversed_text():
    reversed_page = "\n".join(".للضرر تعرض شخص أي :المتضرر" for _ in range(6))
    assert word_order_reversed(reversed_page)


def test_detector_passes_normal_arabic():
    normal = "\n".join("المتضرر: أي شخص تعرض للضرر." for _ in range(6))
    assert not word_order_reversed(normal)


def test_detector_ignores_english():
    assert not word_order_reversed("\n".join(".Leading dot line" for _ in range(10)))


def test_detector_needs_enough_evidence():
    assert not word_order_reversed(".للضرر تعرض\n.شخص أي")


# --------------------------------------------------------------------------
# Word documents
# --------------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _para(text: str) -> str:
    return f"<w:p><w:r><w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"


def _make_docx(path, body: str, footnotes: str | None = None) -> None:
    doc = f'<w:document xmlns:w="{W_NS}"><w:body>{body}<w:sectPr/></w:body></w:document>'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", doc)
        if footnotes is not None:
            z.writestr("word/footnotes.xml", f'<w:footnotes xmlns:w="{W_NS}">{footnotes}</w:footnotes>')


def test_docx_paragraphs_are_read_in_order(tmp_path):
    path = tmp_path / "law.docx"
    _make_docx(path, _para("المادة 1") + _para("الاسلام دين الدولة.") + _para("المادة 2") + _para("نص ثان."))
    doc = load_document(path)
    assert doc.metadata["format"] == "docx"
    assert doc.text.index("المادة 1") < doc.text.index("الاسلام") < doc.text.index("المادة 2")


def test_docx_heading_stays_attached_to_its_article(tmp_path):
    path = tmp_path / "law.docx"
    _make_docx(path, _para("المادة 1") + _para("الاسلام دين الدولة."))
    # Single newline: the chunker never splits there before a blank line.
    assert "المادة 1\nالاسلام دين الدولة." in load_document(path).text


def test_docx_tables_keep_rows_together(tmp_path):
    path = tmp_path / "t.docx"
    row = "<w:tr><w:tc>{a}</w:tc><w:tc>{b}</w:tc></w:tr>"
    table = "<w:tbl>" + row.format(a=_para("Fee"), b=_para("10 JOD")) + "</w:tbl>"
    _make_docx(path, table)
    assert "Fee | 10 JOD" in load_document(path).text


def test_docx_footnotes_are_included(tmp_path):
    path = tmp_path / "n.docx"
    note = f'<w:footnote w:id="1">{_para("Amended in 2022.")}</w:footnote>'
    separator = f'<w:footnote w:type="separator" w:id="0">{_para("----")}</w:footnote>'
    _make_docx(path, _para("Body text."), footnotes=separator + note)
    text = load_document(path).text
    assert "Amended in 2022." in text and "----" not in text


def test_old_binary_doc_gets_a_helpful_error(tmp_path):
    path = tmp_path / "old.docx"
    path.write_bytes(b"\xd0\xcf\x11\xe0 not a zip")  # OLE header of a legacy .doc
    with pytest.raises(RuntimeError, match="save it as .docx"):
        load_document(path)


def test_word_lock_files_are_not_indexed(tmp_path):
    _make_docx(tmp_path / "real.docx", _para("x"))
    (tmp_path / "~$real.docx").write_bytes(b"lock")
    assert [p.name for p in discover([tmp_path])] == ["real.docx"]


# --------------------------------------------------------------------------
# multiple sources in one index
# --------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path) -> Config:
    return Config(index_dir=tmp_path / "index", chunk_tokens=64, chunk_overlap_tokens=12)


def _rag(config) -> RAGPipeline:
    rag = RAGPipeline(config, load=False)
    rag._embedder = FakeEmbedder()
    return rag


def test_mixed_formats_are_searchable_together(tmp_path, config):
    docs = tmp_path / "docs"
    docs.mkdir()
    _make_docx(docs / "constitution.docx", _para("المادة 2") + _para("الاسلام دين الدولة واللغة العربية لغتها الرسمية."))
    (docs / "notes.md").write_text("Commercial registration must be renewed yearly.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest([str(docs)])
    assert rag.search("اللغة الرسمية", top_k=1)[0].chunk.source.endswith("constitution.docx")
    assert rag.search("commercial registration", top_k=1)[0].chunk.source.endswith("notes.md")


def test_identical_files_are_indexed_once(tmp_path, config):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("law.md", "law (1).md"):
        (docs / name).write_text("The registrar keeps the commercial register.", encoding="utf-8")
    rag = _rag(config)
    report = rag.ingest([str(docs)])
    assert report.added_docs == 1
    assert len(report.duplicates) == 1
    assert len({h.chunk.source for h in rag.search("registrar", top_k=5)}) == 1


def test_duplicate_is_indexed_once_the_original_is_deleted(tmp_path, config):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md"):
        (docs / name).write_text("Same text in both files.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest([str(docs)])
    kept = next(iter(rag.store.sources))

    # Deleted from disk, then dropped from the index: the copy takes over.
    Path(kept).unlink()
    rag.remove(kept)
    report = rag.ingest([str(docs)])
    assert report.added_docs == 1 and not report.duplicates


def test_removing_from_index_alone_keeps_the_duplicate_skipped(tmp_path, config):
    """`rag remove` does not delete the file, so re-ingest re-finds it first."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md"):
        (docs / name).write_text("Same text in both files.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest([str(docs)])
    rag.remove(next(iter(rag.store.sources)))
    report = rag.ingest([str(docs)])
    assert report.added_docs == 1 and len(report.duplicates) == 1


def test_index_remembers_its_chunk_size(tmp_path):
    index = tmp_path / "index"
    first = RAGPipeline(Config(index_dir=index, chunk_tokens=100), load=False)
    first._embedder = FakeEmbedder()
    first.ingest_text("some text " * 50, source="memo://a")

    later = RAGPipeline(Config(index_dir=index))  # default chunk size
    assert later.config.chunk_tokens == 100


def test_explicit_chunk_size_is_not_overridden(tmp_path):
    index = tmp_path / "index"
    first = RAGPipeline(Config(index_dir=index, chunk_tokens=100), load=False)
    first._embedder = FakeEmbedder()
    first.ingest_text("some text " * 50, source="memo://a")

    assert RAGPipeline(Config(index_dir=index, chunk_tokens=300)).config.chunk_tokens == 300
    assert RAGPipeline(Config(index_dir=index), adopt_index_settings=False).config.chunk_tokens != 100


def test_chunk_size_drift_does_not_warn_at_query_time(tmp_path):
    index = tmp_path / "index"
    store = VectorStore(Config(index_dir=index, chunk_tokens=100))
    store.add_documents([], embedder=None)
    store.fingerprint = Config(index_dir=index, chunk_tokens=100).index_fingerprint()
    store.save()

    drifted = VectorStore.load(Config(index_dir=index, chunk_tokens=999))
    assert drifted.check_fingerprint(("embed_model",)) is None  # querying: fine
    assert "chunk_tokens" in drifted.check_fingerprint()        # ingesting: warn


# --------------------------------------------------------------------------
# print artifacts, doubled words, titles
# --------------------------------------------------------------------------

from rag.arabic import official_short_title, undouble_words  # noqa: E402
from rag.loaders import _strip_print_artifacts  # noqa: E402


def test_browser_print_footer_is_removed():
    page = "Article text.\n12/9/2023 10:59 PM qistas.com/ar/print/laws?ss=1&es=2&fs=3&sn=1&ar=1\nMore text."
    assert _strip_print_artifacts(page) == "Article text.\nMore text."


def test_page_counters_are_removed_in_both_languages():
    page = "Body.\nPage 3 of 60\n" + "\u0635\u0641\u062d\u0629 \u0661 \u0645\u0646 \u0666\u0660" + "\nEnd."
    assert _strip_print_artifacts(page) == "Body.\nEnd."


def test_ordinary_lines_with_urls_or_numbers_survive():
    page = "See https://example.com for details.\nArticle 3 of the 1996 law applies to 5 of 10 cases in practice here."
    assert _strip_print_artifacts(page) == page


def test_fused_bold_word_is_collapsed():
    term = "\u0627\u0644\u0639\u0642\u0627\u0631"  # al-aqar
    assert undouble_words(term + term + ": definition") == term + ": definition"


def test_naming_clause_gives_the_official_title():
    text = ("\u064a\u0633\u0645\u0649 \u0647\u0630\u0627 \u0627\u0644\u0642\u0627\u0646\u0648\u0646"
            " ( \u0642\u0627\u0646\u0648\u0646 \u0627\u0644\u0639\u0645\u0644 \u0644\u0633\u0646\u0629 1996 ) \u0648\u064a\u0639\u0645\u0644 \u0628\u0647")
    assert official_short_title(text) == "\u0642\u0627\u0646\u0648\u0646 \u0627\u0644\u0639\u0645\u0644 \u0644\u0633\u0646\u0629 1996"


def test_generic_metadata_title_falls_back_to_filename(tmp_path):
    from rag.loaders import Document, _best_title

    path = tmp_path / "Labour_Law_1996.pdf"
    doc = Document("d", str(path), "Qistas", "no naming clause here")
    assert _best_title(doc, path) == "Labour Law 1996"


def test_descriptive_metadata_title_is_kept(tmp_path):
    from rag.loaders import Document, _best_title

    path = tmp_path / "scan001.pdf"
    doc = Document("d", str(path), "Annual Report of the Ministry 2022", "text")
    assert _best_title(doc, path) == "Annual Report of the Ministry 2022"


# --------------------------------------------------------------------------
# deleted and renamed files
# --------------------------------------------------------------------------

def test_deleted_file_is_removed_on_next_ingest(tmp_path, config):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.md").write_text("Labour law working hours.", encoding="utf-8")
    (docs / "gone.md").write_text("Penal code theft penalties.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest([str(docs)])

    (docs / "gone.md").unlink()
    report = rag.ingest([str(docs)])
    assert [Path(s).name for s in report.pruned] == ["gone.md"]
    assert all("gone.md" not in c.source for c in rag.store.chunks)
    assert not rag.search("theft penalties", top_k=5) or all(
        "gone.md" not in h.chunk.source for h in rag.search("theft penalties", top_k=5)
    )


def test_renamed_file_is_reindexed_under_its_new_name(tmp_path, config):
    """Regression: the rename was mistaken for a duplicate of the old name."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "old name.md").write_text("Real estate ownership rules.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest([str(docs)])

    (docs / "old name.md").rename(docs / "new name.md")
    report = rag.ingest([str(docs)])
    assert not report.duplicates
    assert [Path(s).name for s in rag.store.sources] == ["new name.md"]


def test_pruning_is_scoped_to_the_ingested_folder(tmp_path, config):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "one.md").write_text("First folder text.", encoding="utf-8")
    (b / "two.md").write_text("Second folder text.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest([str(a), str(b)])

    (b / "two.md").unlink()
    report = rag.ingest([str(a)])  # only folder a
    assert not report.pruned
    assert any("two.md" in s for s in rag.store.sources)


def test_text_sources_are_never_pruned(tmp_path, config):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Some text.", encoding="utf-8")
    rag = _rag(config)
    rag.ingest_text("Maintenance window is Sunday.", source="memo://ops")
    rag.ingest([str(docs)])
    assert "memo://ops" in rag.store.sources


# --------------------------------------------------------------------------
# extraction cache
# --------------------------------------------------------------------------

def test_extraction_is_reused_for_identical_bytes(tmp_path, monkeypatch):
    import rag.loaders as loaders

    path = tmp_path / "law.md"
    path.write_text("Article text that is expensive to extract.", encoding="utf-8")
    cache = tmp_path / "cache"
    first = load_document(path, cache_dir=cache)
    assert len(list(cache.glob("*.json"))) == 1

    def exploding_loader(_path):
        raise AssertionError("loader must not run on a cache hit")

    monkeypatch.setitem(loaders.LOADERS, ".md", exploding_loader)
    second = load_document(path, cache_dir=cache)
    assert second.text == first.text and second.source == str(path)


def test_changed_file_is_extracted_again(tmp_path):
    path = tmp_path / "law.md"
    cache = tmp_path / "cache"
    path.write_text("Version one.", encoding="utf-8")
    load_document(path, cache_dir=cache)
    path.write_text("Version two, edited.", encoding="utf-8")
    assert load_document(path, cache_dir=cache).text == "Version two, edited."


def test_renamed_file_reuses_cache_but_takes_its_new_identity(tmp_path):
    cache = tmp_path / "cache"
    old = tmp_path / "old.md"
    old.write_text("Same bytes.", encoding="utf-8")
    load_document(old, cache_dir=cache)
    new = old.rename(tmp_path / "new.md")
    doc = load_document(new, cache_dir=cache)
    assert doc.source == str(new)
    assert doc.title == "new.md"  # regression: the cached title was the old filename


# --------------------------------------------------------------------------
# answer context and local reranking
# --------------------------------------------------------------------------

def test_answer_context_fills_a_token_budget(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(8):
        (docs / f"n{i}.md").write_text(f"Working hours rule number {i}. " * 20, encoding="utf-8")
    cfg = Config(index_dir=tmp_path / "idx", context_tokens=150, top_k=8, rerank=False)
    rag = _rag(cfg)
    rag.ingest([str(docs)])
    chosen = rag.context_for("working hours rule", top_k=8)
    assert chosen
    assert sum(h.chunk.token_count for h in chosen) <= 150 or len(chosen) == 1


def test_answer_context_always_keeps_the_top_hit(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "big.md").write_text("Maternity leave is ten weeks. " * 80, encoding="utf-8")
    cfg = Config(index_dir=tmp_path / "idx", context_tokens=10, rerank=False)
    rag = _rag(cfg)
    rag.ingest([str(docs)])
    assert len(rag.context_for("maternity leave")) == 1


def test_local_rerank_reorders_by_cross_encoder_score(monkeypatch, config):
    import rag.crossencoder as ce
    from test_rag import _hit

    class FakeReranker:
        def __init__(self, _config):
            pass

        def score(self, query, passages):
            return [float(len(p)) for p in passages]  # longest passage wins

    monkeypatch.setattr(ce, "CrossEncoderReranker", FakeReranker)
    ranked = ce.local_rerank("q", [_hit("aa"), _hit("aaaa"), _hit("a")], config)
    assert [h.text for h in ranked] == ["aaaa", "aa", "a"]
    assert ranked[0].rerank_score == 4.0


def test_local_rerank_only_scores_the_candidate_pool(monkeypatch, config):
    import rag.crossencoder as ce
    from test_rag import _hit

    seen = []

    class FakeReranker:
        def __init__(self, _config):
            pass

        def score(self, query, passages):
            seen.extend(passages)
            return list(range(len(passages)))

    monkeypatch.setattr(ce, "CrossEncoderReranker", FakeReranker)
    config.rerank_candidates = 2
    ranked = ce.local_rerank("q", [_hit("x"), _hit("y"), _hit("z")], config)
    assert len(seen) == 2
    assert ranked[-1].text == "z"  # beyond the pool: keeps its place at the end
