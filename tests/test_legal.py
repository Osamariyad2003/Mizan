"""Article detection and article-aligned chunking.

Run with: python -m pytest tests -q
"""

from __future__ import annotations

from rag.chunking import chunk_document
from rag.config import Config
from rag.legal import find_articles
from rag.loaders import Document

MADDA = "المادة"


def _law(n: int, style: str = "bare") -> str:
    parts = ["قانون تجريبي", ""]
    for i in range(1, n + 1):
        if style == "bare":
            parts.append(f"{MADDA} {i}\nنص المادة رقم {i} ويتضمن حكما قانونيا خاصا بها.")
        elif style == "db":
            parts.append(f"- {MADDA} {i} احكام عامة تاريخ السريان 2010-04-15\nالباب الاول\nنص المادة رقم {i}.")
        elif style == "inline":
            parts.append(f"{MADDA} {i}- نص المادة رقم {i} على السطر نفسه.")
    return "\n\n".join(parts)


def test_bare_headings_are_found_in_order():
    arts = find_articles(_law(8))
    assert [a.number for a in arts] == list(range(1, 9))


def test_database_headings_capture_topic_date_and_division():
    arts = find_articles(_law(6, "db"))
    assert len(arts) == 6
    assert arts[0].topic == "احكام عامة"
    assert arts[0].effective == "2010-04-15"
    assert arts[0].division == "الباب الاول"


def test_inline_headings_start_the_body_on_the_same_line():
    text = _law(6, "inline")
    arts = find_articles(text)
    assert len(arts) == 6
    assert text[arts[2].body_start :].startswith("نص المادة رقم 3")


def test_a_reference_to_another_article_is_not_a_heading():
    """'Article 70 of this law ...' at a line start must not split the text."""
    text = _law(6) + f"\n{MADDA} 70- من هذا القانون تطبق على ما سبق."
    assert [a.number for a in find_articles(text)] == list(range(1, 7))


def test_out_of_sequence_numbers_are_rejected():
    body = _law(6).replace(f"{MADDA} 4\n", f"{MADDA} 400\n")
    numbers = [a.number for a in find_articles(body)]
    assert 400 not in numbers
    assert numbers == [1, 2, 3, 5, 6]


def test_bis_articles_are_accepted():
    text = _law(3) + f"\n\n{MADDA} 3 مكررة\nنص مكرر.\n\n{MADDA} 4\nنص.\n\n{MADDA} 5\nنص."
    labels = [a.label for a in find_articles(text)]
    assert "3 مكررة" in labels


def test_non_legal_text_has_no_articles():
    assert find_articles("Just an ordinary report.\n\nWith paragraphs.") == []


def test_articles_cover_the_whole_text_after_the_preamble():
    text = _law(7)
    arts = find_articles(text)
    assert arts[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(arts, arts[1:]))


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def _doc(text: str) -> Document:
    return Document("d", "law.docx", "قانون تجريبي", text, metadata={"format": "docx"})


def test_each_article_is_its_own_chunk_without_packing(tmp_path):
    cfg = Config(index_dir=tmp_path, chunk_tokens=500, article_pack_tokens=0)
    chunks = [c for c in chunk_document(_doc(_law(6)), cfg) if c.metadata.get("article")]
    assert [c.metadata["article"] for c in chunks] == [str(i) for i in range(1, 7)]


def test_packing_groups_short_articles_without_splitting_any(tmp_path):
    # Each sample article is ~38 tokens, so 150 packs three or four together.
    cfg = Config(index_dir=tmp_path, chunk_tokens=500, article_pack_tokens=150)
    chunks = [c for c in chunk_document(_doc(_law(12)), cfg) if c.metadata.get("article")]
    assert 1 < len(chunks) < 12
    covered = [label for c in chunks for label in c.metadata["articles"]]
    assert covered == [str(i) for i in range(1, 13)]  # every article, once, in order


def test_chunk_location_names_the_article(tmp_path):
    cfg = Config(index_dir=tmp_path, chunk_tokens=500, article_pack_tokens=0)
    chunk = next(c for c in chunk_document(_doc(_law(6)), cfg) if c.metadata.get("article") == "3")
    assert chunk.location == f"قانون تجريبي — {MADDA} 3"


def test_header_carries_title_and_article_into_the_index(tmp_path):
    cfg = Config(index_dir=tmp_path, chunk_tokens=500, article_pack_tokens=0)
    chunk = next(c for c in chunk_document(_doc(_law(6)), cfg) if c.metadata.get("article") == "3")
    assert chunk.indexed_text.startswith(f"قانون تجريبي — {MADDA} 3")
    assert chunk.text.startswith(f"{MADDA} 3")  # stored text is the source, unchanged


def test_long_article_is_windowed_but_keeps_its_label(tmp_path):
    long_body = " ".join(["حكم"] * 900)
    text = _law(5) + f"\n\n{MADDA} 6\n{long_body}\n\n{MADDA} 7\nنص قصير."
    cfg = Config(index_dir=tmp_path, chunk_tokens=200, article_pack_tokens=0)
    pieces = [c for c in chunk_document(_doc(text), cfg) if c.metadata.get("article") == "6"]
    assert len(pieces) > 1
    assert all(p.header.endswith(f"{MADDA} 6") for p in pieces)


def test_article_chunks_can_be_switched_off(tmp_path):
    cfg = Config(index_dir=tmp_path, chunk_tokens=500, article_chunks=False)
    assert not any(c.metadata.get("article") for c in chunk_document(_doc(_law(6)), cfg))


def test_chunk_offsets_still_map_back_to_the_source(tmp_path):
    cfg = Config(index_dir=tmp_path, chunk_tokens=500, article_pack_tokens=60)
    doc = _doc(_law(10))
    for chunk in chunk_document(doc, cfg):
        assert doc.text[chunk.char_start : chunk.char_end].strip() == chunk.text
