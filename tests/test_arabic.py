"""Arabic-language handling. Run with: python -m pytest tests -q

Plain Arabic words are written literally so the cases are readable; combining
marks and presentation forms are built with chr() because they are invisible
or ambiguous in source.
"""

from __future__ import annotations

import pytest

from rag.arabic import (
    average_word_length,
    has_arabic,
    merge_word_breaks,
    normalize_for_search,
    spacing_looks_broken,
    visual_to_logical,
)
from rag.bm25 import BM25Index, tokenize

FATHA, DAMMA, SHADDA = chr(0x064E), chr(0x064F), chr(0x0651)
LAM_ALEF_LIGATURE = chr(0xFEFB)  # presentation form, as emitted by some PDFs


# --------------------------------------------------------------------------
# spelling-variant folding
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, why",
    [
        ("الإدارة", "hamza under alef vs plain alef"),
        ("الادارة", "exact spelling"),
        ("إدارة", "no definite article"),
        ("ادارة", "no article, no hamza"),
    ],
)
def test_administration_variants_all_find_the_document(query, why):
    docs = [
        "تلتزم الادارة بتطبيق احكام هذا القانون على جميع الموظفين",
        "يعاقب بالحبس كل من يخالف المادة الخامسة من النظام",
    ]
    hits = BM25Index(docs).search(query, top_k=2)
    assert hits and hits[0][0] == 0, why


def test_article_fused_with_conjunction_matches_bare_noun():
    assert tokenize("والقانون") == tokenize("القانون") == tokenize("قانون")


def test_article_fused_with_preposition_matches():
    assert tokenize("للمادة") == tokenize("المادة") == tokenize("مادة")


def test_diacritics_do_not_split_words():
    """Regression: combining marks used to act as word breaks."""
    vowelled = "ت" + DAMMA + "فر" + FATHA + "ض"
    assert tokenize(vowelled) == ["تفرض"]


def test_shadda_and_tatweel_are_ignored():
    assert tokenize("مح" + SHADDA + "كمة") == tokenize("محكمة")
    assert tokenize("قـــانون") == tokenize("قانون")


def test_arabic_indic_digits_match_western_digits():
    assert tokenize("المادة ١٢") == tokenize("المادة 12")


def test_ta_marbuta_and_ha_are_equivalent():
    assert tokenize("مادة") == tokenize("ماده")


def test_alef_maqsura_and_ya_are_equivalent():
    assert tokenize("مستشفى") == tokenize("مستشفي")


def test_waw_that_is_a_root_letter_is_preserved():
    """Stripping a leading waw would break ministry/document/minister."""
    assert tokenize("الوزارة") == tokenize("وزارة")
    assert tokenize("الوثيقة") == tokenize("وثيقة")


def test_short_legal_terms_survive_article_stripping():
    assert tokenize("الحق") == ["حق"]


def test_presentation_forms_fold_to_base_letters():
    assert normalize_for_search(LAM_ALEF_LIGATURE) == "لا"


def test_negation_is_not_a_stopword():
    """In legal text 'la yajuz' (not permitted) inverts the meaning."""
    assert "لا" in tokenize("لا يجوز")


def test_english_tokenization_is_unchanged():
    assert tokenize("The Tenant shall pay rent monthly.") == ["tenant", "shall", "pay", "rent", "monthly"]


def test_mixed_language_text():
    tokens = tokenize("Article 5 من القانون")
    assert "article" in tokens and "5" in tokens and "قانون" in tokens


def test_non_arabic_text_passes_through_normalizer():
    text = "café résumé naïve"
    assert normalize_for_search(text) == text
    assert not has_arabic(text)


# --------------------------------------------------------------------------
# visual-order PDF repair
# --------------------------------------------------------------------------

def test_visual_order_line_is_restored():
    assert visual_to_logical("المادة"[::-1]) == "المادة"


def test_numbers_keep_their_order_when_line_is_reversed():
    visual = "12 " + "المادة"[::-1]
    assert visual_to_logical(visual) == "المادة 12"


def test_decimal_numbers_survive_reversal():
    visual = "3.5 " + "نسبة"[::-1]
    assert visual_to_logical(visual) == "نسبة 3.5"


def test_latin_lines_are_untouched_by_reversal():
    assert visual_to_logical("Section 12(b)") == "Section 12(b)"


def test_merge_takes_letters_from_plain_and_spaces_from_layout():
    plain = "اليومالاول"           # correct letters, glued
    layout = "اليوم االول"         # spaced, lam-alef ligature flipped
    assert merge_word_breaks(plain, layout) == "اليوم الاول"


def test_merge_preserves_line_breaks():
    assert merge_word_breaks("ابجد", "اب\nجد") == "اب\nجد"


def test_merge_refuses_texts_that_differ_beyond_ligature_swaps():
    assert merge_word_breaks("قانون", "قاموس") is None


def test_merge_refuses_different_lengths():
    assert merge_word_breaks("قانون", "قانو ن ي") is None


def test_glued_extraction_is_detected():
    glued = "اليومالاولتعريفبمقدمالدورة " * 3 + "التعليماتالخاصةبحضورالدورات " * 3
    assert spacing_looks_broken(glued)


def test_normal_arabic_is_not_flagged():
    normal = "تلتزم الادارة بتطبيق احكام هذا القانون على جميع الموظفين في الوزارة"
    assert not spacing_looks_broken(normal)
    assert 3 < average_word_length(normal) < 7


def test_english_text_is_never_flagged():
    assert not spacing_looks_broken("Supercalifragilisticexpialidocious " * 10)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_variant_query_retrieves_the_right_article():
    corpus = [
        "المادة ١: تسري أحكام هذا القانون على جميع الموظفين في الإدارات الحكومية.",
        "المادة ٢: يعاقب بالغرامة كل من يخالف أحكام المادة الأولى.",
        "المادة ٣: يعمل بهذا القانون من تاريخ نشره في الجريدة الرسمية.",
    ]
    idx = BM25Index(corpus)
    # Different spelling of administrations, digit style, and no article.
    assert idx.search("الادارات", top_k=1)[0][0] == 0
    assert idx.search("المادة 2 غرامة", top_k=1)[0][0] == 1
    assert idx.search("الجريدة الرسميه", top_k=1)[0][0] == 2
