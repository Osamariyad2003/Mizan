"""BM25 (Okapi) lexical retrieval.

Implemented directly rather than pulled in as a dependency: the whole algorithm
is about sixty lines, and owning it means the tokenizer can be shared with the
rest of the pipeline and tuned for the corpus.

BM25 earns its place next to dense retrieval because embeddings are weak exactly
where keyword search is strong -- rare proper nouns, identifiers, error codes,
version numbers, and any term the embedding model never saw in training.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

from .arabic import has_arabic, light_stem, normalize_for_search

# Unicode-aware: keeps Arabic, Cyrillic, CJK and accented Latin word characters.
# Arabic text is normalized *before* this runs -- see tokenize().
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Deliberately small. Aggressive stopword removal hurts phrase-like queries, and
# BM25's IDF term already discounts words that appear everywhere. Negations
# (la, lam, lan) are kept on purpose: in legal text they carry the meaning.
_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
# Arabic function words as escapes (see arabic.py for why), folded with the same
# normalizer as the text so their spelling variants are caught too.
_ARABIC_STOPWORDS = (
    "\u0641\u064A", "\u0645\u0646", "\u0639\u0644\u0649", "\u0625\u0644\u0649",  # fi, min, ala, ila
    "\u0639\u0646", "\u0647\u0630\u0627", "\u0647\u0630\u0647", "\u0630\u0644\u0643",  # an, hadha, hadhihi, dhalika
    "\u062A\u0644\u0643", "\u0647\u0648", "\u0647\u064A", "\u0623\u0648",  # tilka, huwa, hiya, aw
    "\u0623\u0646", "\u0625\u0646", "\u0627\u0644\u062A\u064A", "\u0627\u0644\u0630\u064A",  # an, inna, allati, alladhi
    "\u0645\u0639", "\u0628\u064A\u0646", "\u0642\u062F", "\u0639\u0646\u062F",  # ma, bayna, qad, inda
)
STOPWORDS = _ENGLISH_STOPWORDS | {normalize_for_search(w) for w in _ARABIC_STOPWORDS}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lower-case word tokens, with Arabic spelling variants folded together.

    Normalization must come first: Arabic diacritics are combining marks, and
    the word regex would otherwise split a vowelled word at every one of them.
    """
    text = normalize_for_search(text)
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if has_arabic(text):  # English-only text skips the per-token pass
        tokens = [light_stem(t) for t in tokens]
    return tokens


class BM25Index:
    """Okapi BM25 over an in-memory corpus.

    ``k1`` controls term-frequency saturation and ``b`` the length
    normalisation; the defaults are the standard ones and are a reasonable
    starting point for mixed prose and code.
    """

    def __init__(self, corpus: Sequence[str] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_len = 0.0
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        if corpus is not None:
            self.build(corpus)

    def build(self, corpus: Sequence[str]) -> "BM25Index":
        self.doc_count = len(corpus)
        self.doc_lens = []
        self.postings = {}
        if self.doc_count == 0:
            self.avg_len = 0.0
            self.idf = {}
            return self

        doc_freq: Counter[str] = Counter()
        for idx, text in enumerate(corpus):
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((idx, tf))
            doc_freq.update(counts.keys())

        self.avg_len = sum(self.doc_lens) / self.doc_count or 1.0
        # Robertson/Sparck-Jones IDF with the +1 smoothing that keeps very
        # common terms at a small positive weight instead of a negative one.
        self.idf = {
            term: math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` for the best ``top_k`` matches."""
        if self.doc_count == 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = {}
        for term in set(query_terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in postings:
                norm = 1 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_len)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (
                    tf * (self.k1 + 1) / (tf + self.k1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def __len__(self) -> int:
        return self.doc_count
