"""Line-by-line notes for rag/bm25.py."""

SUMMARY = (
    "Keyword search, written out in full rather than imported. This is the half of retrieval "
    "that finds an exact article number or a rare term the embedding model never saw."
)

BLOCKS = [
    (1, 10,
     "Two justifications. <b>3-5</b>: the algorithm is sixty lines, and owning it means the "
     "tokenizer can be shared with the rest of the pipeline - which matters here, because the "
     "Arabic folding has to be applied identically to documents and queries, and a third-party "
     "BM25 would tokenise its own way. <b>7-9</b>: why keyword search still earns its place next "
     "to embeddings - it is strong exactly where they are weak, on rare proper nouns, "
     "identifiers, numbers and anything outside the embedding model's vocabulary."),
    (11, 20,
     "Imports. The three functions from <font name='Mono' size='8'>arabic</font> are what make "
     "this index work on Arabic at all."),
    (21, 24,
     "The token pattern: everything that is not a non-word character, with underscore excluded. "
     "Written this way rather than as a letter range, it keeps Arabic, Cyrillic, CJK and "
     "accented Latin without naming any of them. <b>22</b> records the ordering dependency - "
     "Arabic must be normalised before this pattern is applied."),
    (25, 32,
     "The stopword list, and an argument for keeping it short. Aggressive stopword removal hurts "
     "phrase-like queries, and BM25's own IDF term already discounts words that appear "
     "everywhere, so the list only holds words with no discriminating power at all. The comment "
     "notes one deliberate inclusion-by-omission: Arabic negations are kept, because in legal "
     "text the difference between \"shall\" and \"shall not\" is the entire meaning."),
    (33, 43,
     "Arabic function words, as escapes with transliterations in the comments so the list can be "
     "reviewed. <b>42</b> is the important line: the Arabic words are passed through the same "
     "normaliser as the text before being added to the set, so a stopword written with a hamza "
     "in the document still matches the folded entry here."),
    (44, 58,
     "The tokenizer, shared by indexing and querying. <b>51</b> normalises first, and the "
     "docstring says why it cannot be done later: Arabic diacritics are combining marks, so the "
     "word pattern on <b>23</b> would treat each one as a word break and shatter a vowelled word "
     "into fragments. <b>55-56</b> run the stemmer only when the text contains Arabic, so an "
     "English corpus does not pay for a per-token function call that can never change anything."),
    (59, 67,
     "The index class. The docstring names the two tuning constants: <font name='Mono' size='8'>"
     "k1</font> controls how quickly repeated terms stop adding score, and "
     "<font name='Mono' size='8'>b</font> how strongly long documents are penalised."),
    (68, 78,
     "The data structures. <b>74</b> is the inverted index - term to a list of (document, count) "
     "pairs - which is what makes search proportional to the number of documents containing the "
     "query's terms rather than to the size of the corpus. <b>75</b> caches each term's inverse "
     "document frequency, computed once at build time."),
    (79, 87,
     "Building. <b>83-86</b> handle the empty corpus explicitly, which is what keeps an empty "
     "index from dividing by zero when the average document length is computed below."),
    (88, 96,
     "The indexing pass. <b>90-92</b> tokenise each document and record its length - the length "
     "is needed at search time for the normalisation term. <b>93-94</b> append to each term's "
     "posting list. <b>95</b> counts <i>distinct</i> terms per document, which is the definition "
     "of document frequency; counting raw occurrences here would make common words look rarer "
     "than they are."),
    (97, 105,
     "<b>97</b> computes the average length, with the <font name='Mono' size='8'>or 1.0</font> "
     "guarding a corpus of empty strings. <b>100-103</b> are the IDF formula, and the comment "
     "explains the +1: the classical form goes <i>negative</i> for a term appearing in more than "
     "half the documents, which would mean a document is penalised for containing a query term. "
     "The smoothing keeps such terms at a small positive weight instead."),
    (106, 113,
     "Searching. <b>108-112</b> return early for an empty index and for a query that tokenises "
     "to nothing - which happens when a query is entirely stopwords."),
    (114, 125,
     "The scoring loop, which is BM25 itself. <b>115</b> iterates over the <i>set</i> of query "
     "terms, so a word repeated in the question does not count twice. <b>121</b> is the length "
     "normalisation - a long document needs more occurrences to score the same. <b>122-124</b> "
     "are the saturation term: the second occurrence of a word adds much less than the first, "
     "which is what stops a document that repeats one word from beating a document that actually "
     "covers the topic."),
    (126, 130,
     "Rank and return. The whole search is a dictionary scan and a sort - on this corpus it runs "
     "in a fraction of a millisecond, which is why the fused pipeline can afford to run it on "
     "every query alongside the vector search."),
]
