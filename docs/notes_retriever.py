"""Line-by-line notes for rag/retriever.py."""

SUMMARY = (
    "Where a question becomes a ranked list of passages. Four stages: search both indexes, "
    "fuse their rankings, drop near-duplicates, and optionally rerank what survives."
)

BLOCKS = [
    (1, 14,
     "The docstring is the algorithm. The important argument is <b>6-9</b>: fusion uses "
     "<i>ranks</i>, not scores. A cosine similarity and a BM25 score live on scales that cannot "
     "be compared - one is bounded, the other unbounded and corpus-dependent - so any attempt to "
     "weight the raw numbers needs calibration that goes stale the moment the corpus changes. "
     "Ranks have no such problem, which is why the fusion step is three lines long and never "
     "needs tuning."),
    (15, 27, "Imports: the chunk type, the settings, and the store that holds both indexes."),
    (28, 39,
     "The result record. Each hit carries not just its final score but <i>where each retriever "
     "put it</i> - <b>32-35</b> - which is what makes the system debuggable: when an answer is "
     "wrong you can see whether the passage was found by meaning, by keywords, by both, or "
     "promoted by the reranker."),
    (40, 47, "Convenience accessors that spare callers from reaching through to the chunk."),
    (48, 58,
     "The explanation string behind <font name='Mono' size='8'>--why</font>: \"dense #3 (0.812), "
     "bm25 #1 (14.2), rerank 9.5\". Reading a few of these answers questions about retrieval "
     "quality that no amount of staring at the final ranking will."),
    (59, 71,
     "Serialisation for JSON output, so the retrieval step can be scripted and evaluated "
     "independently of the answer step - which is exactly how the evaluation harness uses it."),
    (72, 89,
     "Filters by source path and format. <b>82</b> does a case-insensitive substring match, so "
     "<font name='Mono' size='8'>--source labour</font> finds the Labour Law without the full "
     "path. <b>78-79</b> return <font name='Mono' size='8'>None</font> rather than a "
     "predicate that always passes, which lets the caller skip the filtering work entirely."),
    (90, 102,
     "Reciprocal Rank Fusion. The docstring explains the constant: <b>k</b> damps the "
     "contribution of top ranks, so one retriever's single confident hit cannot dominate a "
     "document that both retrievers rank well. The last sentence is a caution against treating "
     "the equal default weights as a finding."),
    (103, 109,
     "The whole fusion, in five lines: each document's score is the sum over lists of its "
     "weight divided by its rank plus the constant. <b>107</b> adds one because ranks here are "
     "zero-based and the formula is defined on one-based ranks."),
    (110, 118,
     "Min-max scaling to [0, 1]. <b>114-115</b> handle the degenerate case where every value is "
     "the same - returning ones rather than dividing by zero, which keeps a tie from producing "
     "NaNs that would propagate through the selection below."),
    (119, 132,
     "Maximal Marginal Relevance. Two design notes in the docstring, both about correctness "
     "rather than performance. Relevance comes from the <i>fused</i> ranking, not raw cosine, so "
     "the first pick is always the top hybrid hit and diversification only operates below it. "
     "And both terms are normalised, which is what makes the lambda parameter mean what its name "
     "suggests instead of being an arbitrary dial."),
    (133, 139,
     "<b>133-134</b> short-circuit when there is nothing to choose between - with fewer "
     "candidates than the target, every one is selected. <b>137</b> gathers just the candidates' "
     "vectors, so the similarity work below is over a few dozen rows rather than the whole "
     "corpus."),
    (140, 146,
     "The first pick is pure relevance: with nothing selected yet there is no redundancy to "
     "measure, so the top fused hit is always taken first."),
    (147, 155,
     "The MMR step. <b>149</b> defines redundancy as similarity to the <i>most</i> similar "
     "already-selected chunk - not the average, which would let a candidate that duplicates one "
     "selection but differs from the rest slip through. <b>150</b> trades relevance against that "
     "redundancy. The practical effect on this corpus is that three overlapping windows of the "
     "same article no longer occupy three of the twelve context slots."),
    (156, 167, "The retriever holds the store, the embedder and the settings. Nothing else."),
    (168, 179,
     "The query path begins. <b>177-178</b> return immediately for an empty index - which also "
     "avoids loading a 2 GB embedding model to search nothing, a short-circuit that took the "
     "test suite from seventeen seconds to one."),
    (180, 185,
     "Two widening steps. <b>180</b> searches at least four times deeper than the number of "
     "results wanted, because fusion and diversification both need candidates to work with. "
     "<b>182-184</b> widen further when a filter is active, and the comment states the bug this "
     "prevents: filtering after truncation silently shrinks the result set, so a filtered search "
     "would return three results where it should return twelve."),
    (186, 193,
     "Both searches run. <b>188</b> requires both an embedder and an actual matrix, so a "
     "keyword-only index works with no special-casing anywhere else. <b>192</b> always runs - "
     "BM25 costs a fraction of a millisecond, so there is no reason to make it conditional."),
    (194, 202,
     "Filtering, then truncation to the real candidate depth - in that order, which is the point "
     "of the widening above. <b>200-201</b> return early when nothing survived."),
    (203, 207,
     "Four lookup tables recording each retriever's rank and score per chunk. These feed both "
     "the fusion and the explanation string, which is why the original scores are preserved "
     "rather than discarded once ranks are known."),
    (208, 216,
     "Assembling the lists to fuse. Each is added only if it has hits, so an empty dense result "
     "does not contribute an empty list that would shift the weights."),
    (217, 220, "Fuse, sort, and the ranked candidate list is ready."),
    (221, 233,
     "Diversification. <b>221</b> runs it only when there are vectors to measure similarity with "
     "and more candidates than needed. <b>222</b> limits MMR to a pool rather than the whole "
     "candidate list, since diversifying the hundredth-best hit is wasted work. <b>230-232</b> "
     "are a small but important detail: whatever MMR dropped is kept as a tail, so the result "
     "list is always filled to the requested size rather than coming back short because "
     "everything looked redundant."),
    (234, 237,
     "The reranking pool. The comment states the constraint exactly: a reranker can only promote "
     "what it is shown, so the pool is deliberately deeper than the number of results - "
     "promoting the twentieth candidate to first is the entire value of this step."),
    (238, 251,
     "Building the result objects, carrying every score and rank collected along the way."),
    (252, 261,
     "Choosing a reranker. Both imports are inside the branch, so neither torch nor the provider "
     "SDK is loaded by a query that does not rerank. <b>253</b> keeps an old configuration value "
     "working as an alias, which costs one condition and avoids breaking anyone's existing "
     "settings."),
    (262, 267,
     "Final trimming. <b>263-264</b> apply a minimum score only if one was configured - it is "
     "off by default, because a fused RRF score has no natural threshold. <b>265-266</b> "
     "renumber the ranks so they describe the list actually returned."),
]
