"""Line-by-line notes for rag/crossencoder.py."""

SUMMARY = (
    "The reranker that produced the largest single quality gain in this system: a model that "
    "reads the question and a passage together and scores how well one answers the other."
)

BLOCKS = [
    (1, 12,
     "The docstring explains why a second model is worth its cost. Retrieval scores the question "
     "and the passage <i>separately</i> - one embedding each, or counted keyword overlap - which "
     "is what makes searching thousands of chunks fast, and also what makes it unable to judge "
     "whether a passage actually answers the question. A cross-encoder reads the pair together, "
     "which is far more accurate and far too slow to run over a corpus, so it reorders only the "
     "few dozen candidates fusion already found. On this corpus that moved the correct article "
     "into first place for 47% to 76% of the evaluation questions."),
    (13, 21, "Imports; everything heavy is imported inside the loader below."),
    (22, 33,
     "Loading the model, cached on its arguments so a repeated query reuses the loaded weights "
     "instead of paying several seconds again. <b>27-32</b> silence the loading chatter, in its "
     "own try because it is cosmetic."),
    (34, 45,
     "<b>34-35</b> resolve the device. <b>38</b> switches to evaluation mode. <b>39-42</b> are "
     "the line that made this practical on a CPU: int8 quantisation of the linear layers, which "
     "is several times faster at a negligible cost in ranking quality. Note <b>39</b> applies it "
     "only on CPU - on a GPU the full-precision model is already fast and quantising would just "
     "lose accuracy."),
    (46, 54,
     "The reranker object, which is only a thin wrapper around the cached load - so constructing "
     "one per query is free after the first."),
    (55, 65,
     "Scoring. <b>61-63</b> sort by length before batching, for the same reason the embedder "
     "does: every batch is padded to its longest member, and these articles range from one line "
     "to a full page, so unsorted batches spend most of their compute on padding."),
    (66, 76,
     "The pair encoding. <b>70-71</b> pass the question and the passage as two separate "
     "sequences, which is what a cross-encoder expects and what lets it attend across them. "
     "<b>73</b> is a small but decisive detail: truncation is restricted to the second sequence, "
     "so a long passage is cut and the question never is. Truncating the question would be "
     "scoring passages against half a question."),
    (77, 83,
     "<b>77-78</b> run inference without gradient tracking. <b>79-80</b> handle both model "
     "shapes: some emit a single relevance logit, others a pair of class logits, in which case "
     "the last column is the relevant one. <b>81-82</b> write each score back to its original "
     "position, undoing the length sort."),
    (84, 91,
     "The public function. The docstring records the policy: only the configured number of "
     "candidates is scored, and anything beyond that keeps its fused order <i>after</i> the "
     "reranked ones - so the pool is improved and the tail is never lost."),
    (92, 100,
     "<b>96</b> scores <font name='Mono' size='8'>indexed_text</font>, header included, so the "
     "reranker sees \"Labour Law - Article 61\" above the passage and can use it. <b>99</b> "
     "sorts by the new score, and <b>100</b> appends the unscored tail, which is what keeps the "
     "result count unchanged regardless of how deep the reranker reached."),
]
