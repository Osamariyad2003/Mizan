"""Line-by-line notes for rag/rerank.py."""

SUMMARY = (
    "The alternative reranker: ask the language model to score the candidates instead of "
    "running a local cross-encoder. Off by default, and written so that it can never make "
    "retrieval worse than not using it."
)

BLOCKS = [
    (1, 11,
     "The docstring states the division of labour. Hybrid fusion is good at <i>recall</i> - "
     "getting the right passage into the top thirty - and only approximately right about order. "
     "<b>8-10</b> give the honest trade: this costs an API call per query, so the local "
     "cross-encoder is the default and this exists for when spending a call is preferable to "
     "spending CPU time."),
    (12, 21, "Imports, including the provider abstraction so this works on either provider."),
    (22, 28,
     "The scoring prompt. It is short on purpose. The output format is fixed to one line per "
     "excerpt so the response can be parsed by a regular expression, the scale is anchored at "
     "three points rather than left to the model's judgement, and the last two sentences fence "
     "the task: judge relevance to <i>this</i> question, and emit nothing else."),
    (29, 32,
     "<b>29</b> parses the <font name='Mono' size='8'>ID=SCORE</font> lines, anchored per line "
     "so commentary around them cannot be misread as scores. <b>30</b> caps how much of each "
     "passage is sent, which keeps a reranking call from costing as much as the answer itself."),
    (33, 43,
     "The function, and its governing rule, stated in the docstring: on any failure the original "
     "order is returned unchanged. A reranker that can <i>break</i> retrieval is worse than no "
     "reranker, so every path below either improves the order or leaves it alone."),
    (44, 52,
     "Building the prompt. <b>48-51</b> wrap each excerpt in a tag carrying its number, which is "
     "both how the model is told to refer to them and what makes the response parseable. "
     "<b>49</b> sends <font name='Mono' size='8'>indexed_text</font>, so the model sees the law "
     "and article label as well as the passage."),
    (53, 61,
     "The call. <b>59</b> asks for low effort deliberately - the comment explains why: judging "
     "whether a passage is relevant is a quick discrimination task, not a reasoning one, and "
     "paying for extended thinking here would buy nothing."),
    (62, 70,
     "Parsing. <b>65</b> ignores any id outside the range actually sent, so a hallucinated "
     "excerpt number is discarded rather than indexing into the wrong candidate. <b>68-69</b> "
     "are the main safety net: if nothing parsed at all, the fused order is returned untouched."),
    (71, 80,
     "Reordering. <b>74-79</b> sort by the model's score with the fused score as a tiebreak, and "
     "candidates the model did not score sink below those it did while keeping their relative "
     "order - so a partial response still helps rather than scrambling the list. <b>80</b> "
     "re-attaches anything beyond the reranked pool."),
]
