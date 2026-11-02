"""Line-by-line notes for rag/generate.py."""

SUMMARY = (
    "The prompt, and what happens to the answer afterwards. This is the module that decides "
    "the system says 'not in these documents' instead of inventing something plausible."
)

BLOCKS = [
    (1, 17,
     "Three decisions, stated before any code. <b>5-9</b>: retrieved context goes in the system "
     "part rather than the user turn, which is what lets Claude cache it across follow-up "
     "questions. <b>10-13</b> is the one that matters most - an honest \"not in these documents\" "
     "is the single most valuable behaviour in a retrieval system, because a fluent guess is "
     "worse than no answer: it is indistinguishable from a real one. <b>14-16</b>: citations are "
     "parsed back out of the answer, so the caller gets a structured list of what was actually "
     "used rather than prose that happens to contain brackets."),
    (18, 28, "Imports: the settings, the provider abstraction, and the retrieved-chunk type."),
    (29, 39,
     "The system prompt - the most consequential twelve lines in the project. <b>32</b> forbids "
     "outside knowledge and, specifically, filling gaps with plausible detail. <b>33</b> fixes "
     "the citation format and requires the marker next to the claim, not collected at the end, "
     "so a reader can check one sentence against one excerpt. <b>34</b> is corpus-specific: name "
     "the law and article, because that is how legal provisions are referred to - and never cite "
     "an article number that does not appear in an excerpt, which is the instruction that stops "
     "the most convincing kind of fabrication. <b>35</b> requires an explicit admission of a gap. "
     "<b>36</b> surfaces disagreement between excerpts rather than silently choosing - with "
     "amended legislation, two excerpts disagreeing is a fact about the corpus. <b>37</b> keeps "
     "an Arabic question answered in Arabic."),
    (40, 42,
     "The citation pattern. Bounded to three digits, so a year in brackets is not mistaken for "
     "a citation."),
    (43, 52,
     "The answer record: the text, the sources it was given, which ones it actually cited, and "
     "the usage and stop reason from the provider."),
    (53, 61,
     "<b>54-55</b> resolve citation numbers back to the chunks, ignoring any out of range. "
     "<b>57-60</b> is a quality signal in one line: an answer with no citations did not point at "
     "its evidence, which the caller can warn about."),
    (62, 77,
     "Serialisation, used by the HTTP API, the JSON output mode and the evaluation harness. The "
     "cited list carries each source's location, so a consumer gets \"Labour Law - Article 61 "
     "(p.22)\" without having to look anything up."),
    (78, 89,
     "Rendering the context. Each excerpt is numbered and carries its source path and its "
     "human-readable location as attributes - <b>84</b> - which is what makes the citation "
     "instruction on <b>34</b> satisfiable: the model can only name the article because the "
     "location attribute tells it which one this is."),
    (90, 99,
     "Parsing citations back out. <b>96</b> keeps only numbers that refer to an excerpt actually "
     "provided and drops repeats, so the result is the set of sources used, in the order they "
     "were first cited."),
    (100, 110,
     "Assembling the request in the provider-neutral shape: two system parts - rules, then "
     "context - and the conversation with the new question appended. <b>108</b> is what makes "
     "follow-up questions work, since prior turns are carried while the context is rebuilt from "
     "the new retrieval."),
    (111, 126,
     "Generating an answer. <b>119-125</b> handle the empty-retrieval case without calling the "
     "model at all: there is nothing to ground an answer in, so the honest response is the one "
     "written here, and it suggests the two things that actually help."),
    (127, 139,
     "The call, and refusal handling. <b>131-138</b> turn a refusal into a plain statement that "
     "the model declined, carrying the category - rather than an empty answer, which would look "
     "like a bug, or an exception, which would lose the retrieved sources."),
    (140, 149,
     "The successful answer, with citations parsed and usage recorded. Note <b>143</b> keeps "
     "<i>all</i> the retrieved chunks, not just the cited ones, so a reader can see what the "
     "model was given and judge what it chose to ignore."),
    (150, 164,
     "The streaming variant, used by both the terminal and the web API. It is deliberately "
     "thinner than the non-streaming path: citations cannot be parsed until the text is "
     "complete, so the caller does that once the stream ends."),
]
