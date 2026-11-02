"""Line-by-line notes for rag/embeddings.py."""

SUMMARY = (
    "Turns text into vectors, which is what lets a question match a passage that shares no "
    "words with it. Two interchangeable backends behind one small interface, both returning "
    "normalised vectors so nothing downstream knows which was used."
)

BLOCKS = [
    (1, 15,
     "The docstring names both backends and the reason the hosted one is Voyage: Anthropic does "
     "not serve an embeddings endpoint, so Voyage is the option that pairs with Claude, while "
     "Google serves its own. <b>12-14</b> state the invariant that makes the two "
     "interchangeable - both return L2-normalised float32, so cosine similarity is a plain dot "
     "product and no code downstream has to normalise or even ask."),
    (16, 25, "Imports: numpy for the vectors, and the settings object."),
    (26, 33,
     "The interface, written as a Protocol - structural typing, so a backend satisfies it by "
     "having these members, with no base class to inherit. The two encode methods are separate "
     "because some models encode a question differently from a passage, as <b>87-89</b> below "
     "makes use of."),
    (34, 40,
     "Normalisation, with one defensive detail: <b>37</b> floors the norms at a tiny positive "
     "number, so a zero vector - which an empty or whitespace-only input produces - divides "
     "cleanly instead of yielding NaNs that would poison every later comparison."),
    (41, 54,
     "The local backend. <b>45-53</b> import torch and transformers lazily and convert the "
     "import error into the two things a user can act on: the install command, and the flag that "
     "runs the system keyword-only without them."),
    (55, 62,
     "Silencing model-loading chatter, wrapped in its own try because this is cosmetic and must "
     "never be the reason an ingest fails."),
    (63, 77,
     "Setting up the model. <b>67-70</b> resolve \"auto\" to the GPU if one exists. <b>74</b> "
     "puts the model in evaluation mode, which disables dropout - without it, encoding the same "
     "text twice would give slightly different vectors. <b>76</b> reads the output width from "
     "the model itself rather than hard-coding 1024, so changing models changes the dimension "
     "automatically."),
    (78, 90,
     "Three model-family decisions, all of which fail quietly if wrong. <b>81-86</b> choose the "
     "pooling strategy: BGE models are trained to be read from their first token, most others "
     "from the mean of all tokens, and the comment states the consequence of choosing wrong - "
     "retrieval degrades, nothing errors. <b>87-89</b> apply the asymmetric query and passage "
     "prefixes only to the e5 family, which is trained with them; adding them to a model that "
     "was not trained on them is actively harmful."),
    (91, 102,
     "Encoding. <b>92-93</b> return a correctly-shaped empty array, so callers never special-case "
     "an empty batch. <b>96-98</b> are a genuine speed trick: every batch is padded to its "
     "longest member, and a legal corpus mixes one-line articles with multi-page ones, so "
     "sorting by length before batching means short texts are no longer padded out to the length "
     "of the longest article in the corpus. <b>102</b> substitutes a space for an empty text, "
     "since the tokenizer rejects an empty string."),
    (103, 111,
     "The tokenizer call. <b>105-107</b> pad to the batch, truncate at the configured limit, and "
     "<b>109</b> moves the tensors to the same device as the model. <b>110-111</b> disable "
     "gradient tracking - this is inference only, and without it torch would build a backward "
     "graph for every batch and consume memory for nothing."),
    (112, 122,
     "Pooling and restoring order. <b>112-113</b> take the first token for CLS pooling. "
     "<b>115-116</b> are masked mean pooling: multiply by the attention mask so padding "
     "contributes nothing, sum, and divide by the real token count - the clamp preventing a "
     "division by zero. <b>119-120</b> undo the length sort, which is essential: the caller "
     "expects vector <i>i</i> to belong to text <i>i</i>, and a silent permutation here would "
     "mislabel every chunk in the index."),
    (123, 129,
     "The two public methods, differing only in which prefix they pass - empty for every model "
     "except e5."),
    (130, 144,
     "The hosted backend. <b>140</b> constructs the client, which reads the key from the "
     "environment. <b>143</b> sets a provisional dimension that <b>156</b> corrects from the "
     "first real response, so the code does not have to hard-code each model's width."),
    (145, 158,
     "Hosted encoding. <b>149</b> uses larger batches than the local path, because here the cost "
     "is network round trips rather than memory. <b>152</b> passes the input type, which is how "
     "this API expresses the same query/passage asymmetry the local path handles with prefixes."),
    (159, 165, "The same two methods, with this API's own labels for the two input types."),
    (166, 170,
     "Backend selection in three lines: if a Voyage key is present use the hosted service, "
     "otherwise run locally. The local path is the default precisely because it needs no key and "
     "no network once the weights are cached."),
]
