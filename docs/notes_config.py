"""Line-by-line notes for rag/config.py."""

SUMMARY = (
    "Every setting in the system, in one dataclass, plus the loader that reads a .env file. "
    "This module is imported before any other, because the defaults below are evaluated at "
    "import time from the environment."
)

BLOCKS = [
    (1, 8,
     "The module docstring states the precedence rule the rest of the file implements: a "
     "value passed in code beats a real environment variable, which beats the .env file, "
     "which beats the built-in default. Writing it at the top means the rule is stated once "
     "and the code below only has to obey it."),
    (9, 15,
     "<b>Line 10</b> makes all type annotations lazy strings, which is what allows modern "
     "syntax such as <font name='Mono' size='8'>Path | None</font> on older interpreters and "
     "avoids import cycles. <b>12-15</b> are the only imports this module needs: "
     "<font name='Mono' size='8'>os</font> for the environment, dataclasses for the settings "
     "object, <font name='Mono' size='8'>Path</font> for filesystem paths, and "
     "<font name='Mono' size='8'>Any</font> for the dictionary return types."),
    (16, 19,
     "The filename is a named constant rather than the literal \".env\" scattered through the "
     "code, because the CLI prints it back to the user when no key is found - and a message "
     "that names the wrong file is worse than no message."),
    (20, 27,
     "<font name='Mono' size='8'>load_env_file</font> reads one file. The docstring justifies "
     "writing a parser instead of adding the python-dotenv dependency: for a file holding two "
     "API keys, the whole grammar is comments, blank lines, an optional export prefix and "
     "quotes. The last sentence describes a deliberate safety property implemented on line 43."),
    (28, 32,
     "<b>28</b> collects what this call actually set, which the CLI later shows as \"from .env: "
     "GEMINI_API_KEY\". <b>29-32</b> read the file, and a missing or unreadable file returns an "
     "empty dictionary rather than raising: configuration is optional, so its absence must not "
     "stop the program. Catching <font name='Mono' size='8'>UnicodeDecodeError</font> as well "
     "covers a file saved in another encoding."),
    (33, 40,
     "The parse loop. <b>34</b> strips surrounding whitespace. <b>35</b> skips blank lines, "
     "comments, and any line without an \"=\" - that last test is what lets prose sit in the "
     "file without breaking it. <b>37-38</b> drop a leading <font name='Mono' size='8'>export "
     "</font>, so a file copied from a shell script works. <b>39</b> splits on the first \"=\" "
     "only, so a value may itself contain \"=\" (base64 keys often do). <b>40</b> trims "
     "whitespace and one layer of single or double quotes."),
    (41, 48,
     "The precedence rule in one condition. <b>43</b> requires a non-empty key <i>and</i> a "
     "non-empty value <i>and</i> that the variable is not already set. The empty-value test is "
     "why the shipped template, which contains "
     "<font name='Mono' size='8'>GEMINI_API_KEY=</font> with nothing after it, does not look "
     "like a configured key. The <font name='Mono' size='8'>not in os.environ</font> test is "
     "what lets one command override the file for a single run."),
    (49, 57,
     "<font name='Mono' size='8'>load_env_files</font> looks in two places: the directory the "
     "command was run from and the project root, so the key is found whether you run from "
     "inside the project or from somewhere else. <b>53</b> uses "
     "<font name='Mono' size='8'>dict.fromkeys</font> as an order-preserving way to drop the "
     "duplicate when those two directories are the same, which is the normal case."),
    (58, 62,
     "This line runs at import time, and the comment explains why it has to: the dataclass "
     "fields below call <font name='Mono' size='8'>_env_str</font> and friends while the class "
     "body is being executed, so anything the .env file is going to contribute must already be "
     "in <font name='Mono' size='8'>os.environ</font> by then. Moving this line below the class "
     "would silently break .env support."),
    (63, 66,
     "The four readers that follow all have the same shape: look the variable up, fall back to "
     "the default. The string version needs nothing else."),
    (67, 76,
     "The integer version has to handle two different failures. <b>71-72</b>: the variable is "
     "absent, so use the default. <b>73-76</b>: the variable is present but not a number - a "
     "typo like <font name='Mono' size='8'>RAG_TOP_K=ten</font> - in which case the default is "
     "used rather than crashing the program at import time, when a traceback would point at "
     "config.py rather than at the typo."),
    (77, 86, "The float version, identical in shape to the integer one."),
    (87, 93,
     "The boolean version accepts the spellings people actually type. <b>93</b> treats \"1\", "
     "\"true\", \"yes\" and \"on\" as true, case-insensitively, and everything else as false, "
     "so <font name='Mono' size='8'>RAG_RERANK=0</font> and "
     "<font name='Mono' size='8'>RAG_RERANK=false</font> both work."),
    (94, 101,
     "The settings object itself. <b>96</b> makes it a dataclass, so the fields below get an "
     "<font name='Mono' size='8'>__init__</font>, a readable repr and equality for free. "
     "<b>99-101</b> need <font name='Mono' size='8'>field(default_factory=...)</font> rather "
     "than a plain default because the value is a mutable <font name='Mono' size='8'>Path</font> "
     "object; dataclasses refuse mutable defaults, since every instance would otherwise share "
     "one object."),
    (102, 116,
     "Chunking settings. <b>106-108</b> size an ordinary text window and the overlap between "
     "consecutive windows. <b>110</b> switches on the legal path - one chunk per article. "
     "<b>114</b> is a measured default: the comment records that packing short articles "
     "together was tried and lowered quality at every context size, so the value is 0 (never "
     "pack). <b>116</b> controls whether each chunk is indexed with its \"law - article\" label."),
    (117, 132,
     "Embedding settings. The comment on <b>119-123</b> records the comparison that chose this "
     "model, including the alternative for a slower machine. <b>125</b> is the batch size for "
     "the GPU or CPU. <b>126</b> caps the input length the model sees. <b>127-128</b> are "
     "\"auto\" by default: the device is detected, and the pooling strategy is inferred from the "
     "model name, because choosing it wrongly degrades retrieval silently rather than failing. "
     "<b>131-132</b> hold the asymmetric prefixes that only the e5 family wants."),
    (133, 149,
     "Retrieval settings. <b>137-138</b> are the pair that decides how much context an answer "
     "gets: at most 12 passages, and no more than 4,000 tokens of them. <b>139</b> is how deep "
     "each retriever searches before fusion, <b>140</b> the smoothing constant in the fusion "
     "formula. <b>145-146</b> let one retriever be trusted more than the other; both are 1.0 "
     "because the comment's advice is to measure before moving them. <b>147-148</b> control "
     "near-duplicate removal, where 1.0 would mean pure relevance and no diversity."),
    (150, 161,
     "Reranking. <b>155</b> is on by default, and the comment carries the measurement that "
     "justifies the cost - 47% to 76% of questions with the answer ranked first, for about two "
     "seconds. <b>156</b> chooses the local cross-encoder over spending an API call. "
     "<b>158</b> is how many fused candidates get rescored; the reranker can only promote what "
     "it is shown. <b>161</b> turns on int8 weights, which made the large model usable on a CPU."),
    (162, 177,
     "Generation. <b>167</b> defaults to \"auto\", which picks whichever provider has "
     "credentials. <b>169</b> is deliberately empty: an empty model name means \"use this "
     "provider's default\", which is how one setting serves two providers with different model "
     "names. <b>171</b> maps onto Gemini's thinking level and Claude's effort. <b>176-177</b> "
     "set the retry policy for transient provider failures."),
    (178, 184,
     "Where extracted text is cached. <b>181-183</b> let the cache be switched off entirely "
     "with <font name='Mono' size='8'>RAG_EXTRACT_CACHE=off</font>, which the test suite uses "
     "so tests never share state on disk. <b>184</b> otherwise honours an explicit path, or "
     "puts the cache inside the index folder, so deleting the index deletes its cache too."),
    (185, 187,
     "A method rather than a field, because it is read at query time: setting "
     "<font name='Mono' size='8'>RAG_DENSE=0</font> in an already-running process takes effect "
     "on the next call, which is convenient when experimenting."),
    (188, 192,
     "<font name='Mono' size='8'>to_dict</font> serialises the settings - the HTTP API and the "
     "test suite both use it. <b>191</b> converts the Path to a string, because "
     "<font name='Mono' size='8'>Path</font> objects are not JSON-serialisable."),
    (193, 204,
     "The fingerprint: the subset of settings that, if changed, makes an existing index "
     "invalid. It is written into the index and compared on load. Embedding vectors from two "
     "different models cannot be compared at all, and chunk boundaries decide what the vectors "
     "even describe - so a mismatch here is reported rather than silently producing wrong "
     "results. Settings <i>not</i> in this list, such as top_k, can be changed freely without "
     "re-indexing."),
    (205, 207,
     "One shared instance for callers that do not want to build their own. The pipeline copies "
     "it rather than mutating it, so that one component adopting an index's chunk settings "
     "cannot leak into another."),
]
