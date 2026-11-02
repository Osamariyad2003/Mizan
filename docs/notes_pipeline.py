"""Line-by-line notes for rag/pipeline.py."""

SUMMARY = (
    "The object everything else drives: the CLI, the web server and the evaluation harness all "
    "call these methods. It wires the store, the retriever and the model together and owns the "
    "decisions that span them."
)

BLOCKS = [
    (1, 11,
     "The docstring is a complete session: construct, ingest, ask, then print the answer and the "
     "sources it cited. <b>9-10</b> deliberately show the citation loop, because an answer "
     "without its sources is only half of what this system produces."),
    (12, 29,
     "Imports, which read as a map of the package: chunking, config, generation, loaders, "
     "retrieval, storage. The pipeline is the only module that touches all of them."),
    (30, 46,
     "Construction. <b>38-40</b> copy the shared default settings rather than using the object "
     "itself - the comment gives the reason: this pipeline may adopt the index's chunk settings "
     "below, and mutating the shared default would leak those into every other pipeline in the "
     "same process. <b>41</b> loads the index unless told not to. <b>42-43</b> leave the "
     "embedder unloaded; it is built on first use."),
    (47, 64,
     "Adopting the index's own chunking settings, which exists because of a real failure mode "
     "spelled out in the docstring: an index built with 1,200-token chunks would silently get "
     "512-token chunks for every file added later, unless the environment variable happened to "
     "be set again. <b>59</b> leaves an explicit environment setting alone, and <b>62</b> only "
     "overrides a value still at its default - so a caller who chose a size keeps it, and a "
     "caller who did not gets consistency with the index."),
    (65, 74,
     "The embedder, loaded lazily. The docstring states the policy: dense retrieval is a quality "
     "upgrade, not a hard requirement, so a missing model or an absent torch must degrade the "
     "system rather than take it down."),
    (75, 87,
     "<b>75</b> and the failure flag together mean the load is attempted exactly once - without "
     "the flag, a machine with no torch would retry the import on every single query. "
     "<b>83-85</b> catch everything and warn, which is the difference between \"search still "
     "works, keyword-only\" and a traceback. <b>76-78</b> handle dense retrieval being switched "
     "off deliberately, which takes the same path without the warning."),
    (88, 93,
     "The retriever is constructed per access rather than cached, because it holds no state - it "
     "is three references - and building it fresh means a configuration change takes effect "
     "immediately."),
    (94, 107,
     "Ingestion, and a subtle piece of housekeeping explained in the docstring: files that were "
     "indexed from these directories but no longer exist are removed <i>first</i>. Without that, "
     "a deleted document's passages keep appearing in answers, and a renamed file is mistaken "
     "for a duplicate of its own former name - which would silently skip it."),
    (108, 126,
     "The ingest body. <b>111-116</b> load each discovered file, recording a failure and "
     "continuing rather than aborting - one unreadable PDF in a folder of two hundred must not "
     "stop the other 199. <b>118-120</b> hand everything to the store. <b>121-123</b> merge the "
     "pruning results into the store's report so the caller sees one summary. <b>124</b> saves "
     "once at the end rather than per document."),
    (127, 136,
     "Pruning, scoped deliberately. The docstring names both limits: only sources under the "
     "directories actually being ingested are considered, so ingesting one folder never deletes "
     "another folder's documents, and text ingested directly as a string is never pruned because "
     "it has no file to be missing."),
    (137, 146,
     "<b>138-141</b> skip anything that still exists, and anything whose source is a URL rather "
     "than a path. <b>143</b> is the scoping test: only sources genuinely inside one of the "
     "ingested roots are removed. <b>144</b> records how many chunks each pruned source "
     "contributed, which is what lets the summary say \"-312 stale chunks\"."),
    (147, 166,
     "Indexing a string directly, for content that never lands on disk. It builds a Document by "
     "hand with the same identity and content hashes the file loader would produce, so the store "
     "cannot tell the difference. <b>163</b> forces the add, since there is no file whose "
     "modification time could be compared."),
    (167, 176, "Removing one source, and clearing the index entirely. Both save immediately."),
    (177, 190,
     "Search without generation - which needs no API key at all, and is the half of the system "
     "that can be evaluated offline. <b>185-188</b> short-circuit an empty index, and the "
     "comment says what that saves: loading embedding weights takes seconds and would be pure "
     "waste with nothing to search."),
    (191, 205,
     "Context selection, and the reasoning that sets this system apart from a fixed top-k. Chunk "
     "sizes vary tenfold - a constitutional article can be one sentence, a page window a "
     "thousand tokens - so a fixed count of passages gives wildly different amounts of evidence "
     "from one question to the next. <b>202-204</b> record the measurement: filling a token "
     "budget with the best-ranked chunks puts the answering passage in front of the model far "
     "more often than taking a fixed six."),
    (206, 217,
     "The budget loop. <b>212</b> carries the guarantee stated in the docstring: the "
     "<font name='Mono' size='8'>chosen and</font> test means the top-ranked hit is always "
     "included even if it alone exceeds the budget - the best passage is never dropped for being "
     "large. Everything after it must fit."),
    (218, 228,
     "Asking a question: select context, then generate. Two lines, because every decision was "
     "made elsewhere."),
    (229, 243,
     "The streaming variant, whose signature is the interesting part. It returns the sources "
     "<i>and</i> a generator, sources first, so a UI can render the citation list immediately "
     "and let the answer arrive into it - rather than showing nothing until the model finishes."),
    (244, 257,
     "Introspection, built to answer the question users actually have: why is it not working. "
     "<b>250</b> reports settings drift. <b>252</b> reports which providers have credentials. "
     "<b>253-257</b> report which keys came from the .env file and what that file is called - "
     "the comment names the exact confusion this resolves, a key that is set somewhere but not "
     "being picked up."),
    (258, 269,
     "<b>258-265</b> resolve the provider and model inside a try, because having no credentials "
     "is a normal state for this command - retrieval works without them, and "
     "<font name='Mono' size='8'>stats</font> must still print rather than raise. <b>268</b> "
     "looks a chunk up by id, which is how the web UI expands a citation into its full text."),
]
