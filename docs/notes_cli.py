"""Line-by-line notes for rag/cli.py."""

SUMMARY = (
    "Nine commands over the pipeline. Most of the length is presentation, and most of the "
    "decisions in it are about making failures legible: what went wrong, and what to do next."
)

BLOCKS = [
    (1, 9,
     "The docstring is the command list, so <font name='Mono' size='8'>python -m rag --help</font> "
     "and the module's own documentation cannot drift apart."),
    (10, 28,
     "Imports. Typer builds the argument parsing from type hints, and rich handles the terminal "
     "output. <b>21</b> imports the markup escaper, which <b>88</b> and others need: square "
     "brackets in document text are rich markup, so an unescaped Arabic title containing them "
     "would be swallowed or would corrupt the colouring."),
    (29, 35,
     "The application object. <b>31</b> prints help when called with no arguments, which is the "
     "right behaviour for a tool with nine subcommands."),
    (36, 59,
     "Building a pipeline from the common flags, used by every command. Each override is applied "
     "only when the flag was given, so an unset flag falls through to the environment and then "
     "to the default. <b>58</b> is the subtle one: index settings are adopted <i>unless</i> the "
     "user passed an explicit chunk size, so an explicit flag always wins over what the index "
     "remembers."),
    (60, 68,
     "The staleness warning, with the scope rule in its comment: a search only goes stale if the "
     "embedding model changed, while chunk settings are an ingest-time concern - so querying "
     "checks one key and ingesting checks them all."),
    (69, 80,
     "The ingest command's arguments. <b>75-78</b> document the chunk size where the user will "
     "actually look for it, including the measured value that suits an Arabic corpus."),
    (81, 92,
     "<b>86-88</b> print per-file progress, skipping unchanged files so re-running on a large "
     "corpus prints only what changed. <b>88</b> escapes both the filename and the status, "
     "because both can contain brackets. <b>90</b> shows a spinner during the slow part."),
    (93, 100,
     "The summary, then three lists that each name a specific outcome: files removed because "
     "they no longer exist, files skipped as duplicates - naming the file kept, so the decision "
     "can be checked - and files that failed, with the error. A silent skip is a bug report "
     "waiting to happen."),
    (101, 107,
     "A closing line stating what the index now is: how many chunks, from how many sources, "
     "whether dense retrieval is active, and at what chunk size. That one line answers most "
     "questions a user has after an ingest."),
    (108, 120,
     "The search command. Its docstring carries the fact worth advertising: retrieval needs no "
     "API key, so the system is useful before any credentials exist."),
    (121, 133,
     "<b>125-129</b> print JSON with the standard library rather than rich's JSON printer, and "
     "the comment records why: rich wraps long values to the terminal width, which corrupts the "
     "output the moment it is piped into anything. <b>130-132</b> exit non-zero on no matches, "
     "so a script can tell, and suggest the command that explains why."),
    (134, 145,
     "Result panels. <b>135</b> truncates to 400 characters unless "
     "<font name='Mono' size='8'>--full</font> was passed. <b>140</b> puts the retrieval "
     "explanation in the subtitle - which retriever found this, at what rank, with what score - "
     "so tuning is a matter of reading the output."),
    (146, 163,
     "The ask command, which carries the fullest set of options: model, provider, filters, "
     "reranking, and whether to show sources."),
    (164, 177,
     "<b>164-171</b> are the JSON path, which cannot stream because the whole answer is needed "
     "to serialise it. <b>173</b> is the normal path, taking sources and a stream together. "
     "<b>174-176</b> stop before calling the model when retrieval found nothing - there is no "
     "point paying for a request that can only be answered with \"not in the documents\"."),
    (178, 191,
     "Streaming to the terminal. <b>180-181</b> show a spinner until the <i>first</i> token "
     "arrives and then drop it, which is exactly as long as the user is waiting with nothing to "
     "read. <b>179</b> and <b>182</b> accumulate the text as it prints, because the citations "
     "have to be parsed from the complete answer afterwards."),
    (192, 206,
     "The source table. <b>195</b> parses the citations out of the streamed text, and "
     "<b>201</b> marks the ones the answer actually used - so a reader sees both what was "
     "retrieved and what was relied on, which is the honest presentation of a RAG result."),
    (207, 227,
     "The chat command. Its docstring states the design in four words: each turn retrieves "
     "afresh. A follow-up question gets its own retrieval rather than relying on passages "
     "fetched for an earlier one."),
    (228, 243,
     "The input loop. <b>232-234</b> treat Ctrl-C and end-of-input as a clean exit rather than a "
     "traceback. <b>237-242</b> are the two commands: quit, in three spellings, and clear the "
     "history."),
    (244, 259,
     "A turn. <b>246-248</b> keep the session alive after an error - a rate limit should not end "
     "a conversation. <b>250</b> renders markdown. <b>257-258</b> cap the history at four "
     "exchanges, and the comment gives the reasoning: in this system retrieval carries the "
     "context, so a long history adds cost without adding grounding."),
    (260, 271,
     "The stats command: index location, size, embedder and retrieval mode."),
    (272, 292,
     "The part of stats that exists to answer \"why is it not answering\". <b>272-277</b> name "
     "the provider and model, or say plainly that there is no key and retrieval still works. "
     "<b>278-281</b> show which keys were found <i>and</i> which of them came from the .env "
     "file, which is what distinguishes \"no key\" from \"a key that is not being picked up\". "
     "<b>283-291</b> each print a next step: where to put a key, that settings have drifted, or "
     "that nothing is indexed yet."),
    (293, 308,
     "Listing indexed files with their formats and chunk counts - the quickest way to check "
     "whether a document that should be searchable actually made it in."),
    (309, 323,
     "Removing one source. <b>320-322</b> report a path that was not indexed and exit non-zero, "
     "rather than silently reporting success for a no-op."),
    (324, 340,
     "Deleting the index. <b>333-335</b> handle there being nothing to delete. <b>336-337</b> "
     "confirm first unless the user passed the flag that says not to - this is the one "
     "destructive command, and it should be hard to run by accident."),
    (341, 360,
     "Running the HTTP API. <b>349-353</b> turn a missing dependency into the install command. "
     "<b>354-357</b> pass the index location through the environment, because the server is "
     "started by name in a separate process and cannot be handed an object. <b>358</b> prints "
     "the URL, since the test page is the point of this command."),
    (361, 376,
     "A Windows-specific fix, documented because it is invisible until it bites. An interactive "
     "console receives Unicode through the console API, but a pipe or a redirect falls back to "
     "the ANSI code page - which on an Arabic-locale Windows is cp1256. Arabic output then "
     "arrives as mojibake in anything reading it, or fails outright on characters the code page "
     "lacks. <b>371-375</b> force UTF-8 on both streams, tolerating streams that have been "
     "replaced or detached."),
    (377, 388,
     "The entry point. <b>382-384</b> turn Ctrl-C into a short message and the conventional exit "
     "code for an interrupt, rather than a traceback that looks like a crash."),
]
