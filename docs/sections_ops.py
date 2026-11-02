"""Documentation sections 13-18 and the appendices."""

from __future__ import annotations

from reportlab.lib.units import mm
from reportlab.platypus import PageBreak

from make_pdf import DIAGRAM, S, SNAPSHOT, bullets, code, p, panel, table

W = 165 * mm


def _cols(*fractions: float) -> list[float]:
    return [W * f for f in fractions]


def section_data_flow() -> list:
    hops = [
        ["#", "What moves", "From", "To", "Form", "Size (this corpus)"],
        ["1", "Document bytes", "documents/ on disk", "Python process memory",
         "PDF / DOCX binary", "70 KB - 2.5 MB each"],
        ["2", "Extracted text", "Process memory", "Index folder (cache)",
         "JSON, UTF-8", "12 files"],
        ["3", "Passages", "Process memory", "chunks.jsonl",
         "One JSON object per line", f"{SNAPSHOT['index_files'].get('chunks.jsonl',0)/1e6:.1f} MB"],
        ["4", "Passage text", "Process memory", "Embedding model (same process)",
         "Python strings in batches of 16", f"{SNAPSHOT['index']['chunks']:,} passages"],
        ["5", "Vectors", "Model", "embeddings.npy",
         "float32 matrix", f"{SNAPSHOT['index_files'].get('embeddings.npy',0)/1e6:.1f} MB"],
        ["6", "Model weights", "Hugging Face (once)", "Local model cache",
         "safetensors", "2.2 GB embedder + 2.2 GB reranker"],
        ["7", "Question", "Terminal or browser", "Python process",
         "UTF-8 string, or JSON over HTTP", "tens of bytes"],
        ["8", "Question vector", "Model", "Process memory", "1024 float32", "4 KB"],
        ["9", "Candidate passages", "Index in RAM", "Reranker (same process)",
         "20 question-passage pairs", "~20 x 300 tokens"],
        ["10", "Prompt", "Python process", "Gemini or Claude over HTTPS",
         "JSON: system instruction + contents", "~3,000 tokens (~12 KB)"],
        ["11", "Answer", "Provider over HTTPS", "Python process",
         "JSON, or a stream of chunks", "tens to hundreds of tokens"],
        ["12", "Answer + sources", "Python process", "Browser",
         "Server-sent events", "a few KB"],
    ]
    sse = """
  browser                              server                      Gemini
     |   POST /ask/stream (JSON)          |                           |
     |----------------------------------->|                           |
     |                                    |-- retrieve + rerank       |
     |   event: sources   [10 passages]   |   (~2 s, all local)       |
     |<-----------------------------------|                           |
     |                                    |---- HTTPS request ------->|
     |   event: delta     {"text": "..."} |<---- stream of chunks ----|
     |<-----------------------------------|                           |
     |   event: delta     {"text": "..."} |                           |
     |<-----------------------------------|                           |
     |   event: done                      |                           |
     |<-----------------------------------|                           |
"""
    return [
        p("13  How data moves: every hop", "h1"),
        p("The question \"where does the data actually go?\" has a short answer - almost "
          "nowhere - and a precise one. Here is every movement of data in the system:"),
        table(hops, _cols(0.05, 0.21, 0.18, 0.19, 0.21, 0.16)),
        p("Hops 1 to 5 happen once per document, at ingestion. Hops 7 to 12 happen per "
          "question. Only hops 6 and 10/11 cross the network.", "small"),
        panel("<b>What leaves this machine, exactly.</b> One HTTPS request per question, "
              "containing: the system instructions; the ten or so retrieved passages, each "
              "labelled with its law, article, page and source filename; and the question "
              "itself. A measured example came to 3,003 input tokens, with 26 tokens of answer "
              "and 324 tokens of internal reasoning billed back. <b>Not</b> sent: the other "
              "3,820 passages, the index, the embeddings, the documents themselves, or "
              "anything about the machine. The API key travels in the request header over TLS. "
              "If a question matches nothing, no request is made at all."),
        p("The one network dependency at startup", "h2"),
        p("The embedding and reranking models are downloaded once from Hugging Face (about "
          "4.4 GB together) and cached in the user profile. After that, indexing and searching "
          "work with no network at all - which is why search keeps working with no API key and "
          "no internet, and only answering needs either."),
        p("Inside the web page", "h2"),
        p("The browser posts JSON and reads a Server-Sent Events stream. Sources arrive first, "
          "so the passages can be shown while the answer is still being written:"),
        code(sse, DIAGRAM),
        p("Server-sent events normally ride on a GET request through the browser's EventSource "
          "API, which cannot carry a JSON body. The page therefore posts and parses the event "
          "stream itself - about fifteen lines of JavaScript - which keeps the request shape "
          "the same as the JSON API.", "small"),
    ]


def section_interfaces() -> list:
    cli_rows = [
        ["Command", "What it does"],
        ["python -m rag ingest documents", "Index a folder. Unchanged files are skipped, edited "
                                           "ones re-indexed, deleted ones removed."],
        ["python -m rag search \"...\"", "Retrieve passages only. Needs no API key. Shows which "
                                         "retriever found each one, with scores."],
        ["python -m rag ask \"...\"", "Retrieve, then answer with citations. Streams as it writes."],
        ["python -m rag chat", "Multi-turn session; loads the models once, so each question "
                               "costs only its own time."],
        ["python -m rag stats", "What is indexed, which provider will answer, which keys were "
                                "found and where they came from."],
        ["python -m rag sources / remove / reset", "List indexed files; drop one; delete the index."],
        ["python -m rag serve --port 8000", "HTTP API and the test page."],
    ]
    api_rows = [
        ["Endpoint", "Body / result"],
        ["GET /", "The test page"],
        ["GET /health, GET /stats", "Index size, provider, model, which keys were found"],
        ["GET /examples", "A few questions from the evaluation set, for the page"],
        ["POST /search", '{"query": "...", "top_k": 5, "source": null, "rerank": null} '
                         "-> ranked passages with scores"],
        ["POST /ask", '{"question": "...", "history": []} -> answer, citations, sources, usage'],
        ["POST /ask/stream", "Same body; a sources event, then delta events, then done"],
        ["POST /ingest", "Disabled unless RAG_ALLOW_INGEST=1, because it reads server-side paths"],
    ]
    return [
        p("14  Interfaces: command line, HTTP, web page", "h1"),
        p("Three ways in, one pipeline behind them. All three call the same facade "
          "(<font name='Mono' size='8'>RAGPipeline</font>), so they cannot drift apart."),
        p("Command line", "h2"),
        table(cli_rows, _cols(0.33, 0.67)),
        p("A one-off command spends about 16 s loading the two models before it does anything; "
          "<font name='Mono' size='8'>chat</font> and <font name='Mono' size='8'>serve</font> "
          "load them once and then answer in about 2 s plus the provider's time.", "small"),
        p("HTTP API", "h2"),
        table(api_rows, _cols(0.28, 0.72)),
        p("The test page", "h2"),
        p("A single self-contained HTML file with no CDN and no build step, served at the root. "
          "It has a question box, an Ask button that streams the answer with clickable "
          "citations, and a Search button that retrieves without using an API key. Each "
          "passage shows its law and article, its dense, keyword and rerank scores, and the "
          "timing. It exists mostly because Arabic renders right-to-left correctly in a "
          "browser and cannot in a Windows terminal."),
    ]


def section_config() -> list:
    rows = [
        ["Setting", "Default", "Effect"],
        ["GEMINI_API_KEY / ANTHROPIC_API_KEY", "-", "Enables answering. Nothing else needs a key."],
        ["RAG_LLM_PROVIDER", "auto", "auto picks whichever key exists, preferring Gemini"],
        ["RAG_MODEL", "per provider", "gemini-3.8-flash or claude-opus-5 unless set"],
        ["RAG_INDEX_DIR", ".rag_index", "Where the index lives; several indexes can coexist"],
        ["RAG_EMBED_MODEL", "BAAI/bge-m3", "Changing it invalidates the index (re-ingest with --force)"],
        ["RAG_CHUNK_TOKENS", "512", "1,200 for this Arabic corpus; remembered by the index"],
        ["RAG_ARTICLE_CHUNKS", "1", "One passage per article for legislation"],
        ["RAG_RERANK / RAG_RERANKER_MODEL", "1 / bge-reranker-v2-m3", "The cross-encoder stage"],
        ["RAG_RERANKER_INT8", "1", "int8 weights on CPU: 5x faster, same quality"],
        ["RAG_TOP_K / RAG_CONTEXT_TOKENS", "12 / 4000", "Passages considered, and the budget sent"],
        ["RAG_DENSE_WEIGHT / RAG_LEXICAL_WEIGHT", "1.0 / 1.0", "Fusion balance; measure before changing"],
        ["RAG_EFFORT", "medium", "Maps onto Gemini's thinking level"],
        ["RAG_RETRY_ATTEMPTS / RAG_RETRY_SECONDS", "2 / 1.5", "Retries for 503 and 429, doubling"],
        ["RAG_DENSE", "1", "0 falls back to keyword-only, with no model downloads at all"],
    ]
    return [
        p("15  Configuration", "h1"),
        p("Everything is one dataclass with one environment variable per field. Values are "
          "resolved in this order, each overriding the one before it:"),
        *bullets([
            "The defaults in <font name='Mono' size='8'>rag/config.py</font>, chosen from the "
            "measurements in the next section",
            "A <font name='Mono' size='8'>.env</font> file in the project folder - where the "
            "API key goes",
            "A real environment variable, which wins over the file, so one command can be run "
            "differently without editing anything",
            "A <font name='Mono' size='8'>Config</font> object passed in code, or a command-line "
            "flag such as <font name='Mono' size='8'>--provider</font>",
        ]),
        table(rows, _cols(0.33, 0.17, 0.5)),
        p("Some settings are remembered by the index itself: chunk size is recorded when the "
          "index is built and reused automatically afterwards, so adding a file later cannot "
          "silently chunk it differently from the rest.", "small"),
    ]


def section_measurement() -> list:
    progress = [
        ["Change", "Answer ranked first", "Answer within 1,000 tokens"],
        ["Starting point: 1,200-token page windows, e5-small embeddings", "43%", "43%"],
        ["One passage per article, labelled with law and article", "47%", "61%"],
        ["+ local cross-encoder reranker (int8)", "76%", "84%"],
        ["<b>+ bge-m3 embeddings (current)</b>", "<b>80%</b>", "<b>90%</b>"],
    ]
    budget = [
        ["Context sent to the model", "Page windows", "One article per passage"],
        ["1,000 tokens", "43%", "61%"],
        ["2,000 tokens", "57%", "71%"],
        ["4,000 tokens", "69%", "84%"],
    ]
    return [
        p("16  How the system was measured", "h1"),
        p("Every change above was kept or discarded on evidence from a fixed test set: 49 "
          "questions over these laws - 41 Arabic, 5 English, 3 article-number lookups - each "
          "tied to the article that answers it. A retrieval counts as correct only if the "
          "passage comes from the right law <i>and</i> contains the answering text, compared "
          "after the same Arabic normalisation the search uses."),
        table(progress, _cols(0.52, 0.24, 0.24)),
        p("With the shipped settings: 94% of questions have the right article in the top five, "
          "mean reciprocal rank 0.86, all five English questions and all three article lookups "
          "ranked first.", "small"),
        p("The measurement mistake worth repeating", "h2"),
        p("Switching to one passage per article first looked like a regression: \"recall@5\" "
          "fell from 80% to 65%. The metric was wrong, not the change. A 1,200-token page "
          "window holds five or ten articles, so it counts as a hit whenever <i>any</i> of "
          "them answers the question, while an article passage has to be the exact article. "
          "Comparing five against five handed the page windows four times the text. Measured "
          "by what actually matters - how often the answer reaches the model for the same "
          "number of tokens - articles win everywhere:"),
        table(budget, _cols(0.4, 0.3, 0.3)),
        p("Things that were tried and rejected", "h2"),
        *bullets([
            "<b>Packing short articles together</b> (400- and 800-token groups): worse at "
            "every budget than one article per passage.",
            "<b>multilingual-e5-large embeddings:</b> 71% ranked first after reranking, below "
            "both bge-m3 (80%) and even the much smaller e5-small (76%). Bigger is not "
            "automatically better.",
            "<b>The reranker at full precision:</b> same quality as int8, 2.3 times slower.",
            "<b>Rewriting BM25 for speed:</b> profiling showed search takes 0.1 ms and the "
            "600 ms figure came from the measurement harness rebuilding the index on every "
            "query. The fix was to the harness; the code was left alone.",
        ]),
        panel("<b>Honest caveat.</b> The test set was written by the same person who built the "
              "system, from the documents being indexed. That flatters the absolute numbers. "
              "The comparisons between versions are the trustworthy part, because every "
              "version faced identical questions."),
    ]


def section_testing() -> list:
    rows = [["Test file", "Tests", "What it covers"]]
    descriptions = {
        "test_rag.py": "Chunking offsets and overlap, BM25, fusion, MMR, the index round-trip, "
                       "filters, citation parsing",
        "test_arabic.py": "Spelling normalisation, diacritics, digits, the PDF repairs, the "
                          "detectors that decide when to apply them",
        "test_legal.py": "Article heading styles, the ordering guard against false headings, "
                         "article-aligned chunking",
        "test_sources.py": "Word files, duplicates, deleted and renamed files, the extraction "
                           "cache, the context budget",
        "test_llm.py": "Provider selection, Gemini and Claude request shapes, refusals, "
                       "retries, error messages, .env loading",
        "test_server.py": "Every HTTP endpoint, the SSE event order, the page, the disabled "
                          "ingest route",
        "conftest.py": "Shared settings: no model weights, no network",
    }
    for name, count in SNAPSHOT["tests"].items():
        rows.append([name, str(count) if count else "-", descriptions.get(name, "")])
    return [
        p("17  Testing", "h1"),
        p(f"{sum(SNAPSHOT['tests'].values())} test functions ({len(SNAPSHOT['tests'])} files) "
          "run in about two seconds, with no network, no API key and no model weights."),
        table(rows, _cols(0.22, 0.1, 0.68)),
        p("How that is possible", "h2"),
        p("Tests use a deterministic fake embedder: the same text always produces the same "
          "vector, derived from a hash. That exercises all the plumbing - persistence, fusion, "
          "filtering, offset integrity, citation parsing - without loading 2.2 GB of weights. "
          "Embedding <i>quality</i> is a different question, and that is what the evaluation "
          "harness in section 16 is for."),
        p("The Gemini tests are built the same way: requests are constructed with the real SDK "
          "types and handed to a fake transport, so a wrong field name or an invalid value "
          "fails in the test suite rather than on the first real call."),
        p("What the tests do not cover", "h2"),
        *bullets([
            "The quality of what the model writes. The tests check that a refusal is handled, "
            "that citations are parsed and that the request is well formed; whether the answer "
            "is <i>good</i> is judged by reading it.",
            "Real provider behaviour. A live 503 was observed once in practice and is now "
            "retried, but the retry itself is tested against a simulated failure.",
            "Extraction quality on PDFs other than these twelve. The repairs are guarded by "
            "detectors, so an unfamiliar file should pass through untouched rather than be "
            "damaged - but only these files have been read closely.",
        ]),
    ]


def section_limits() -> list:
    return [
        p("18  Limitations and next steps", "h1"),
        p("Known limitations", "h2"),
        *bullets([
            "<b>Scanned PDFs yield nothing.</b> Extraction reads embedded text only. A scanned "
            "page needs OCR first.",
            "<b>Some font junk survives.</b> In the database exports a few glyphs decode to "
            "runs of random letters - between 0.1% and 3% of words depending on the file. The "
            "word stem before the junk is intact, so retrieval mostly survives, but a quoted "
            "sentence can show it.",
            "<b>Three test questions still miss.</b> They ask in everyday words for things the "
            "law names differently, or target one entry inside a very long definitions article.",
            "<b>Keyword stemming is prefix-only.</b> Plural and suffixed forms do not match by "
            "keyword; the embeddings cover most of that gap.",
            "<b>One writer at a time.</b> Two processes indexing the same folder concurrently "
            "will have one overwrite the other.",
            "<b>Providers have bad minutes.</b> Transient failures are retried, but a stream "
            "that breaks mid-answer cannot be.",
        ]),
        p("Where it would go next", "h2"),
        *bullets([
            "<b>Expand the test set with real questions.</b> The current one was written by "
            "the system's author; questions from actual users would measure what matters.",
            "<b>Arabic suffix stemming,</b> measured rather than assumed - it may help keyword "
            "search or may blur distinct terms.",
            "<b>Cross-references between laws.</b> Articles cite other articles constantly; "
            "following those links when assembling context would answer layered questions.",
            "<b>A GPU, if the corpus grows.</b> Indexing is the only slow part, and it is the "
            "part a GPU would change by an order of magnitude.",
        ]),
    ]


def appendix_files() -> list:
    rows = [["File", "Lines"]]
    for name, count in SNAPSHOT["modules"].items():
        rows.append([f"rag/{name}", str(count)])
    rows.append(["rag/web/index.html", str(SNAPSHOT["web_lines"])])
    rows.append(["eval/evaluate.py", str(SNAPSHOT["eval_lines"])])
    total = sum(SNAPSHOT["modules"].values()) + SNAPSHOT["web_lines"] + SNAPSHOT["eval_lines"]
    rows.append(["<b>Total (excluding tests)</b>", f"<b>{total:,}</b>"])

    docs = [["Document", "Articles indexed", "Word order repaired"]]
    for name, meta in sorted(SNAPSHOT["documents"].items(), key=lambda kv: -(kv[1]["chunks"] or 0)):
        docs.append([meta["title"] or name, str(meta["chunks"]), "yes" if meta["repaired_order"] else "-"])

    return [
        PageBreak(),
        p("Appendix A  File-by-file map", "h1"),
        table(rows, _cols(0.5, 0.5)),
        p("The indexed corpus", "h2"),
        p("Titles below are the ones the system extracted from each law's own naming clause; "
          "they are what appears in citations.", "small"),
        table(docs, _cols(0.62, 0.2, 0.18)),
    ]


def appendix_glossary() -> list:
    terms = [
        ["Term", "Meaning"],
        ["RAG", "Retrieval-augmented generation: find the relevant text first, then ask a "
                "model to answer using only that text."],
        ["Chunk / passage", "A piece of a document indexed as one unit. Here, one article."],
        ["Token", "The unit models count in - roughly a short word or word-part. Arabic costs "
                  "more tokens per word than English in most tokenizers."],
        ["Embedding", "A list of numbers representing meaning, so that similar texts have "
                      "similar lists. 1,024 numbers per passage here."],
        ["Cosine similarity", "How closely two embeddings point in the same direction; 1.0 is "
                              "identical, 0 unrelated."],
        ["BM25", "The standard keyword ranking formula: rare words count more, repetition has "
                 "diminishing returns, long passages are penalised."],
        ["RRF", "Reciprocal Rank Fusion: merge two rankings using positions rather than "
                "scores, which avoids comparing incomparable numbers."],
        ["MMR", "Maximal Marginal Relevance: pick results that are relevant but not "
                "near-copies of each other."],
        ["Cross-encoder", "A model that reads question and passage together and scores "
                          "relevance - slower and far more accurate than comparing embeddings."],
        ["int8 quantisation", "Storing model weights as 8-bit integers instead of 32-bit "
                              "floats: several times faster on a CPU, with little quality cost."],
        ["Presentation forms", "Unicode code points for the contextual shapes of Arabic "
                               "letters. Some PDFs store these instead of normal letters."],
        ["Tanween", "The Arabic grammatical ending written as a doubled vowel mark; one font "
                    "in this corpus decodes it to the wrong character."],
        ["SSE", "Server-sent events: a one-way stream of text events over HTTP, used to show "
                "the answer as it is written."],
    ]
    return [
        p("Appendix B  Glossary", "h1"),
        table(terms, _cols(0.22, 0.78)),
    ]
