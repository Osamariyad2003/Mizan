# RAG system

Retrieval-augmented generation over local documents: hybrid dense + BM25
retrieval, offset-tracked chunks, and cited answers from Gemini or Claude.

Indexing and search run entirely locally — no API key, no network after the
first model download. A key is needed only to *generate* answers, from either
Gemini or Claude.

```bash
pip install -r requirements.txt
# paste a key into .env (GEMINI_API_KEY=...) -- only needed for answers

python -m rag ingest documents --chunk-tokens 1200  # index a folder (PDF, Word, text...)
python -m rag search "rollback procedure"           # retrieve (no API key)
python -m rag ask "How do I roll back a deploy?"    # answer with citations
python -m rag chat                                  # multi-turn session
```

## Answer provider: Gemini or Claude

Only the final answer comes from a hosted model. Loading, chunking, embedding,
keyword search and reranking all run on your machine, so **switching provider
needs no re-indexing** and costs nothing in retrieval quality.

**Where the key goes:** open `.env` in this folder and paste it after
`GEMINI_API_KEY=`. Nothing else to do — it is read automatically.

```
GEMINI_API_KEY=AIza...your-key...
```

`python -m rag stats` confirms it: *keys found  gemini  (from .env: GEMINI_API_KEY)*.
A real environment variable overrides the file for one run
(`set GEMINI_API_KEY=... && python -m rag ask "..."`), which is handy for trying
a second key. `.env` is git-ignored; `.env.example` is the template to copy on
another machine.

| | |
|---|---|
| Pick automatically | Whichever key is present, preferring Gemini (it has a free tier): `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) → Gemini, `ANTHROPIC_API_KEY` → Claude |
| Force one | `RAG_LLM_PROVIDER=gemini` / `claude`, or `--provider gemini` per command |
| Default models | `gemini-3.8-flash` / `claude-opus-5`; override with `RAG_MODEL` or `--model` |
| Check what is configured | `python -m rag stats` prints the provider, model and which keys it found |

Free Gemini key: <https://aistudio.google.com/apikey>. Vertex AI also works —
set `GOOGLE_GENAI_USE_VERTEXAI=1` plus the usual project/location variables and
the SDK picks them up.

Provider differences the code handles for you: Gemini's assistant turns are
called `model`, its thinking is set by level (your `RAG_EFFORT` maps onto it)
and cannot be switched off, and a refusal arrives as a `finish_reason` rather
than a stop reason. Claude keeps its prompt-cache breakpoint on the retrieved
context; Gemini caches repeated prefixes by itself. Everything else — the
prompt, the `[n]` citations, the refusal handling — is shared.

## How it works

```
documents ─▶ load ─▶ split ──────────▶ embed ──┐
  pdf/docx     repair  one article per    ├─▶ .rag_index/  (vectors + chunks + BM25)
  md/txt/csv   Arabic  chunk (laws), or   │
                       token windows ─────┘
                                                  │
question ─▶ ┌─ dense (cosine)  ─┐                 │
            ├─ BM25 (lexical)  ─┴─▶ RRF ─▶ MMR ─▶ cross-encoder rerank ─▶ fill 4,000 tokens
                                                                           │
                                           Gemini or Claude ◀─ excerpts labelled "law — article"
                                                      │
                                                   answer citing article and [n]
```

**Hybrid retrieval.** Dense embeddings match paraphrases; BM25 matches the
things embeddings are worst at — identifiers, error codes, version numbers,
rare proper nouns. Results are combined with Reciprocal Rank Fusion, which
fuses *ranks* rather than scores, because a cosine similarity and a BM25 score
are not on comparable scales.

**MMR** drops near-duplicates, which overlapping chunk windows produce in
quantity. The top hit is always the top fused hit; diversity only applies below
it.

**Citations are structural, not cosmetic.** Every chunk keeps its character
offsets into the source file, so a citation resolves to a real page or position
and the answer can be checked against the original. Cited excerpt numbers are
parsed back out of the answer, so you get a structured list of what was
actually used — `answer.cited_sources`, not prose containing brackets.

**Refusal is a feature.** The model is instructed to say the context does not
contain the answer rather than fill the gap. A fluent guess is worse than no
answer, because it is indistinguishable from a real one.

## Multiple sources

Point `ingest` at a folder and every supported file in it is indexed into one
index; questions are answered from whichever documents are relevant, and each
citation names its document and page.

| Format | Notes |
|---|---|
| `.pdf` | Needs an embedded text layer; scanned pages need OCR first. |
| `.docx` | Paragraphs, tables and footnotes, in document order. Headings stay attached to the text under them. Legacy binary `.doc`: open in Word and save as `.docx`. |
| `.txt` `.md` `.html` `.csv` `.json` | Plain text, markup stripped, one record per row. |

- **Duplicates are skipped.** A file whose content is identical to one already
  indexed (the same PDF downloaded twice) is reported and not indexed again,
  so it cannot fill the results with repeats.
- **Adding files later is cheap.** Re-run `ingest` on the folder: new files are
  added, edited ones re-indexed, unchanged ones skipped.
- **Settings stick to the index.** `--chunk-tokens` is recorded on the first
  ingest and reused automatically afterwards.
- **Titles.** Citations show a law's own short title from its naming clause
  (`يسمى هذا القانون (...)`) when it has one, otherwise the filename. Rename a
  file to change how it is cited.

## Arabic documents

Arabic is fully supported, including mixed Arabic/English corpora and English
questions over Arabic documents (and the reverse).

| Stage | How Arabic is handled |
|---|---|
| Embeddings | `bge-m3` is trained on Arabic; spelling variants, colloquial wording and English questions over Arabic laws match without help. |
| Keyword search | Text is normalized before tokenizing: hamza/madda forms of alef fold to bare alef, ta marbuta to ha, alef maqsura to ya; diacritics and tatweel are removed; Arabic-Indic digits become 0-9; the definite article and fused prefixes are stripped. So `الإدارة`, `الادارة`, `إدارة` and `ادارة` are the same term, and `للمادة ١٢` matches `المادة 12`. |
| PDF extraction | Three export defects are detected and repaired automatically, and `ingest` reports each one: words glued together (`اليومالاول`); every line's word order reversed, with PDF glyph codes instead of letters (typical of legal-database exports); and a font bug that turns the `اً` ending into `ﻏ`. Browser print footers (URL, timestamp, page counter) are stripped. |
| Answers | The model answers in the language the question was asked in. |
| Terminal | Output is forced to UTF-8, so `--json` and redirected output stay valid on Arabic-locale Windows. |

Two deliberate choices in the keyword normalizer: a bare leading `و` is **not**
stripped, because in legal vocabulary it is as often a root letter
(`وزارة`, `وثيقة`, `وزير`) as the conjunction "and"; and negations such as
`لا` are **not** stopwords, because in legal text they invert the meaning.

**For an Arabic-only corpus, set `RAG_CHUNK_TOKENS=1200`.** Chunks are sized
with an English-trained tokenizer that splits Arabic into roughly three times
as many tokens as the embedding model does, so the 512 default gives Arabic
chunks a third of the text an English chunk gets — often splitting a single
legal article. 1200 restores comparable chunk sizes and stays under the
embedder's 512-token input limit. Leave the default for mixed-language corpora,
where 1200 would truncate the English chunks.

## Legal documents: article-level retrieval

Laws are asked about and cited by article, so the article is the unit of
retrieval:

- **One chunk per article.** Article headings are detected in all three common
  styles (legal-database exports, Word, official PDFs) and each article becomes
  a chunk; an article longer than `RAG_CHUNK_TOKENS` is windowed within its own
  bounds. A heading only counts if its number continues the sequence, so an
  in-text reference ("under Article 70 of this law") never splits an article.
  Checked against real article counts: 131 for the Constitution, 1,449 for the
  Civil Code.
- **Labelled chunks.** Every chunk is indexed with a label such as
  `قانون العقوبات لسنة 1960 — المادة 170 — الرشوة`, so a passage that never names
  its own law can still be found by it, and the model sees which article it is
  reading. Citations read `قانون العمل لسنة 1996 — المادة 61 (p.22)`.
- **Headings re-attached to their text.** Some legal-database PDFs extract
  every article heading on a page first and all the article bodies after them,
  so no body can be tied to its number. These are rebuilt from layout-mode
  extraction, which keeps each heading above its own text, while taking the
  cleaner letters from plain extraction.
- **Reranked, then budgeted.** A local cross-encoder rescores the top 20
  candidates by reading question and passage together, then as many passages
  as fit in `RAG_CONTEXT_TOKENS` (4,000) go to the model — about ten articles.

## Commands

| Command | What it does |
|---|---|
| `ingest PATHS...` | Index files/directories. Unchanged files are skipped; edited ones are re-chunked and their old chunks dropped. |
| `search QUERY` | Retrieve chunks, with a per-result breakdown of which retriever found it. No API key. |
| `ask QUESTION` | Retrieve, then answer with citations. Streams by default. |
| `chat` | Multi-turn. Each turn retrieves afresh; history stays short because retrieval, not history, carries the context. |
| `stats` / `sources` | What is indexed. |
| `remove SOURCE` / `reset` | Drop one source, or the whole index. |
| `serve` | HTTP API on `:8000`. |

Useful flags: `-k N` (results), `--source SUBSTR` and `--format pdf` (filters),
`--rerank` / `--no-rerank` (cross-encoder reordering), `--provider gemini|claude`, `--json`.

## Library use

```python
from rag import RAGPipeline

rag = RAGPipeline()
rag.ingest(["./docs"])

answer = rag.ask("What is the escalation policy?")
print(answer.text)
for source in answer.cited_sources:
    print(f"  {source.location}  ({source.chunk.source})")

# Retrieval only
for hit in rag.search("quota exceeded", top_k=5):
    print(hit.location, hit.why())     # "deploy.pdf p.3  dense #2 (0.81), bm25 #1 (7.4)"

# Streaming, sources first so a UI can render citations while text arrives
sources, stream = rag.ask_stream("Why did the build fail?")
for delta in stream:
    print(delta, end="")
```

## HTTP API

```bash
python -m rag serve --port 8000     # then open http://127.0.0.1:8000
```

Opening the root URL gives a **test page**: a question box, an *Ask* button
(streams the answer, with each `[n]` clickable to jump to the passage it cites)
and a *Search* button that retrieves passages without using an API key. Each
passage shows its law and article, its dense / BM25 / rerank scores, and the
timing. Arabic renders right-to-left properly here, which a Windows terminal
cannot do. The page is one self-contained file, [rag/web/index.html](rag/web/index.html),
with no CDN, so it works offline; example questions come from the evaluation set.

| Endpoint | |
|---|---|
| `GET /` | the test page |
| `GET /health`, `GET /stats` | index state, provider and which keys were found |
| `GET /examples` | a few questions from the evaluation set, for the page |
| `POST /search` | `{"query": "...", "top_k": 5}` |
| `POST /ask` | `{"question": "...", "history": []}` |
| `POST /ask/stream` | SSE: a `sources` event, then `delta` events, then `done` |
| `POST /ingest` | **disabled by default** — it reads arbitrary server-side paths; enable with `RAG_ALLOW_INGEST=1` behind your own auth |

## Configuration

Every setting is an environment variable or a `Config` field.

| Variable | Default | |
|---|---|---|
| `RAG_INDEX_DIR` | `.rag_index` | where the index lives |
| `RAG_EMBED_MODEL` | `BAAI/bge-m3` | 1024-dim multilingual; `intfloat/multilingual-e5-small` for a light, fast alternative |
| `RAG_DENSE` | `1` | `0` for BM25-only (no torch needed) |
| `RAG_CHUNK_TOKENS` / `RAG_CHUNK_OVERLAP` | `512` / `64` | for laws: the size beyond which one article is windowed |
| `RAG_ARTICLE_CHUNKS` | `1` | one chunk per article for legislation |
| `RAG_ARTICLE_PACK_TOKENS` | `0` | pack short articles together (measured: hurts) |
| `RAG_CONTEXT_HEADERS` | `1` | index each chunk with its "law — article" label |
| `RAG_TOP_K` / `RAG_CONTEXT_TOKENS` | `12` / `4000` | passages considered / token budget sent to the model |
| `RAG_CANDIDATE_K` | `30` | per-retriever recall before fusion |
| `RAG_RERANK` | `1` | cross-encoder reranking of the top candidates |
| `RAG_RERANKER` / `RAG_RERANKER_MODEL` | `local` / `BAAI/bge-reranker-v2-m3` | `llm` spends an API call per query on the configured provider instead |
| `RAG_RERANKER_INT8` | `1` | int8 weights on CPU: 5× faster, same quality (measured) |
| `RAG_RERANK_CANDIDATES` | `20` | how many fused candidates the reranker rescores |
| `RAG_DENSE_WEIGHT` / `RAG_LEXICAL_WEIGHT` | `1.0` / `1.0` | fusion balance — measure before changing |
| `RAG_USE_MMR` / `RAG_MMR_LAMBDA` | `1` / `0.7` | `1.0` = pure relevance, no diversity |
| `RAG_EXTRACT_CACHE` | `<index>/extracted` | extracted-text cache; `off` disables |
| `RAG_LLM_PROVIDER` | `auto` | `gemini` / `claude`; auto picks whichever key is present, preferring Gemini |
| `RAG_MODEL` | per provider | `gemini-3.8-flash` or `claude-opus-5` unless set |
| `RAG_EFFORT` | `medium` | `low`–`max`; maps onto Gemini's thinking levels |
| `RAG_RETRY_ATTEMPTS` / `RAG_RETRY_SECONDS` | `2` / `1.5` | retries for transient 503 / 429, with doubling backoff |
| `VOYAGE_API_KEY` | — | if set, uses Voyage embeddings instead of local weights |

For a fast setup on a slow machine: `RAG_EMBED_MODEL=intfloat/multilingual-e5-small`
and `RAG_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — indexing in
about 2 minutes instead of 14, and ~0.3 s per query instead of ~2 s, for "right
passage ranked first" of 63% instead of 80% on the legal test set.

## Evaluation

Retrieval quality is the part of a RAG system that actually moves answer
quality, and it is measurable without touching the generator.

```bash
python eval/evaluate.py                                   # legal test set, compare modes
python eval/evaluate.py --modes hybrid "hybrid + rerank" --show-misses
python eval/evaluate.py --sweep                           # tune fusion weights
python eval/evaluate.py --generate 20                     # draft cases with the model
```

`eval/laws_questions.jsonl` holds 49 questions over the laws in `documents/`:
41 Arabic, 5 English, 3 article-number lookups. Each is tied to the article
that answers it, and a hit requires **both** the right law and the answering
text in the passage. The questions were written by the same person who built
the system, so treat the absolute numbers as optimistic; the comparisons
between versions are the useful part.

What each change bought, measured in sequence:

| step | ranked 1st | answer within 1,000 tokens of context |
|---|---|---|
| starting point (1,200-token page windows, e5-small) | 43% | 43% |
| one chunk per article, labelled with law and article | 47% | 61% |
| + local cross-encoder reranker (int8) | 76% | 84% |
| + bge-m3 embeddings | **80%** | **90%** |

With the shipped defaults: recall@5 94%, MRR 0.86, English questions 5/5,
article lookups 3/3. Things that were tried and measured *worse* were left
out: packing short articles together, `multilingual-e5-large` (71% ranked
first after reranking), and the reranker at full precision (same quality as
int8 at 2.3× the latency).

## Design notes

**Why a file-based index.** A dedicated vector database earns its operational
cost somewhere north of a million chunks. Below that, an in-memory `float32`
matrix and an exact dot product are faster than an approximate index, exactly
accurate, and trivial to back up, diff, or delete. The retrieval interface is
narrow enough that swapping in FAISS or pgvector means reimplementing
`VectorStore.search_dense` and nothing else.

The matrix is read into RAM rather than memory-mapped. Every query multiplies
the whole matrix, so a mapping saves no I/O — it only adds an open file handle,
which on Windows makes a running `serve` block the next `ingest` from saving.

**Chunking is the highest-leverage part.** Legislation is split per article
(above). Everything else is packed to a token budget along the strongest
available boundary — blank line, then line, then sentence, then word — and each
chunk re-opens with the tail of the previous one so a fact straddling a
boundary stays retrievable. When a paragraph is larger than the overlap budget,
a *partial* tail is carried instead of silently dropping the overlap to zero.

**Compare retrieval at equal context, not equal k.** "Recall@5" rewards big
chunks: a 1,200-token page window "hits" whenever any of the half-dozen
articles on it answers the question. Measured that way, per-article chunks
looked *worse* than page windows (65% vs 80%). Measured by what actually
matters — how often the answer reaches the model for the same number of
tokens — they are better at every budget (84% vs 69% at 4,000 tokens). The
evaluation harness reports both.

**Extraction is cached.** Extracted text is stored under the index, keyed by a
hash of each file's bytes, so re-ingesting after a settings change (or into a
second index, with `RAG_EXTRACT_CACHE` pointing at a shared folder) skips the
slow layout-mode PDF extraction.

**Prompt caching.** Retrieved context sits in the system block behind a cache
breakpoint, so follow-up turns over the same retrieval reuse it. Below the
model's minimum cacheable prefix the parameter is simply inert — check
`answer.usage["cache_read_input_tokens"]` to see whether it engaged.

**Graceful degradation.** Missing torch, a failed model download, or
`RAG_DENSE=0` drops the system to BM25-only with a warning instead of failing.
A missing API key blocks answer generation and nothing else.

## Tests

```bash
python -m pytest tests -q     # 187 tests, ~2s, no network, API key or model weights
```

Tests use a deterministic fake embedder, so they exercise the plumbing —
persistence, fusion, filtering, offset integrity, citation parsing — without
model weights. Embedding *quality* is what `eval/` is for.

## Documentation

[docs/RAG-system-explained.pdf](docs/RAG-system-explained.pdf) is a 19-page
walkthrough of the whole system: the architecture, every pipeline stage, how
Arabic is repaired, how the index is laid out on disk, where data travels (and
what leaves the machine), the interfaces, the measurements, and a glossary.

[docs/RAG-code-walkthrough.pdf](docs/RAG-code-walkthrough.pdf) is its
companion: 97 pages that print all 4,145 lines of `rag/` with line numbers, in
blocks, each block followed by what it does and why.

```bash
python docs/snapshot.py        # re-read counts, sizes and versions from the code
python docs/make_pdf.py        # rebuild the architecture PDF from that snapshot
python docs/make_code_pdf.py   # rebuild the line-by-line walkthrough
```

Both are generated from the code, not written alongside it. The numbers in the
first are read from the running system; the second prints the source files
themselves, and `walkthrough.check_coverage` aborts the build unless the
annotation blocks tile every module exactly — no gaps, no overlaps. So a module
that grows a function without a note for it fails the build rather than
quietly becoming a document that claims more than it covers.

## Known limitations

- **Providers have bad minutes.** Gemini returned a 503 once in a handful of
  real requests here. Transient failures (503, 429) are retried with a
  doubling backoff, but a stream that breaks *after* text has arrived cannot be
  retried — the answer is kept, marked "cut short", and you retry the question.
- **Scanned PDFs need OCR.** Text extraction reads embedded text only; a
  scanned page yields nothing. Pre-process with OCR before indexing.
- **Some legal-database PDFs contain font junk.** A few glyphs decode to runs
  of random letters (`صعودئصثغئطضت...` where `صعوداً` belongs) -- about
  0.1-3% of words, depending on the file. The word stem before the junk is
  intact, so retrieval mostly survives it, but quoted text can show it.
- **One-off commands are slow; sessions are not.** Each `python -m rag search`
  or `ask` loads both models from disk first (~16 s on the test machine) and
  then searches in ~2 s. Use `chat` or `serve`, which load them once.
- **First indexing takes a while on CPU.** bge-m3 embedded the 12 laws
  (3,830 articles) in about 14 minutes. It is a one-time cost: re-ingesting
  only processes new or changed files.
- **Vocabulary gaps remain.** The three test questions still missed ask in
  everyday words for things the law names differently (`وسائل الإثبات` for
  `البينات`), or target one entry inside a very long definitions article.
- **Arabic PDF repair is not perfect.** When a PDF places two words with no
  measurable gap between them, they stay joined (about 3% of words on the test
  PDF). Multi-column Arabic layouts may also come out in the wrong column
  order. Spot-check files that `ingest` reports as repaired.
- **Keyword stemming is light.** Only prefixes are stripped, so plural and
  suffixed forms (`موظف` / `الموظفين`) do not match by keyword. Dense retrieval
  covers most of this gap.
- **CSV files are capped at 2000 rows.** Large tables belong in a database with
  a SQL tool, not in a vector index.
- **Re-indexing is per-file.** Editing one line re-chunks and re-embeds that
  whole file.
- **Changing the embedding model invalidates the index.** `stats` warns when
  the fingerprint drifts; re-run `ingest --force`.
- **Writes are single-writer.** Two processes ingesting into the same index
  concurrently will have one overwrite the other; a running `serve` alongside
  `ingest` is fine, but it keeps serving the index it loaded at startup until
  restarted.
# Mizan

