"""Documentation sections 4-12: the pipeline, stage by stage."""

from __future__ import annotations

from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph

from make_pdf import DIAGRAM, S, SNAPSHOT, ar, bullets, code, p, panel, table

W = 165 * mm


def _cols(*fractions: float) -> list[float]:
    return [W * f for f in fractions]


def section_reading() -> list:
    rows = [
        ["Format", "How it is read", "Notes"],
        ["PDF", "pypdf, plain and layout modes",
         "Page boundaries kept as character offsets, so a citation can name a page"],
        [".docx", "zipfile + ElementTree over word/document.xml",
         "Paragraphs, tables (one row per line) and footnotes, in document order"],
        ["txt, md, log, source code", "Read as UTF-8", "Whitespace normalised"],
        ["csv, tsv", "csv.DictReader", "One record per block, capped at 2,000 rows"],
        ["json, jsonl", "json module", "Pretty-printed, or one record per line"],
        ["html", "Tag stripping", "Scripts and styles dropped, entities decoded"],
    ]
    doc_shape = """
Document:
    doc_id      "9f12c4..."           stable id derived from the absolute path
    source      "documents/law.pdf"   where it came from
    title       "Labour Law 1996"     what citations will show
    text        "- Article 1 ..."     the whole document as one string
    metadata    {format, pages, content_hash, repairs applied}
    page_spans  [(0, 2480, "1"), (2480, 5090, "2"), ...]
                  start  end   page   <- maps any character back to a page
"""
    return [
        p("4  Stage 1 - Reading the documents", "h1"),
        p("Every format is funnelled into one shape, so nothing downstream needs to know where "
          "the text came from:"),
        code(doc_shape),
        table(rows, _cols(0.2, 0.3, 0.5)),
        p("Three things happen around the reading itself", "h2"),
        *bullets([
            "<b>Extraction is cached.</b> Layout-mode PDF extraction is the slowest part of "
            "ingestion. Its result is stored under the index, keyed by a SHA-256 hash of the "
            "file's bytes, so re-indexing after a settings change skips it entirely. The cache "
            f"currently holds {SNAPSHOT['extract_cache_files']} entries. Change one byte of a "
            "file and the hash changes, forcing a fresh read.",
            "<b>Identical files are indexed once.</b> Two of the supplied PDFs were the same "
            "file under different names. The second is detected by content hash, reported and "
            "skipped - otherwise every answer would quote it twice and spend two of its ten "
            "passages on one document.",
            "<b>Deleted files are removed.</b> Ingesting a folder first drops index entries "
            "whose file no longer exists there, so a deleted law stops appearing in answers, "
            "and a renamed file is re-indexed under its new name instead of being mistaken for "
            "a duplicate of its own former name.",
        ]),
        p("Where citation titles come from", "h2"),
        p("Eleven of the twelve PDFs carry the same embedded title - the name of the database "
          "they were exported from - which would make every citation identical. The loader "
          "therefore looks for the law's own naming clause, the standard formula in Article 1 "
          "of Arab legislation:"),
        table([[Paragraph(ar("يسمى هذا القانون ( قانون العمل لسنة 1996 ) ويعمل به"), S["cellar"]),
                Paragraph("the title becomes <b>Labour Law of 1996</b>", S["cell"])]],
              _cols(0.55, 0.45), header=False),
        p("With no such clause the filename is used, which is why renaming a file changes how "
          "it is cited.", "small"),
    ]


def section_arabic() -> list:
    defects = [
        ["Defect", "What extraction produced", "After repair"],
        ["Words glued together",
         Paragraph(ar("اليومالاولتعريفبمقدم"), S["cellar"]),
         Paragraph(ar("اليوم الاول تعريف بمقدم"), S["cellar"])],
        ["Word order reversed on every line",
         Paragraph(ar(".للضرر تعرض شخص أي :المتضرر"), S["cellar"]),
         Paragraph(ar("المتضرر: أي شخص تعرض للضرر."), S["cellar"])],
        ["Glyph codes, and a font bug on the tanween ending",
         Paragraph(ar("وﻓﻘﻏ لاحكام"), S["cellar"]),
         Paragraph(ar("وفقا لاحكام"), S["cellar"])],
        ["Bold words drawn twice and fused",
         Paragraph(ar("العقارالعقار"), S["cellar"]),
         Paragraph(ar("العقار"), S["cellar"])],
    ]
    normal = [
        ["Written as", "Also written as", "Indexed as"],
        [Paragraph(ar("الإدارة"), S["cellar"]),
         Paragraph(ar("الادارة / إدارة / ادارة"), S["cellar"]),
         Paragraph(ar("اداره"), S["cellar"])],
        [Paragraph(ar("للمادة ١٢"), S["cellar"]),
         Paragraph(ar("المادة 12"), S["cellar"]),
         Paragraph(ar("ماده 12"), S["cellar"])],
        [Paragraph(ar("مستشفى"), S["cellar"]),
         Paragraph(ar("مستشفي"), S["cellar"]),
         Paragraph(ar("مستشفي"), S["cellar"])],
    ]
    return [
        p("5  Stage 2 - Repairing Arabic text", "h1"),
        p("This stage exists because the supplied PDFs do not extract cleanly. Eleven of the "
          "twelve came from a legal database that writes Arabic into the PDF as positioned "
          "glyphs in display order. Nothing downstream survives that: the keyword index would "
          "hold nonsense words, the embeddings would describe nonsense, and any quote in an "
          "answer would be unreadable."),
        table(defects, _cols(0.3, 0.35, 0.35)),
        p("Each defect is detected rather than assumed, so a well-formed PDF is never "
          "touched:", "small"),
        *bullets([
            "<b>Glued words</b> are found by average word length - Arabic words average four "
            "to six letters, and a page averaging above nine is run-together text.",
            "<b>Reversed word order</b> is found by punctuation: normal text ends lines with a "
            "full stop, reversed text starts them with one. Whole lines are counted rather "
            "than tokens, because reversed clause numbering (\"1. x\" arriving as \"x. 1\") "
            "would otherwise look normal.",
            "<b>Glyph codes</b> (the Unicode presentation forms, U+FB50-U+FEFF) are folded "
            "back to ordinary letters. One font maps the tanween ending to the wrong code "
            "point; it is recognisable because the character it produces cannot legally end an "
            "Arabic word.",
            "<b>Print footers</b> - a browser URL, a timestamp and a page counter on every "
            "page - are stripped. They had been landing in every passage, diluting both "
            "searches.",
        ]),
        p("The two-extraction trick", "h2"),
        p("Plain extraction gets the letters right but loses the spaces and detaches every "
          "article heading from its text. Layout extraction keeps the spacing and the "
          "structure but decodes some glyphs into longer junk. Neither is usable alone, so the "
          "loader takes the <i>order</i> from layout mode and the <i>letters</i> from plain "
          "mode, matching the two line by line on shared words. On the Labour Law this dropped "
          "the junk rate from 2.06% to 0.10%."),
        p("Normalisation for search", "h2"),
        p("Separately from repair, Arabic is normalised before being indexed or searched, so "
          "that the several valid spellings of a word become one term. The same function is "
          "applied to documents and to questions, so it only has to be consistent:"),
        table(normal, _cols(0.33, 0.37, 0.3)),
        p("Diacritics and tatweel are removed, Arabic-Indic digits become 0-9, and the "
          "definite article and fused prepositions are stripped. Two deliberate exceptions: a "
          "leading waw is kept, because in legal vocabulary it is as often part of the word "
          "(wizara, ministry) as the conjunction \"and\"; and negations are kept out of the "
          "stopword list, because in law they invert the meaning."),
    ]


def section_articles() -> list:
    accept = """
def _accept(number, bis, previous):
    if previous is None or number == 1:    # first article, or an annex restarting
        return True
    if bis and number == previous:         # "Article 42 bis"
        return True
    return previous < number <= previous + MAX_FORWARD_JUMP   # gaps for repealed articles
"""
    counts = [
        ["Document", "Articles found", "Actual length of the law"],
        ["Constitution (Word file)", "131", "131"],
        ["Civil Code", "1,449", "1,449"],
        ["Penal Code", "479", "476 plus bis articles"],
        ["Code of Criminal Procedure", "371", "370"],
        ["Companies Law", "291", "289 plus bis articles"],
        ["Labour Law", "142", "142"],
    ]
    return [
        p("6  Stage 3 - Finding the articles", "h1"),
        p("Laws are quoted by article, not by page, so the article is the unit the system "
          "works in. Three heading styles cover every document in this corpus:"),
        *bullets([
            "<font name='Mono' size='8.5'>- Article 61 &lt;topic&gt; effective-date 2019-05-16</font> "
            "- legal-database exports",
            "<font name='Mono' size='8.5'>Article 7</font> alone on a line - the Word file and "
            "most official PDFs",
            "<font name='Mono' size='8.5'>Article 4- first sentence...</font> - heading and "
            "text sharing a line",
        ]),
        p("The hard part is not finding headings but rejecting false ones: legal text is full "
          "of sentences like \"under Article 70 of this law\". The guard is that article "
          "numbers run in order, so a candidate is accepted only if it continues the sequence. "
          "A mid-sentence reference almost never does."),
        code(accept),
        p("That the rule works is checkable against the real laws, which is the strongest "
          "evidence available:"),
        table(counts, _cols(0.4, 0.25, 0.35)),
        panel("<b>Why this mattered so much.</b> In the database exports, plain extraction put "
              "every article number on a page into one block and all the article texts after "
              "them. Numbers and texts were both present but no longer connected, so any "
              "answer citing an article number would have been guessing. Rebuilding the "
              "pairing from layout-mode extraction is what makes a citation such as "
              "\"Article 170 of the Penal Code\" trustworthy."),
    ]


def section_chunking() -> list:
    chunk_shape = """
Chunk:
    text        "- Article 61 ... fourteen days of annual leave ..."   the article itself
    header      "Labour Law 1996 - Article 61"    prepended for indexing and for the model
    char_start  18423   char_end 19877            offsets back into the document
    pages       ["22"]                            for the citation
    metadata    {article: "61", article_number: 61, topic: ..., format: "pdf"}
    token_count 312

    indexed_text == header + newline + text    <- what is embedded and keyword-indexed
"""
    return [
        p("7  Stage 4 - Chunking and labelling", "h1"),
        p("A passage is one article. Articles longer than the size limit are split into "
          "overlapping windows inside their own boundaries, so a chunk never straddles two "
          f"articles. Of the {SNAPSHOT['index']['chunks']:,} passages in the index, "
          f"{SNAPSHOT['article_chunks']:,} carry an article label; the rest are title pages "
          "and preambles."),
        code(chunk_shape),
        p("Three details carry a lot of weight", "h2"),
        *bullets([
            "<b>The label is part of what is searched.</b> An article rarely names its own law "
            "- it says \"a worker is entitled to...\", not \"under the Labour Law a worker...\". "
            "Prepending \"Labour Law 1996 - Article 61\" lets a question that names the law "
            "find it, and lets the model see which article it is reading.",
            "<b>Offsets are kept.</b> Every chunk knows its character range in the source "
            "document. That is what lets a citation name a page, and what lets any quote be "
            "checked against the original file.",
            "<b>Articles are not packed together.</b> Grouping short articles into bigger "
            "chunks was tried and measured worse at every context size (section 16), so each "
            "article stands alone.",
        ]),
        panel("<b>A trap worth knowing about.</b> Chunk sizes are measured with tiktoken, an "
              "English-trained tokenizer that splits Arabic into roughly three times as many "
              "tokens as the embedding model does. The default of 512 therefore gave Arabic "
              "chunks a third of the text an English chunk gets, often cutting a legal article "
              "in half. This corpus is indexed at 1,200, which restores comparable sizes and "
              "still fits inside the embedder's 512-token input window."),
    ]


def section_embedding() -> list:
    return [
        p("8  Stage 5 - Embedding", "h1"),
        p("An embedding is a list of numbers describing what a piece of text means. Texts with "
          "similar meanings get similar lists, so \"annual leave\" lands near \"fourteen days "
          "of paid leave per year\" even though they share no words. Here every passage "
          f"becomes {SNAPSHOT['index']['dim']:,} numbers, produced by "
          f"{SNAPSHOT['index']['embedder']} running locally on the CPU."),
        *bullets([
            "<b>Pooling.</b> The model emits one vector per token, and they must be reduced to "
            "one vector per passage. BGE models are trained to be read from their first token, "
            "most others from the average of all tokens. Choosing wrong does not fail - it "
            "quietly retrieves worse - so the choice follows the model name.",
            "<b>Normalisation.</b> Every vector is scaled to length 1, which turns comparing "
            "two vectors into a single multiplication (a dot product).",
            "<b>Asymmetric prefixes.</b> Some models (the e5 family) are trained with "
            "\"query:\" and \"passage:\" prefixes and retrieve worse without them; bge-m3 is "
            "not, so prefixes are applied only when the model name calls for them.",
            "<b>Length-sorted batching.</b> A batch is padded to its longest member, and this "
            "corpus mixes one-line articles with full pages. Sorting by length before batching "
            "cut indexing from 211 s to 135 s on the smaller model.",
        ]),
        p(f"Indexing all {SNAPSHOT['index']['chunks']:,} passages with bge-m3 took about 14 "
          "minutes on this machine's CPU - a one-time cost, since re-ingesting only processes "
          "new or changed files. Embedding a question takes about 59 ms."),
    ]


def section_index() -> list:
    files = SNAPSHOT["index_files"]
    rows = [
        ["File", "Size", "What is in it"],
        ["chunks.jsonl", f"{files.get('chunks.jsonl', 0)/1e6:.1f} MB",
         "One JSON object per passage per line: text, header, offsets, pages, metadata. "
         "Line N corresponds to row N of the vector matrix."],
        ["embeddings.npy", f"{files.get('embeddings.npy', 0)/1e6:.1f} MB",
         f"A NumPy array of {SNAPSHOT['index']['chunks']:,} x {SNAPSHOT['index']['dim']} "
         "float32 values: the passage vectors"],
        ["meta.json", f"{files.get('meta.json', 0)/1024:.0f} KB",
         "Which embedding model and chunk settings built this index, plus a record per source "
         "file (content hash, chunk count, repairs applied)"],
        ["extracted/", f"{SNAPSHOT['extract_cache_files']} files",
         "Cached extraction results, keyed by file content hash"],
    ]
    return [
        p("9  The index on disk", "h1"),
        p("The index is three files in a folder. There is no database process, nothing to "
          "install and nothing to administer; the whole thing can be copied, backed up, "
          "inspected in a text editor, or deleted."),
        table(rows, _cols(0.2, 0.13, 0.67)),
        p("Invariants that keep it honest", "h2"),
        *bullets([
            "<b>Row N of the matrix is line N of the chunks file.</b> Everything depends on "
            "this. Deleting a source rebuilds both together; adding passages without an "
            "embedder deletes the vectors rather than leaving a half-covered matrix.",
            "<b>Writes are atomic.</b> Each file is written under a temporary name and then "
            "renamed, so an interrupted save cannot leave a half-written index. If the counts "
            "disagree on load, the vectors are dropped and keyword search still works.",
            "<b>The index records how it was built.</b> If the embedding model or the chunk "
            "settings change, the mismatch is reported instead of silently producing bad "
            "results - vectors from two different models cannot be compared.",
            "<b>The matrix is read into RAM, not memory-mapped.</b> Every query multiplies the "
            "whole matrix anyway, so a mapping saves no work; and on Windows an open mapping "
            "blocks the next save, which was a real bug here.",
        ]),
    ]


def section_retrieval() -> list:
    rrf = """
def reciprocal_rank_fusion(ranked_lists, k=60, weights=None):
    fused = {}
    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx] if weights else 1.0
        for rank, (doc_idx, _score) in enumerate(ranked):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank + 1)
    return fused
"""
    return [
        p("10  Stage 6 - Retrieval", "h1"),
        p("Two searches run over the index, because they fail in different places."),
        p("Keyword search (BM25)", "h2"),
        p("BM25 scores a passage by how often the question's words appear in it, weighted so "
          "that rare words count for more (a passage containing \"tanfeeth\" is more "
          "informative than one containing \"the\"), with diminishing returns for repetition "
          "and a correction for passage length. It is exact: it finds article numbers, names "
          "and defined terms that an embedding may blur. It cannot match a paraphrase, and it "
          "cannot match across languages at all. Searching all "
          f"{SNAPSHOT['index']['chunks']:,} passages takes about 0.1 ms."),
        p("Meaning search (dense)", "h2"),
        p("The question is embedded with the same model as the passages, and the best matches "
          "are the passages whose vectors point most nearly in the same direction - a cosine "
          "similarity, which for normalised vectors is one multiplication of the question "
          "vector against the whole matrix. It matches paraphrases and crosses languages; it "
          "is vaguer about exact identifiers. About 59 ms, dominated by embedding the question."),
        p("Fusion", "h2"),
        p("The two rankings cannot be added directly: a cosine similarity of 0.87 and a BM25 "
          "score of 24.2 are not on the same scale, and any attempt to weight them needs "
          "per-corpus calibration that goes stale. Reciprocal Rank Fusion instead uses only "
          "the <i>positions</i>: each list contributes 1/(60 + rank), and a passage that both "
          "searches rank highly wins."),
        code(rrf),
        p("Diversification and filtering", "h2"),
        *bullets([
            "<b>MMR</b> (Maximal Marginal Relevance) walks down the fused list and skips "
            "passages that are near-copies of ones already chosen, so the ten passages sent to "
            "the model are ten different articles rather than one article and nine variations "
            "of it. The top hit is always kept.",
            "<b>Filters</b> restrict a search to one file or one format. They are applied "
            "before truncation - filtering after cutting to ten would silently return fewer "
            "than ten.",
        ]),
    ]


def section_rerank() -> list:
    rows = [
        ["Setup", "Answer ranked first", "Answer within 1,000 tokens", "Time per question"],
        ["No reranking", "47%", "59%", "0.2 s"],
        ["Small model (118M), int8", "63%", "82%", "0.3 s"],
        ["Large model (568M), float32", "73%", "82%", "5.2 s"],
        ["<b>Large model, int8 (default)</b>", "<b>76%</b>", "<b>84%</b>", "<b>2.2 s</b>"],
    ]
    return [
        p("11  Stage 7 - Reranking and the context budget", "h1"),
        p("Retrieval compares the question and each passage <i>separately</i> - one embedding "
          "per side, or counted words - which is what makes it fast enough to search thousands "
          "of passages. It is good at getting the right passage into the top twenty and only "
          "approximately right about the order within them."),
        p("A cross-encoder reads the question and one passage <i>together</i> and scores how "
          "well that passage answers that question. It is far more accurate and far too slow "
          "to run over a whole corpus, so it only reorders the twenty candidates that fusion "
          "already found. This was the single biggest quality gain in the project:"),
        table(rows, _cols(0.34, 0.2, 0.26, 0.2)),
        p("Two optimisations made the large model usable on a CPU: int8 quantisation (storing "
          "the model's weights as 8-bit integers instead of 32-bit floats) and sorting the "
          "batch by passage length so that less padding is computed. Together they took it "
          "from 11.4 s to 2.2 s per question with no loss of quality - the quantised model "
          "measured marginally better, which is within noise.", "small"),
        p("The context budget", "h2"),
        p("Passages are then taken in order until about 4,000 tokens are filled - roughly ten "
          "articles - rather than a fixed count. Article lengths vary more than tenfold here, "
          "so \"ten passages\" can mean 800 tokens or 9,000. Budgeting by tokens makes the "
          "cost and the latency of the next step predictable, and section 16 shows it also "
          "retrieves better."),
    ]


def section_generation() -> list:
    prompt = """
system:   You answer questions using only the excerpts provided in <context>.
          - Ground every factual claim in the excerpts...
          - Cite with bracketed numbers matching the excerpt ids, e.g. [2]
          - When an excerpt's location names a law and an article, name them...
          - If the excerpts do not contain the answer, say so plainly...
          - Answer in the language the question was asked in.

          <context>
          <excerpt id="1" source="documents/qistas.pdf"
                   location="Labour Law 1996 - Article 61 (p.22)">
          - Article 61 ... fourteen days of annual leave ...
          </excerpt>
          ... nine more ...
          </context>

user:     How many days of annual leave does a worker get?
"""
    provider_rows = [
        ["Concern", "Claude", "Gemini"],
        ["Assistant turns are called", "assistant", "model"],
        ["Thinking", "adaptive, with an effort level", "always on, set by level, cannot be disabled"],
        ["Prompt caching", "explicit breakpoint on the context", "automatic for repeated prefixes"],
        ["A refusal arrives as", "stop_reason = refusal", "a finish_reason on the candidate"],
        ["Default model", "claude-opus-5", "gemini-3.8-flash"],
    ]
    return [
        p("12  Stage 8 - Generating the answer", "h1"),
        p("Only this stage leaves the machine. The retrieved passages are numbered and labelled "
          "with their law and article, and the instructions are explicit about grounding:"),
        code(prompt),
        p("What the instructions buy", "h2"),
        *bullets([
            "<b>Citations that resolve.</b> The <font name='Mono' size='8'>[n]</font> markers "
            "are parsed back out of the answer afterwards, so the caller gets a structured "
            "list of which passages were actually used - not prose that happens to contain "
            "brackets. On the web page they are clickable.",
            "<b>Refusal as a feature.</b> The model is told to say the passages do not contain "
            "the answer rather than fill the gap. A fluent guess is worse than no answer, "
            "because it is indistinguishable from a real one.",
            "<b>Answer in the question's language.</b> Ask in Arabic, get Arabic; ask in "
            "English about an Arabic law, get English.",
        ]),
        p("One interface, two providers", "h2"),
        p("Both providers sit behind one small interface: a list of system parts and a list of "
          "messages in, text or a stream of text out. Everything provider-shaped stays inside "
          "its backend, so the prompt, the citation parsing and the refusal handling are "
          "written once."),
        table(provider_rows, _cols(0.26, 0.37, 0.37)),
        p("Transient failures (503 overloaded, 429 rate limited) are retried with a doubling "
          "backoff; a 4xx is not retried, because it will not improve. A stream that breaks "
          "after text has already arrived cannot be retried either - the partial answer is "
          "kept, marked \"cut short\", and the question can be asked again."),
    ]
