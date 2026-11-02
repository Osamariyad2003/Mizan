"""Line-by-line notes for rag/loaders.py."""

SUMMARY = (
    "Turns a file on disk into plain text plus the page map that makes citations possible. "
    "Most of the length is PDF repair: the Arabic legal PDFs in this corpus arrive glued, "
    "reversed, or with their article numbers detached from their text."
)

BLOCKS = [
    (1, 7,
     "The docstring names the contract every loader below honours: return a "
     "<font name='Mono' size='8'>Document</font> whose text is the whole file, and a list of "
     "<font name='Mono' size='8'>page_spans</font> saying which character range came from which "
     "page. Without that second list a PDF citation could name the document but not the page, "
     "which is the difference between a usable citation and a decorative one."),
    (8, 18,
     "Imports. <font name='Mono' size='8'>hashlib</font> provides both the document id and the "
     "cache key, <font name='Mono' size='8'>csv</font> and <font name='Mono' size='8'>json</font> "
     "serve their own loaders, and <font name='Mono' size='8'>logging</font> is here only to "
     "silence pypdf on line 51."),
    (19, 30,
     "Everything imported from <font name='Mono' size='8'>arabic</font> is a repair or a "
     "detector. Keeping them in their own module means this file decides <i>when</i> to repair "
     "text and that module decides <i>how</i>, which is what lets the repairs be tested on "
     "strings without ever opening a PDF."),
    (31, 39,
     "Two sets of extensions. <b>32-36</b> are files read as plain text, including source code, "
     "so the system can index a codebase as happily as a statute. <b>37-39</b> add the formats "
     "that need a real parser. Membership of this second set is what "
     "<font name='Mono' size='8'>discover</font> checks before offering a file for indexing."),
    (40, 46,
     "<b>42-45</b> are directories to walk straight past. Without this, pointing the indexer at "
     "a project directory would index its virtual environment and its own "
     "<font name='Mono' size='8'>.rag_index</font> folder - a system indexing its own index. "
     "<b>46</b> caps a single file at 32 MB, which keeps one pathological file from stalling an "
     "ingest."),
    (47, 51,
     "A one-line fix for a noisy dependency. pypdf warns once per malformed cross-reference "
     "entry it recovers from, and exported PDFs have thousands; the warnings are harmless and "
     "they bury the actual ingest progress, so the library's logger is raised to ERROR."),
    (52, 62,
     "The <font name='Mono' size='8'>Document</font> dataclass: the single shape every loader "
     "returns, so the rest of the system never learns which format a document came from. "
     "<b>60</b> carries format-specific extras - page counts, which pages needed Arabic repair - "
     "as a free-form dictionary rather than as fields that would be empty for most formats. "
     "<b>62</b> is the page map, deliberately empty for formats that have no pages."),
    (63, 68,
     "Given a character offset, which page was it on. A plain linear scan: a 300-page document "
     "has 300 spans, and this is called once per chunk, so the obvious implementation is also "
     "the right one."),
    (69, 78,
     "The range version, used when a chunk spans a page break. <b>73-75</b> is an interval "
     "overlap test, not containment, so a chunk that starts near the bottom of page 4 and ends "
     "on page 5 reports both. <b>76-77</b> remove duplicates while keeping page order, which "
     "matters because \"pages 4-5\" reads correctly and \"pages 5, 4\" does not."),
    (79, 82,
     "The document id is a hash of the resolved absolute path. Deriving it rather than counting "
     "means re-ingesting a file produces the same id, which is what lets the store replace that "
     "document's chunks instead of adding a second copy."),
    (83, 93,
     "The whitespace repairs every format needs. <b>86-87</b> normalise line endings and delete "
     "the two invisible characters that most often survive extraction: a non-breaking space and "
     "a byte-order mark. <b>89</b> rejoins a word split across a line break by hyphenation, so "
     "\"inter-\\nnational\" is searchable as one word. <b>90-91</b> collapse runs of spaces and "
     "limit blank lines to one, which preserves paragraph boundaries - the chunker's strongest "
     "split point - without letting a dozen blank lines count as a dozen boundaries."),
    (94, 98,
     "A banner comment. The loaders below are each self-contained: one format, one function, "
     "returning the same Document."),
    (99, 108,
     "Plain text, the simple case. <b>100</b> uses "
     "<font name='Mono' size='8'>errors=\"replace\"</font> so a file saved in another encoding "
     "yields a few replacement characters rather than aborting the ingest. <b>106</b> records "
     "the extension as the format."),
    (109, 130,
     "The PDF loader begins. <b>111-114</b> import pypdf lazily and translate the import error "
     "into an instruction, because \"pip install pypdf\" is more useful than a traceback. "
     "<b>120-123</b> wrap each page: one malformed page in a 200-page statute must not lose the "
     "other 199. <b>125</b> is ordering that matters - glyph codes are folded into real letters "
     "<i>before</i> anything measures the text, because every check below counts words. "
     "<b>126-129</b> attempt the Arabic spacing repair and record which pages needed it."),
    (131, 135,
     "Word order is decided once for the whole document, not per page. A PDF comes from one "
     "producer, so either all its pages are stored in display order or none are; and a single "
     "page may hold too few sentences to judge reliably. Deciding once removes that coin-flip."),
    (136, 153,
     "Assembling the document and its page map. <b>140-145</b> run only for a reversed PDF: each "
     "line is flipped back into reading order, then <font name='Mono' size='8'>_order_like_layout"
     "</font> is given the same page extracted a second way. <b>146</b> normalises. "
     "<b>149-152</b> are the page map being built: each page contributes a block, and the span "
     "recorded is exactly the character range that block occupies in the final string - which is "
     "why the cursor is advanced by the block length including its trailing blank line."),
    (154, 168,
     "Metadata and the finished document. <b>155</b> reads the PDF's embedded title defensively, "
     "since the metadata object may be absent entirely. <b>157-160</b> record that a repair "
     "happened and on which pages - that is what lets the CLI print \"repaired Arabic spacing on "
     "12 pages\" instead of silently changing the text. <b>164</b> prefers the embedded title and "
     "falls back to the filename."),
    (169, 181,
     "Layout-mode extraction, which preserves the visual position of text on the page. "
     "<b>177-179</b> carry a subtle ordering rule: the line is reversed <i>before</i> glyph codes "
     "are folded into letters, because a lam-alef ligature is a single glyph code at this stage "
     "and must stay one character while the line is reversed - fold first and it becomes two "
     "letters that the reversal puts in the wrong order."),
    (182, 184,
     "The word pattern used for matching below: ASCII word characters plus the Arabic block, so "
     "a mixed Arabic and Latin line tokenises sensibly."),
    (185, 199,
     "The function that fixes the hardest defect in this corpus. The docstring states it exactly: "
     "plain extraction of some legal-database PDFs emits every article heading on a page first "
     "and every article body afterwards, so no body can be attached to its number - the document "
     "is intact but unusable. Layout extraction keeps each heading above its own text but "
     "decodes some glyphs worse. So the <i>order</i> is taken from layout and the <i>text</i> "
     "from plain. <b>199</b> builds the pool of plain lines as word sets, ready to match."),
    (200, 219,
     "The matching loop. <b>203-205</b> leave heading lines in their layout form, because the "
     "article splitter depends on that word order. <b>208-213</b> score every unused plain line "
     "by Jaccard overlap - shared words over total words - and keep the best. <b>214-216</b> "
     "accept only at 0.6 or better and mark the line used, so two layout lines cannot both claim "
     "the same plain line. <b>217-218</b> keep the layout line when nothing matches well enough, "
     "which is the safe failure: slightly worse glyphs, never wrong text."),
    (220, 223,
     "The heading pattern, written with explicit escapes: a dash, the Arabic word for \"article\", "
     "then a number, which is how these exports print a heading."),
    (224, 234,
     "Patterns for print noise. Browser print footers carry the source URL with its query string "
     "and a timestamp; page counters read \"3 of 40\". Both repeat on every single page, so they "
     "contribute nothing anyone would search for while diluting every chunk they land in. "
     "<b>231</b> accepts Arabic-Indic digits as well as ASCII ones."),
    (235, 245,
     "Applying those patterns. <b>240</b> deliberately requires both a URL-ish fragment "
     "<i>and</i> at least four query-string characters, so a line that merely mentions a website "
     "survives and only a machine-generated footer is dropped. <b>242</b> likewise bounds the "
     "page-counter test to short lines, so a sentence containing \"3 of 40\" is kept."),
    (246, 261,
     "The spacing repair. The docstring explains the defect and the remedy: with glyphs stored "
     "in display order, plain extraction loses the spaces between words and layout extraction "
     "keeps them but reverses every line. The fix takes letters from one and word breaks from "
     "the other. <b>260-261</b> are the guard - a page that extracts cleanly is returned "
     "untouched, so the repair can never damage a well-formed document."),
    (262, 272,
     "<b>263-265</b> re-extract in layout mode, giving up quietly if that fails. <b>266</b> "
     "reverses each line back into reading order. <b>267-268</b> are the second guard: unless "
     "the candidate's average word length is clearly shorter - meaning words really were glued "
     "together and are now separated - the repair is abandoned. <b>271</b> prefers the exact "
     "merge, falling back to the reversed layout text, which is still far better than glued "
     "words when the two extractions disagree beyond ligature swaps."),
    (273, 280,
     "Constants for Word documents. <b>274-275</b> are the XML namespaces; Word tags every "
     "element with them, so every tag comparison below has to include the prefix. "
     "<b>278-279</b> define what counts as a heading: short, and not ending in sentence "
     "punctuation - including the Arabic question mark, comma and semicolon."),
    (281, 292,
     "Reading one Word paragraph. Word splits a sentence into runs whenever formatting changes, "
     "so the text of a paragraph is scattered across many nodes. <b>284</b> walks all of them in "
     "document order and <b>286-291</b> keep the three that carry content: text, tabs, and line "
     "breaks. Everything else - styling, revision marks, bookmarks - is skipped by omission."),
    (293, 320,
     "Walking the document body. <b>301-306</b> yield paragraphs. <b>307-317</b> treat a table "
     "as one unit, joining cells with \" | \" and rows with newlines, so a table is not read as "
     "a pile of unrelated fragments. <b>318-319</b> recurse into anything else, which is what "
     "makes content controls and similar wrappers transparent, while skipping "
     "<font name='Mono' size='8'>sectPr</font> - section properties, which hold page setup, not "
     "text."),
    (321, 339,
     "The Word loader. The docstring records the useful asymmetry: Word stores text in reading "
     "order, so none of the PDF repairs above are needed. <b>329-330</b> import inside the "
     "function because a .docx is just a zip of XML - no third-party dependency at all. "
     "<b>334-338</b> turn the \"bad zip\" error into the actual explanation, which is almost "
     "always an old binary .doc renamed."),
    (340, 359,
     "<b>341-342</b> parse the body. <b>345-353</b> append footnotes and endnotes, skipping the "
     "separator entries Word stores alongside them; in legal documents these notes often carry "
     "the amendment history, so dropping them would lose exactly the sentence a user asks about. "
     "<b>356-358</b> read the embedded title if the file has one."),
    (360, 373,
     "Rebuilding the paragraph structure as text. <b>364-368</b> classify each block, and "
     "<b>369</b> encodes the result: an ordinary paragraph is followed by a blank line - the "
     "chunker's strongest boundary - but a heading gets only a newline, which glues it to the "
     "text it introduces so they cannot be split into different chunks."),
    (374, 381,
     "The finished Word document, with paragraph and note counts in its metadata."),
    (382, 401,
     "Tabular files. The docstring states the choice: one record per block, rendered as "
     "\"column: value\" pairs. Preserving the visual table layout would look right and retrieve "
     "badly, because a row's values would be far from their column names. <b>389</b> picks the "
     "delimiter from the extension. <b>394-396</b> cap the rows read. <b>397</b> drops empty "
     "cells, so a sparse row stays short. <b>399</b> labels each block with its row number."),
    (402, 413,
     "The returned document, with <b>410</b> recording whether the row cap was hit - so a "
     "truncated ingest is visible rather than silent."),
    (414, 441,
     "JSON and JSON Lines. <b>417-428</b> handle the line-delimited case, one record per block "
     "with its number, falling back to the raw line when a record does not parse - a corrupt "
     "line is still worth indexing as text. <b>430-433</b> handle ordinary JSON by re-dumping it "
     "indented, which gives the chunker line structure to split on. <b>424</b> and <b>431</b> "
     "both pass <font name='Mono' size='8'>ensure_ascii=False</font>, without which every Arabic "
     "character would be stored as a backslash escape and nothing would ever match it."),
    (442, 450,
     "A deliberately small HTML reader: strip tags, drop scripts and styles, find the title, "
     "decode the handful of entities that actually appear in prose. <b>448</b> uses "
     "<font name='Mono' size='8'>chr(34)</font> and <font name='Mono' size='8'>chr(39)</font> "
     "rather than quote characters, which keeps the dictionary readable."),
    (451, 467,
     "<b>455</b> removes script and style elements <i>with their contents</i>, which must happen "
     "before tag stripping or the JavaScript inside them would become body text. <b>456</b> "
     "turns block-level closing tags into newlines, preserving paragraph structure. <b>457</b> "
     "removes what is left of the markup."),
    (468, 479,
     "The dispatch table: extension to loader. Anything not listed falls back to "
     "<font name='Mono' size='8'>load_text</font>, so an unknown but readable file is still "
     "indexed rather than refused."),
    (480, 495,
     "Choosing the title a citation will display, in descending order of trustworthiness. "
     "<b>489-491</b> prefer the instrument's own naming clause - the sentence inside the law "
     "that says what the law is called - which is the most authoritative name available. "
     "<b>492</b> accepts embedded metadata only when it differs from the filename and runs to at "
     "least three words, because export tools routinely stamp every PDF they produce with their "
     "own one-word product name. <b>494</b> falls back to the filename, which the user controls."),
    (496, 501,
     "The cache version. Every change to extraction output must bump this number, because the "
     "cache key is the file's bytes - which do not change when the code does. Without the "
     "version in the key, improving the Arabic repair would leave every previously ingested "
     "document stuck with its old, broken text."),
    (502, 528,
     "Extraction with a content-addressed cache. The docstring gives the justification: "
     "layout-mode PDF extraction dominates ingest time and depends only on the file's bytes, so "
     "re-indexing after a chunking change should not repeat it. <b>513-514</b> key the cache on "
     "a hash of the content plus the extractor version. <b>518</b> restores the page spans as "
     "tuples, since JSON has no tuple type. <b>520-524</b> re-derive identity from where the "
     "file is <i>now</i>: the same bytes may have been cached under a different name, and the "
     "id, source and filename-derived title must follow the current path. <b>526-527</b> treat "
     "any unreadable cache entry as a miss."),
    (529, 538,
     "The cache write. <b>532-534</b> write to a temporary file and rename it into place, so a "
     "process killed mid-write leaves no half-written entry that would later parse as garbage. "
     "<b>535-536</b> swallow filesystem errors entirely: caching is an optimisation, and a "
     "read-only disk should slow an ingest down, not fail it."),
    (539, 552,
     "The public entry point. <b>543-544</b> apply the title rules only to PDF and Word files, "
     "the formats where the filename is often the worst available name. <b>546-547</b> record "
     "size and modification time. <b>548-550</b> hash the extracted <i>text</i>, not the file - "
     "that hash is what later lets the store notice two different files whose content is "
     "identical, and skip the duplicate."),
    (553, 578,
     "Walking paths into indexable files. <b>558-560</b> accept an explicitly named file without "
     "question - if a user names it, index it. <b>563</b> chooses recursive or flat traversal. "
     "<b>567</b> rejects anything inside a skipped directory, <b>569</b> anything whose extension "
     "is unsupported, and <b>571</b> Word and Excel lock files, which are real files with real "
     "extensions and no content. <b>573-577</b> check the size limit inside a try, because a "
     "broken symlink or a file deleted during the walk raises here and should simply be skipped."),
]
