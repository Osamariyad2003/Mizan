"""Line-by-line notes for rag/store.py."""

SUMMARY = (
    "The index itself: what is on disk, how it is loaded and saved, and how documents are "
    "added without creating duplicates or leaving the vectors out of step with the chunks."
)

BLOCKS = [
    (1, 19,
     "The docstring is three decisions. <b>3-7</b> is the on-disk layout, and the alignment rule "
     "that everything depends on: row <i>i</i> of the matrix is line <i>i</i> of the chunk file. "
     "<b>9-14</b> defend being file-based - a vector database earns its operational cost "
     "somewhere north of a million chunks, and below that an exact dot product is faster than an "
     "approximate index, as well as exactly accurate and trivial to back up. <b>16-18</b> record "
     "a bug and its fix: memory-mapping the matrix saved nothing, because every query multiplies "
     "all of it, and left a file handle open that Windows then refused to let the next save "
     "replace."),
    (20, 42,
     "Imports and the three filenames, named once so no string literal has to be kept in sync "
     "across the load and save paths."),
    (43, 54,
     "The ingest report: what happened, in numbers the CLI can print. <b>51-53</b> are typed as "
     "lists but default to <font name='Mono' size='8'>None</font>, because a dataclass cannot "
     "take a mutable default - every report would share one list."),
    (55, 62, "Which is why the lists are created here, once per report."),
    (63, 79,
     "The one-line summary. <b>70-77</b> add each clause only when it is non-zero, so an "
     "ordinary ingest prints a short line and an interesting one prints what made it "
     "interesting - rather than four zeros the reader has to scan past every time."),
    (80, 93,
     "The store's state. <b>88</b> is the per-source bookkeeping that makes incremental ingest "
     "possible: content hash, modification time and chunk count, keyed by file path. <b>91</b> "
     "and <b>92</b> are caches - the keyword index, built on demand, and a flag recording "
     "whether the vectors have actually changed."),
    (94, 99, "A section marker, and the size of the index in chunks."),
    (100, 108,
     "The keyword index, built the first time it is asked for. The comment gives the reasoning "
     "for not persisting it: building it costs well under a second for tens of thousands of "
     "chunks, and a rebuilt index can never be stale against the chunks it was built from - "
     "whereas a persisted one could silently disagree with them. Note <b>106</b> indexes "
     "<font name='Mono' size='8'>indexed_text</font>, so the law title and article number are "
     "keyword-searchable too."),
    (109, 111,
     "Dropping the cache, called after every mutation. One line, called from four places, which "
     "is how the cache cannot be forgotten."),
    (112, 123,
     "Loading. <b>116-117</b> return an empty store when there is no index yet, so every caller "
     "can load unconditionally and ask about the size afterwards. <b>119-122</b> restore the "
     "metadata, including the fingerprint that records which settings built this index."),
    (124, 128,
     "Chunks are stored one JSON object per line, so they stream in and a single corrupt line "
     "could be identified rather than destroying the file's structure."),
    (129, 139,
     "Loading the vectors, with a consistency check that matters. <b>133-136</b> handle a crash "
     "between the two writes: if the matrix has a different number of rows than there are "
     "chunks, every row is potentially mapped to the wrong chunk, so the vectors are dropped "
     "entirely. Keyword search keeps working and the next ingest rebuilds them - a degraded "
     "index that is correct, instead of a complete one that is silently wrong."),
    (140, 147,
     "Saving, atomically. The chunks are written to a temporary file and renamed into place, so "
     "an interrupted save leaves the previous index intact rather than a half-written one."),
    (148, 160,
     "The vectors. <b>152</b> is the optimisation that makes re-running ingest on an unchanged "
     "corpus cheap - without it, every run would rewrite hundreds of megabytes to produce an "
     "identical file. <b>153-154</b> carry a real trap in a comment: "
     "<font name='Mono' size='8'>np.save</font> appends \".npy\" to whatever name it is given, "
     "so a temporary file named without that suffix is written somewhere other than where the "
     "rename then looks. <b>157-158</b> delete the file when the vectors have been dropped, so a "
     "stale matrix cannot be loaded next time and matched against the wrong chunks."),
    (161, 171,
     "The metadata file, written last and also atomically. It records who embedded the index and "
     "with what settings, which is what <font name='Mono' size='8'>check_fingerprint</font> "
     "compares against later."),
    (172, 180,
     "Deleting the index, in memory and on disk. <b>173</b> releases the matrix <i>before</i> "
     "<b>175</b> removes the directory, which on Windows is the difference between a clean delete "
     "and a permission error."),
    (181, 195,
     "Removing one source. <b>185-188</b> rebuild the chunk list without it. <b>189-190</b> is "
     "the line that keeps the index coherent: the same index list is applied to the matrix, so "
     "rows and chunks stay aligned. Deleting from one and not the other would leave every chunk "
     "after the deletion point holding its neighbour's vector."),
    (196, 201,
     "Whether a document needs re-indexing: unknown source, or changed content hash. Hashing the "
     "<i>extracted text</i> rather than the file means a PDF re-saved with new metadata but "
     "identical content is correctly recognised as unchanged."),
    (202, 212, "The main ingest entry point, with its report and the list of chunks to embed."),
    (213, 221,
     "Duplicate detection, and the comment explains what it is worth: the same document "
     "downloaded twice under two names would return every passage twice, spending half the "
     "context budget on repeats and presenting two citations that are really one source."),
    (222, 230,
     "<b>224-229</b> skip a document whose content hash matches a different file already "
     "indexed, recording both names so the CLI can say which was kept."),
    (231, 236, "Skipping unchanged documents, unless the user forced a rebuild."),
    (237, 243,
     "Update versus insert. <b>239</b> removes the old chunks first, which is what makes "
     "re-ingesting an edited file replace its content instead of adding a second copy "
     "alongside it."),
    (244, 255,
     "Chunking, and recording the per-source state that the next run's skip check will read."),
    (256, 268,
     "Reporting Arabic repairs. <b>256-263</b> record what was repaired both in the index "
     "metadata and in a human-readable list. <b>264-267</b> print it, so a user ingesting a "
     "corpus sees that pages were repaired at the moment it happens - rather than discovering "
     "silently altered text later. <b>266</b> adds the resolved title when it differs from the "
     "filename, which is how the naming-clause extraction becomes visible."),
    (269, 276,
     "Embedding the new chunks. <b>274</b> encodes <font name='Mono' size='8'>indexed_text</font> "
     "- header included - so what is embedded is exactly what is keyword-indexed. <b>276</b> "
     "stamps the current settings onto the index."),
    (277, 292,
     "The half-dense case, prevented deliberately. Adding chunks with no embedder would leave a "
     "matrix covering only part of the corpus, which hides the new chunks from vector search and "
     "misaligns every row index on the next delete. Rather than carry that, <b>289-291</b> drop "
     "the vectors entirely and <b>283-288</b> say so in a warning that names the remedy. A "
     "smaller, correct index beats a larger one that is quietly broken."),
    (293, 295, "The new chunks join the list, and the count goes into the report."),
    (296, 311,
     "Appending the new vectors. <b>301-305</b> refuse to stack matrices of different widths, "
     "which is what happens when the embedding model is changed without re-indexing - and the "
     "error names the command that fixes it. Without this check numpy would raise something far "
     "less helpful, or worse, broadcast."),
    (312, 327,
     "Dense search. The whole thing is <b>319</b>: one matrix-vector product, which is cosine "
     "similarity because every vector is L2-normalised. <b>324-325</b> use "
     "<font name='Mono' size='8'>argpartition</font> to find the top k in linear time and sort "
     "only those k, rather than sorting the entire corpus to look at twenty rows."),
    (328, 330, "Keyword search, delegated to the BM25 index."),
    (331, 348,
     "Index statistics, which is what <font name='Mono' size='8'>rag stats</font> prints. "
     "<b>343</b> reports whether dense search is available at all, which is the first thing to "
     "check when results look purely lexical."),
    (349, 356,
     "The settings-drift check. The docstring explains the "
     "<font name='Mono' size='8'>keys</font> parameter: at query time only the embedding model "
     "can invalidate anything, because chunk size was fixed at ingest and cannot go stale for a "
     "search - so a query checks one key and an ingest checks them all."),
    (357, 367,
     "<b>359-363</b> collect every setting whose current value differs from the indexed one. "
     "<b>366-367</b> report them with both values and the command to fix it, because \"index "
     "settings changed\" without saying which, from what, to what, is not an actionable message."),
]
