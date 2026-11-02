"""Line-by-line notes for rag/chunking.py."""

SUMMARY = (
    "Turns a document into the units that are actually searched. Two strategies live here: "
    "one article per chunk for legislation, and overlapping token windows for everything else. "
    "Every chunk keeps exact offsets back into its document."
)

BLOCKS = [
    (1, 11,
     "The docstring states the two ideas. First, text is split along the <i>strongest available "
     "boundary</i> - blank line, then line, then sentence, then word - so a chunk breaks where a "
     "human would break it. Second, every chunk keeps its character offsets into the parent "
     "document, which is what lets a citation name a page and what lets any answer be checked "
     "against the original file."),
    (12, 24,
     "Imports. <font name='Mono' size='8'>lru_cache</font> is here for the token encoder, which "
     "is expensive to construct and must be built once. The three local imports are this "
     "module's whole world: settings, the article finder, and the Document shape."),
    (25, 33,
     "<b>27-31</b> are the four boundary patterns, in descending strength. <b>30</b> treats the "
     "Arabic question mark and the Arabic full stop as sentence ends alongside the Latin ones - "
     "without them, an Arabic paragraph has no sentence boundaries at all and the splitter would "
     "fall straight through to splitting on spaces."),
    (34, 44,
     "The token encoder, cached so it is built once per process. <b>41-42</b> return "
     "<font name='Mono' size='8'>None</font> rather than raising, which makes tiktoken genuinely "
     "optional - the system still runs, just with approximate sizing."),
    (45, 56,
     "Counting tokens. <b>52-53</b> are the fallback: roughly four characters per token. The "
     "docstring is careful to scope the approximation - it decides chunk sizes only, and the "
     "exact token count of the final prompt is the provider's business. <b>54</b> passes "
     "<font name='Mono' size='8'>disallowed_special=()</font>, without which a document "
     "containing a literal token-marker string would raise instead of being counted."),
    (57, 69,
     "The <font name='Mono' size='8'>Chunk</font> record: what it says, where it came from, and "
     "how big it is. <b>59</b> is a stable id built from the document id and the ordinal, so the "
     "same document re-ingested produces the same chunk ids. <b>65-66</b> are the offsets into "
     "the parent document and <b>68</b> the pages they fall on."),
    (70, 74,
     "The header field, with the comment that justifies it: a passage lifted out of a 500-page "
     "law almost never names the law it belongs to, so a question naming the law would not match "
     "the passage answering it. The header puts the missing context back."),
    (75, 78,
     "What actually gets embedded and keyword-indexed: the header joined to the text. This one "
     "property is why the law's title and article number participate in matching rather than "
     "being decoration added at display time."),
    (79, 94,
     "Building the human-readable location a citation shows. <b>86</b> renders a multi-page chunk "
     "as a range rather than a list. <b>87-90</b> prefer the article form, which is how a legal "
     "source should be cited. <b>91-92</b> fall back to a page, and <b>93</b> to a chunk number, "
     "so every chunk can always say where it came from, whatever the format."),
    (95, 102,
     "Conversion to and from plain dictionaries - the index on disk is JSON, so every chunk has "
     "to survive a round trip through it."),
    (103, 115,
     "Splitting while tracking absolute positions. The subtlety is <b>110</b>: the cursor jumps "
     "to the <i>end</i> of the separator match, so the separator itself belongs to neither span. "
     "That is what keeps blank lines out of chunk text while leaving the offsets exact. "
     "<b>111-112</b> emit the final span, which has no separator after it."),
    (116, 127,
     "The recursive splitter. <b>118</b> is the ladder of boundaries. <b>124</b> stops as soon as "
     "a segment fits, which is the point: a paragraph that fits is never split into sentences, "
     "so chunks follow the document's own structure wherever they can."),
    (128, 140,
     "The escape hatch. When every boundary has been tried and a segment still does not fit - a "
     "single enormous paragraph with no spaces, a base64 blob, a table rendered as one line - "
     "<b>129-133</b> cut on a character estimate. It is crude by design: without it a chunk "
     "could exceed the model's limit, and a hard failure at generation time is worse than a "
     "slightly awkward split. <b>135-136</b> are the recursion itself, descending one level at a "
     "time."),
    (141, 155,
     "The entry point, and the fork in the road. <b>150</b> looks for articles only if article "
     "chunking is enabled. <b>151-152</b> take the legal path when articles were found - and "
     "recall that the finder returns nothing unless it is confident, so this is a genuine "
     "either/or rather than a guess. <b>153-154</b> are the ordinary path, with the document "
     "title as the header."),
    (156, 166,
     "Article-aligned chunking. The docstring explains the packing option and why it exists: many "
     "constitutional articles are a single sentence, and a chunk that small carries too little "
     "context for meaning-based matching. It also records the safeguard - each article keeps its "
     "own heading inside the text, so attribution survives even when articles are packed "
     "together. In this corpus packing measured worse and is off by default."),
    (167, 178,
     "<b>171-177</b> rescue the preamble: everything before Article 1 - the title page, the "
     "enacting formula, the publication details - is chunked the ordinary way rather than "
     "discarded. Without this, the text that names and dates the law would never be retrievable."),
    (179, 195,
     "Emitting one group of articles. <b>183</b> labels a group as a range. <b>184</b> collects "
     "up to three distinct topics, preserving order. <b>185-186</b> build the header: law title, "
     "article label, topics - which is the string that makes a chunk findable by a question "
     "naming any of the three. <b>187-194</b> record the structured metadata, each conditional "
     "entry included only when it has a value, so chunks do not carry empty keys."),
    (196, 214,
     "The ordinary case: the group fits, so it becomes one chunk. <b>201</b> builds the "
     "zero-padded id, which keeps chunk ids sorting in document order. <b>210</b> resolves which "
     "pages this character range touched - the link back from an article to a printed page."),
    (215, 223,
     "The exception: one article longer than a whole chunk. It is windowed <i>within its own "
     "bounds</i>, and every window carries the same header and metadata - so a long article "
     "becomes several chunks that all still cite the same article number."),
    (224, 238,
     "The packing loop. <b>227-228</b> skip articles whose body is empty. <b>230-232</b> flush "
     "the current group when adding the next article would exceed the packing budget. Since that "
     "budget is 0 by default, the condition is true on every iteration and each article becomes "
     "its own chunk - the measured best setting, reached without a special case in the code. "
     "<b>235-236</b> emit whatever is left."),
    (239, 257,
     "Window chunking. <b>251-254</b> atomise the span and shift every offset back into document "
     "coordinates, which is what allows this function to be called for a sub-range - a single "
     "long article - as easily as for a whole file."),
    (258, 262,
     "<b>258</b> costs each atom in tokens once, so the packing loop below never re-tokenises. "
     "<b>260-261</b> are the accumulator: a buffer of atoms and its running token count."),
    (263, 274,
     "Building the overlap. <b>267-271</b> walk backwards from the end of the chunk just emitted, "
     "taking whole atoms while they fit the overlap budget, and <b>270</b> inserts at the front "
     "so the tail stays in reading order."),
    (275, 294,
     "The partial-tail path, which exists because of a real bug. For dense prose a single "
     "paragraph is larger than the whole overlap budget, so no whole atom ever fits - and the "
     "overlap silently became zero, exactly where it was needed most. <b>282-284</b> estimate "
     "how many characters the overlap is worth and cut there. <b>285-286</b> refuse a cut that "
     "would duplicate the entire atom, which would make no forward progress. <b>288-290</b> "
     "advance to the next space so the overlap does not begin in the middle of a word. The "
     "guards at <b>280</b>, <b>285</b> and <b>291</b> all return an empty tail rather than "
     "risking a loop that never advances."),
    (295, 301,
     "Emitting a chunk. <b>299</b> takes its bounds from the first and last atoms in the buffer, "
     "so the offsets are the real text positions, not a running estimate."),
    (302, 306,
     "The small-fragment rule: a buffer below the minimum size is <i>kept</i> rather than "
     "emitted, so it joins the next chunk. The two extra conditions matter - not on the final "
     "flush, and not if this is the first chunk - because otherwise the tail of a document, or a "
     "short document entire, would be silently dropped."),
    (307, 329,
     "<b>308</b> continues the ordinal across calls, which is how a windowed article keeps "
     "numbering in sequence with the chunks around it. <b>321</b> copies the supplied metadata "
     "rather than sharing it, so the article metadata handed in is not mutated by every chunk "
     "that receives it. <b>325-328</b> are the difference between finishing and continuing: a "
     "final flush empties the buffer, an ordinary one re-opens it with the overlap tail."),
    (330, 335,
     "The main packing loop, in five lines: flush when the next atom would overflow, then append "
     "it. The order matters - flushing first means the overflowing atom starts the next chunk "
     "rather than being lost."),
    (336, 344,
     "The remainder. <b>340</b> is the check that prevents a duplicate final chunk: if the "
     "buffer holds nothing but the overlap tail, and the previous chunk already extends that "
     "far, there is nothing new to emit."),
    (345, 353, "Chunking many documents is just chunking each one, concatenated."),
    (354, 356,
     "A hash of a chunk's text, used by the store to recognise identical content arriving from "
     "two different files."),
]
