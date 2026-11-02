"""Line-by-line notes for rag/arabic.py."""

SUMMARY = (
    "Every Arabic-specific rule in the system, isolated so it can be tested on strings. "
    "Two separate jobs: folding spelling variants so search matches, and undoing the damage "
    "PDFs do to Arabic text."
)

BLOCKS = [
    (1, 23,
     "The module docstring is the design rationale for the whole file. Both problems it names "
     "fail <i>silently</i> - search simply returns nothing useful, with no error anywhere - "
     "which is why they are stated explicitly rather than left to be rediscovered. The second "
     "sentence of problem 1 is the key insight: normalisation is applied identically to "
     "documents and to queries, so it only has to be <i>consistent</i>, never linguistically "
     "correct. <b>21-22</b> explain why every Arabic character in this file is written as a "
     "numeric escape: several are combining marks or invisible, and a literal copy of them in "
     "source code cannot be reviewed by eye."),
    (24, 33,
     "<b>31-33</b> are the character-range tests. The first spans Arabic, Arabic Supplement and "
     "both presentation-form blocks, so it detects Arabic even in raw PDF glyph codes. The third "
     "matches only the presentation forms, which is how the code tells \"text a person typed\" "
     "from \"glyph codes a PDF stored\"."),
    (34, 42,
     "The diacritics pattern, with the comment that explains why removing them is not cosmetic: "
     "they are combining marks, and a word-character tokenizer treats a combining mark as a word "
     "break - so a vowelled word silently tokenises into three fragments that match nothing. "
     "<b>42</b> is the tatweel, a pure stretching character with no meaning at all."),
    (43, 56,
     "The folding table. <b>47-50</b> collapse the four written forms of alef into one, which is "
     "the single most valuable fold in Arabic search because the hamza is routinely omitted. "
     "<b>51-52</b> fold alef maqsura to ya and ta marbuta to ha, the two other endings people "
     "spell either way. <b>53-54</b> map both sets of Arabic-Indic digits to ASCII, so \"article "
     "61\" matches whichever digits the document used."),
    (57, 76,
     "The prefix list, and the most interesting comment in the file. Arabic fuses the definite "
     "article with prepositions and conjunctions, so one noun appears as a dozen tokens. The "
     "list is longest-first because matching is first-wins. <b>62-65</b> record a deliberate "
     "omission: a bare leading waw is <i>not</i> stripped, because it is as often a root letter "
     "- in ministry, document, minister - as it is the word \"and\", and stripping it would "
     "break exactly the words a legal corpus is full of."),
    (77, 79,
     "The minimum stem length. Without it, stripping a two-letter prefix from a four-letter word "
     "could leave a fragment that matches everything; the comment names the real case it "
     "protects, the two-letter legal term for \"right\"."),
    (80, 83,
     "A cheap test used as a guard throughout: if there is no Arabic, every function below "
     "returns its input untouched, which is what makes the module safe on English or mixed text."),
    (84, 96,
     "The normaliser, applied to documents and queries alike. <b>89-90</b> return non-Arabic text "
     "unchanged. <b>91-93</b> run NFKC only when presentation forms are present, converting PDF "
     "glyph codes into the letters a person would actually type. <b>94</b> removes diacritics and "
     "tatweel - note this happens <i>before</i> any tokenizing elsewhere in the system, which is "
     "the fix for the fragmenting described above. <b>95</b> applies the folding table."),
    (97, 104,
     "The stemmer, applied per token after folding. <b>101</b> carries both conditions: the token "
     "starts with the prefix <i>and</i> enough letters would remain. Returning on the first match "
     "is what makes the longest-first ordering of the prefix list significant."),
    (105, 111,
     "Patterns for runs that keep left-to-right order even inside a right-to-left line: numbers "
     "in any digit set and Latin words, including internal separators so a date or a decimal "
     "stays one unit."),
    (112, 125,
     "Reversing a visual-order line. The docstring states the complication honestly: reversing "
     "the whole line fixes the Arabic but also reverses the numbers and Latin words, which were "
     "already correct. <b>122</b> reverses everything; <b>123</b> then flips each left-to-right "
     "run back, so \"12\" does not become \"21\". <b>120-121</b> leave a line with no Arabic "
     "completely alone."),
    (126, 129, "A section banner: below here, the repairs for PDF-specific damage."),
    (130, 136,
     "A single mis-mapped glyph, documented because it looks arbitrary. An initial-form letter "
     "cannot appear at the end of a word, so when it does the font has mis-mapped something - "
     "here, a common Arabic adverbial ending. <b>132-133</b> state the ordering constraint: this "
     "must be fixed <i>before</i> folding, because folding erases the positional information "
     "that makes the error detectable at all."),
    (137, 148,
     "Folding glyph codes to letters. <b>139-141</b> explain why the two presentation-form blocks "
     "are targeted individually instead of running NFKC over everything: NFKC would also rewrite "
     "unrelated characters such as fractions and superscripts, changing text that was never "
     "broken. <b>143-144</b> skip the work entirely when there is nothing to fold."),
    (149, 151,
     "The four characters that end a sentence in Arabic, including the Arabic comma and "
     "semicolon, which are distinct code points from their Latin lookalikes."),
    (152, 170,
     "The word-order detector, and a neat piece of reasoning. Normal text ends lines with "
     "punctuation; word-reversed text starts them with it. <b>161-167</b> count both and "
     "<b>168</b> requires a clear majority - at least five such lines, and more than twice as "
     "many as the normal pattern - so a document is only declared reversed on real evidence. "
     "<b>155-159</b> explain why whole lines are counted rather than tokens: reversed clause "
     "numbering puts a period at a token's end, which would masquerade as normal text."),
    (171, 184,
     "<b>173-178</b> define what survives reversal as one unit: a number or Latin run with its "
     "separators, a run of Arabic letters, whitespace, or any single character - the last "
     "alternative guaranteeing the pattern consumes everything. <b>179</b> mirrors brackets, "
     "which point the wrong way after reversal. <b>182</b> catches punctuation stranded before "
     "the next word instead of after the previous one; no Arabic typography puts it there, so "
     "its presence is unambiguous."),
    (185, 198,
     "Reversing word order while keeping each word's letters. <b>194</b> splits into units, "
     "<b>195</b> reverses the sequence and mirrors brackets in one pass, and <b>196</b> "
     "reattaches the stranded punctuation."),
    (199, 204,
     "A fused-duplicate word, caused by PDFs that draw bold text twice. <b>202</b> requires at "
     "least three repeated letters and uses look-around to require word boundaries. The comment "
     "explains the three-letter floor precisely: genuine Arabic reduplication repeats only two "
     "letters, so real words are never damaged by this rule."),
    (205, 209, "Applying that pattern."),
    (210, 218,
     "The naming clause. Jordanian legislation names itself in its own Article 1 - \"this law "
     "shall be called ( ... )\" - and <b>212-217</b> match that sentence for all three "
     "instrument types, capturing the title in the brackets. The 4-to-90 character bound keeps a "
     "stray pair of brackets from being mistaken for a title."),
    (219, 229,
     "Extracting that title. <b>226</b> searches only the first 20,000 characters, since a "
     "naming clause appears at the top or not at all. The docstring states why this outranks PDF "
     "metadata: export tools stamp their own name into every file they produce, so an entire "
     "legal database can arrive with one identical title."),
    (230, 232,
     "The two letters of the lam-alef ligature, as a set so order does not matter in the "
     "comparison below."),
    (233, 245,
     "The merge that repairs glued Arabic. The docstring states exactly why each extraction mode "
     "is half right: plain mode has every letter in reading order but no spaces, and layout "
     "mode has the spaces but, once reversed, comes out with every lam-alef ligature swapped - "
     "because that ligature is a single glyph decoding to two letters already in reading order, "
     "which the reversal then flips."),
    (246, 252,
     "The safety check that makes this function trustworthy. <b>247-248</b> require the same "
     "letter count and <b>249-251</b> require every letter to match, with the lam-alef swap as "
     "the single permitted difference. Any other disagreement returns "
     "<font name='Mono' size='8'>None</font>, so the caller falls back rather than producing "
     "text where letters and spaces have drifted out of alignment - which would be worse than "
     "the original defect, because it would look correct."),
    (253, 259,
     "The merge itself: walk the spaced version, emit its whitespace as-is, and draw every "
     "non-space character from the plain version's letters. The result has plain mode's letters "
     "in layout mode's word positions."),
    (260, 264,
     "Average length of the Arabic words in a string - the measurement both the detector below "
     "and the loader's repair guard are built on."),
    (265, 273,
     "The glued-text detector. Arabic words run four to six letters, so an average well above "
     "that means words have run together rather than that the document uses unusually long ones. "
     "<b>271-272</b> refuse to judge a page with too few words, where the average would be noise."),
]
