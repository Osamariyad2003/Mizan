"""Line-by-line notes for rag/legal.py."""

SUMMARY = (
    "Finds where each article of a law begins and ends. This is the module that lets a chunk "
    "be 'Labour Law, Article 61' instead of 'Labour Law, characters 41,000-41,500', which "
    "changes both what retrieval matches and what a citation can say."
)

BLOCKS = [
    (1, 22,
     "The docstring argues the design before any code appears. People ask about laws by article "
     "number, so the article - not the page, not a fixed-size window - is the natural unit. "
     "<b>8-13</b> list the three heading styles actually found in this corpus, which is why "
     "there are three patterns below rather than one. <b>15-19</b> state the hard problem: a "
     "line that merely <i>mentions</i> an article must not start one, and the defence is "
     "numbering - real articles run in increasing order, and a passing reference almost never "
     "happens to continue the sequence."),
    (23, 33,
     "The Arabic fragments the patterns are built from, each named and commented: the word "
     "\"article\", the suffix for an inserted article (the equivalent of \"61 bis\"), the phrase "
     "\"effective date\", and the preposition \"of\" that <b>102</b> uses to reject references."),
    (34, 45,
     "The three heading patterns, assembled from those fragments. <b>35-37</b> is the "
     "legal-database export: a leading dash, the number, an optional bis, a topic, and an "
     "optional effective date - four capture groups, so a heading yields metadata and not just "
     "a boundary. <b>39-41</b> is a heading alone on its line, with the brackets, dashes and "
     "colons different publishers use. <b>43-45</b> is a heading whose first sentence runs on "
     "after the dash, which is why it captures the remaining text as group 3."),
    (46, 52,
     "The division pattern: the six words Arabic legislation uses for book, part, chapter, "
     "branch, section and division. A division line is not an article, but it tells you where "
     "an article sits, which is what makes a citation read \"Chapter 3: Termination, Article 61\"."),
    (53, 56,
     "Two tuned constants. <b>53</b> allows the numbering to skip ahead by up to 50, because "
     "repealed and renumbered articles leave real gaps in real statutes. <b>54</b> is the floor "
     "below which the text is not treated as legislation at all - a contract that happens to "
     "contain one article-shaped line should fall back to ordinary chunking."),
    (57, 67,
     "The <font name='Mono' size='8'>Article</font> record. The three offsets are the useful "
     "part: <b>61</b> is where the heading starts, <b>62</b> is where the body starts, and "
     "<b>63</b> is where the next article begins. Keeping heading and body boundaries separate "
     "is what lets a chunk be labelled with its heading while being <i>searched</i> on its body. "
     "<b>64-66</b> carry the metadata a database heading supplies, each defaulting to empty "
     "because the other two heading styles do not provide it."),
    (68, 72,
     "The citation string, built from the document's own label, so \"42 bis\" cites as the "
     "document writes it rather than as a number the code invented."),
    (73, 80,
     "The guard that separates real headings from references, in six lines. <b>74</b> accepts the "
     "first article seen, and accepts a restart at 1 - which is what an annexed regulation "
     "printed after the main law looks like. <b>76-77</b> accept a bis article repeating the "
     "previous number, because that is exactly what \"61 bis\" is. <b>78</b> is the main rule: "
     "strictly forward, by no more than the allowed jump. A sentence citing Article 70 inside "
     "Article 12 fails this test, which is the whole point."),
    (81, 90,
     "<b>84-89</b> build a parallel list of character offsets, one per line, so the loop can "
     "report positions in the original string while iterating over lines. <b>89</b> adds one for "
     "the newline that <font name='Mono' size='8'>split</font> removed - the detail that keeps "
     "every offset in this module exact rather than approximately right."),
    (91, 104,
     "The scan begins. <b>94</b> is a while loop rather than a for loop because a database "
     "heading consumes its following division line and must be able to skip it. <b>97-100</b> "
     "try the three patterns in order of specificity, each only if the previous did not match. "
     "<b>102-103</b> are the reference guard at the sentence level: if an inline heading's text "
     "begins with \"of\", the line reads \"Article 70 of this law...\" and is a citation, not a "
     "heading."),
    (105, 121,
     "Accepting a heading. <b>109</b> finds the heading's true start by adding back the "
     "indentation the strip removed. <b>110</b> assumes the body begins on the next line. "
     "<b>111-112</b> pull topic and effective date, available only from a database heading. "
     "<b>113-115</b> correct the body start for an inline heading, where the body begins on the "
     "same line just after the dash. <b>117-120</b> handle the database layout where a part name "
     "follows the heading on its own line: it is recorded as the division, the body start moves "
     "past it, and the loop index skips it. <b>121</b> builds the label in the document's own "
     "wording."),
    (122, 132,
     "The article is recorded. <b>125</b> clamps the body start to the length of the text, "
     "guarding the case where a heading is the very last line and the body would otherwise begin "
     "past the end; <font name='Mono' size='8'>end</font> is set to the end of the document and "
     "corrected later. <b>129</b> advances the numbering guard, which is what makes the sequence "
     "test work at all."),
    (133, 141,
     "Division lines that are not attached to an article. <b>133</b> bounds the line length, "
     "since a long line beginning with the word \"chapter\" is prose. <b>135-137</b> look ahead "
     "one line for the division's title and combine them, but only if that next line is not "
     "itself an article heading - otherwise the heading would be swallowed into the division "
     "name and lost."),
    (142, 146,
     "Two finishing steps. <b>142-143</b> set each article's end to the next one's start, which "
     "is how the last article keeps its end at the end of the document. <b>144-145</b> apply the "
     "minimum: if too few articles were found, return nothing at all, so the caller falls back "
     "to ordinary window chunking rather than carving the document at a handful of accidental "
     "matches."),
]
