"""Line-by-line notes for rag/__init__.py."""

SUMMARY = (
    "The package's front door: what you get when you write 'import rag', and a three-line "
    "example of the whole system."
)

BLOCKS = [
    (1, 10,
     "The docstring doubles as the quickest possible tutorial: construct a pipeline, point it at "
     "a folder, ask a question. Three lines of real, working code is a more honest description "
     "of an API than a paragraph about it - and because it is in the module docstring, "
     "<font name='Mono' size='8'>help(rag)</font> shows it."),
    (11, 19,
     "The re-exports, which define the supported surface. Importing them here means a caller "
     "writes <font name='Mono' size='8'>from rag import RAGPipeline</font> rather than naming "
     "the module it happens to live in today - so the internal layout can change without "
     "breaking anyone. The imports are ordered by module name, not by importance, which keeps "
     "the list mechanical to maintain."),
    (20, 21,
     "The version. One string, read by the HTTP API's health endpoint and by the CLI's "
     "<font name='Mono' size='8'>--version</font> flag, so both report the same thing."),
    (22, 38,
     "The explicit public list. It controls what a star-import brings in, but its real value is "
     "as documentation: fourteen names out of roughly a hundred defined across the package, "
     "which is the difference between what is supported and what merely exists. The pipeline "
     "comes first because it is the one thing most callers need."),
]
