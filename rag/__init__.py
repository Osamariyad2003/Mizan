"""A hybrid-retrieval RAG system: local embeddings + BM25, answers from Gemini or Claude.

Quick start::

    from rag import RAGPipeline

    rag = RAGPipeline()
    rag.ingest(["./docs"])
    print(rag.ask("What changed in the v2 release?").text)
"""

from .config import Config, DEFAULT_CONFIG
from .chunking import Chunk, chunk_document, count_tokens
from .generate import Answer
from .loaders import Document, load_document, discover
from .pipeline import RAGPipeline
from .retriever import RetrievedChunk, Retriever, make_filter
from .store import VectorStore, IngestReport

__version__ = "1.0.0"

__all__ = [
    "RAGPipeline",
    "Config",
    "DEFAULT_CONFIG",
    "Answer",
    "Chunk",
    "Document",
    "RetrievedChunk",
    "Retriever",
    "VectorStore",
    "IngestReport",
    "chunk_document",
    "count_tokens",
    "load_document",
    "discover",
    "make_filter",
]
