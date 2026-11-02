"""Test-wide settings, applied before any ``rag`` module is imported.

Unit tests exercise plumbing with a fake embedder; they must never download or
load real model weights, which reranking-on-by-default would otherwise do on
every search. Extraction caching is off so tests never share state on disk.
"""

import os

os.environ.setdefault("RAG_RERANK", "0")
os.environ.setdefault("RAG_EXTRACT_CACHE", "off")
