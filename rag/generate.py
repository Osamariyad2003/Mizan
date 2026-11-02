"""Grounded answer generation with inline citations.

Design decisions worth knowing:

* Retrieved context goes in the **system** part, not the user turn. On Claude it
  carries a cache breakpoint, so follow-up turns over the same passages reuse it;
  Gemini caches repeated prefixes on its own. Nothing here is provider-specific:
  the request is handed to a backend in ``llm.py``, which is what makes the same
  prompt, citations and refusal handling work on Claude or Gemini.
* The model is told to answer *only* from the context and to say so when the
  context does not contain the answer. An honest "not in these documents" is the
  single most valuable behaviour in a RAG system -- a fluent guess is worse than
  no answer because it is indistinguishable from a real one.
* Citations are ``[n]`` markers keyed to numbered sources. They are parsed back
  out afterwards so the caller gets a structured list of what was actually used,
  not just prose that happens to contain brackets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .config import Config, DEFAULT_CONFIG
from .llm import Backend, get_backend
from .retriever import RetrievedChunk

SYSTEM_PROMPT = """You answer questions using only the excerpts provided in <context>.

Rules:
- Ground every factual claim in the excerpts. Do not use outside knowledge, and do not fill gaps with plausible-sounding detail.
- Cite with bracketed numbers matching the excerpt ids, e.g. [2] or [1][4]. Place a citation immediately after the claim it supports.
- When an excerpt's location names a law and an article, also name them in the answer (e.g. "under Article 61 of the Labour Law [2]"), since that is how legal provisions are referred to. Never cite an article number that does not appear in an excerpt's location or text.
- If the excerpts do not contain the answer, say so plainly and state what is missing. Do not guess.
- If the excerpts disagree, surface the disagreement rather than silently picking one.
- Answer in the language the question was asked in.
- Be direct. No preamble, no restating the question."""

CITATION_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    model: str | None = None

    @property
    def cited_sources(self) -> list[RetrievedChunk]:
        return [self.sources[i - 1] for i in self.cited_indices if 1 <= i <= len(self.sources)]

    @property
    def grounded(self) -> bool:
        """Whether the answer actually pointed at its evidence."""
        return bool(self.cited_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "cited": [
                {"n": i, "location": self.sources[i - 1].location,
                 "source": self.sources[i - 1].chunk.source}
                for i in self.cited_indices
                if 1 <= i <= len(self.sources)
            ],
            "sources": [s.to_dict() for s in self.sources],
            "usage": self.usage,
            "model": self.model,
            "stop_reason": self.stop_reason,
        }


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, attributed excerpts."""
    blocks = []
    for i, hit in enumerate(chunks, start=1):
        blocks.append(
            f'<excerpt id="{i}" source="{hit.chunk.source}" location="{hit.location}">\n'
            f"{hit.text}\n"
            f"</excerpt>"
        )
    return "<context>\n" + "\n\n".join(blocks) + "\n</context>"


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Extract cited excerpt numbers, in order of first appearance."""
    seen: list[int] = []
    for match in CITATION_RE.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def _build_request(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """System parts and messages, in the provider-neutral shape."""
    system_parts = [SYSTEM_PROMPT, format_context(chunks)]
    messages = list(history or []) + [{"role": "user", "content": question}]
    return system_parts, messages


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    config: Config = DEFAULT_CONFIG,
    history: Sequence[dict[str, Any]] | None = None,
) -> Answer:
    """Answer the question from the retrieved context, on either provider."""
    if not chunks:
        return Answer(
            question=question,
            text="Nothing in the index matched that question, so there is no "
                 "grounded answer to give. Try rephrasing, or index more documents.",
            sources=[],
        )

    backend: Backend = get_backend(config)
    system_parts, messages = _build_request(question, chunks, history)
    completion = backend.complete(system_parts, messages)

    if completion.refused:
        return Answer(
            question=question,
            text=f"The model declined to answer this request ({completion.refusal_detail or 'unspecified'}).",
            sources=list(chunks),
            stop_reason="refusal",
            model=completion.model,
        )

    return Answer(
        question=question,
        text=completion.text,
        sources=list(chunks),
        cited_indices=parse_citations(completion.text, len(chunks)),
        usage=completion.usage,
        stop_reason=completion.stop_reason,
        model=completion.model,
    )


def stream_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    config: Config = DEFAULT_CONFIG,
    history: Sequence[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Yield answer text incrementally. Terminal UIs and SSE both use this."""
    if not chunks:
        yield "Nothing in the index matched that question."
        return

    backend: Backend = get_backend(config)
    system_parts, messages = _build_request(question, chunks, history)
    yield from backend.stream(system_parts, messages)
