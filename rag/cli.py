"""Command-line interface.

    python -m rag ingest ./docs
    python -m rag search "deployment rollback"
    python -m rag ask "How do I roll back a deploy?"
    python -m rag chat
    python -m rag stats
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .config import Config
from .pipeline import RAGPipeline
from .retriever import make_filter

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Hybrid-retrieval RAG over your local documents, answered by Gemini or Claude.",
)
console = Console()


def _pipeline(
    index_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    rerank: Optional[bool] = None,
    chunk_tokens: Optional[int] = None,
    provider: Optional[str] = None,
) -> RAGPipeline:
    cfg = Config()
    if index_dir:
        cfg.index_dir = Path(index_dir)
    if chunk_tokens:
        cfg.chunk_tokens = chunk_tokens
    if top_k:
        cfg.top_k = top_k
    if model:
        cfg.model = model
    if provider:
        cfg.llm_provider = provider
    if rerank is not None:
        cfg.rerank = rerank
    return RAGPipeline(cfg, adopt_index_settings=chunk_tokens is None)


def _warn_if_stale(rag: RAGPipeline, querying: bool = True) -> None:
    # A search only goes stale if the embedding model changed; chunk settings
    # are an ingest-time concern.
    warning = rag.store.check_fingerprint(("embed_model",) if querying else None)
    if warning:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="Files or directories to index."),
    index_dir: Optional[str] = typer.Option(None, "--index", help="Index location."),
    force: bool = typer.Option(False, "--force", help="Re-index even if unchanged."),
    no_recursive: bool = typer.Option(False, "--no-recursive", help="Do not descend into subdirectories."),
    chunk_tokens: Optional[int] = typer.Option(
        None, "--chunk-tokens",
        help="Chunk size. Remembered by the index; 1200 suits Arabic-only corpora.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Index documents. Unchanged files are skipped, so re-running is cheap."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    rag = _pipeline(index_dir, chunk_tokens=chunk_tokens)
    _warn_if_stale(rag, querying=False)

    def progress(source: str, status: str) -> None:
        if not quiet and status != "unchanged":
            console.print(f"  [dim]{escape(Path(source).name)}[/dim] -> {escape(status)}")

    with console.status("Indexing..."):
        report = rag.ingest(paths, recursive=not no_recursive, force=force, progress=progress)

    console.print(f"[green]OK[/green] {report.summary()}")
    for gone in report.pruned:
        console.print(f"  [yellow]removed[/yellow] {escape(Path(gone).name)} (file no longer exists)")
    for skipped, kept in report.duplicates:
        console.print(f"  [yellow]duplicate[/yellow] {escape(Path(skipped).name)} is identical to {escape(Path(kept).name)}")
    for source, error in report.failures:
        console.print(f"  [red]failed[/red] {escape(source)}: {escape(str(error))}")

    stats = rag.stats()
    mode = f"dense+bm25 ({stats['embedder']})" if stats["dense"] else "bm25 only"
    console.print(
        f"[dim]Index: {stats['chunks']} chunks from {stats['sources']} sources, {mode}, "
        f"{rag.config.chunk_tokens}-token chunks[/dim]"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    index_dir: Optional[str] = typer.Option(None, "--index"),
    source: Optional[str] = typer.Option(None, "--source", help="Only chunks whose path contains this."),
    fmt: Optional[str] = typer.Option(None, "--format", help="Only this document format, e.g. pdf."),
    rerank: Optional[bool] = typer.Option(None, "--rerank/--no-rerank", help="Reorder results with the reranker (default: on)."),
    full: bool = typer.Option(False, "--full", help="Print whole chunks."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve chunks without generating an answer. Needs no API key."""
    rag = _pipeline(index_dir, top_k=top_k)
    _warn_if_stale(rag)
    hits = rag.search(query, top_k=top_k, chunk_filter=make_filter(source, fmt), rerank=rerank)

    if as_json:
        # Plain stdout, not console.print_json: rich wraps long values to the
        # terminal width, which corrupts the JSON as soon as it is piped.
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False))
        return
    if not hits:
        console.print("[yellow]No matches.[/yellow] Is anything indexed? Try `python -m rag stats`.")
        raise typer.Exit(1)

    for hit in hits:
        body = hit.text if full else (hit.text[:400] + ("..." if len(hit.text) > 400 else ""))
        console.print(
            Panel(
                escape(body),
                title=f"[bold]{hit.rank + 1}. {escape(hit.location)}[/bold]",
                subtitle=f"[dim]{hit.why()}[/dim]",
                border_style="blue",
            )
        )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the indexed documents."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Max passages (default: fill the token budget)."),
    index_dir: Optional[str] = typer.Option(None, "--index"),
    model: Optional[str] = typer.Option(None, "--model", help="Model id, e.g. gemini-3.8-flash."),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini | claude | openrouter (default: whichever has a key)."),
    source: Optional[str] = typer.Option(None, "--source"),
    fmt: Optional[str] = typer.Option(None, "--format"),
    rerank: Optional[bool] = typer.Option(None, "--rerank/--no-rerank", help="Reorder results with the reranker (default: on)."),
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Answer a question using the indexed documents, with citations."""
    rag = _pipeline(index_dir, top_k=top_k, model=model, rerank=rerank, provider=provider)
    _warn_if_stale(rag)
    chunk_filter = make_filter(source, fmt)

    if as_json:
        try:
            answer = rag.ask(question, top_k=top_k, chunk_filter=chunk_filter)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        print(json.dumps(answer.to_dict(), ensure_ascii=False))
        return

    hits, stream = rag.ask_stream(question, top_k=top_k, chunk_filter=chunk_filter)
    if not hits:
        console.print("[yellow]Nothing indexed matches that question.[/yellow]")
        raise typer.Exit(1)

    try:
        parts: list[str] = []
        with console.status("Thinking..."):
            first = next(stream, "")
        parts.append(first)
        console.print(first, end="")
        for delta in stream:
            parts.append(delta)
            console.print(delta, end="")
        console.print()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if show_sources:
        from .generate import parse_citations

        cited = parse_citations("".join(parts), len(hits))
        table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2, 0, 0))
        table.add_column("#")
        table.add_column("Source")
        table.add_column("Location")
        for i, hit in enumerate(hits, start=1):
            marker = "[green]*[/green]" if i in cited else " "
            table.add_row(f"{marker}{i}", escape(Path(hit.chunk.source).name), escape(hit.location))
        console.print()
        console.print(table)
        console.print("[dim]* = cited in the answer[/dim]")


@app.command()
def chat(
    index_dir: Optional[str] = typer.Option(None, "--index"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Max passages (default: fill the token budget)."),
    model: Optional[str] = typer.Option(None, "--model", help="Model id, e.g. gemini-3.8-flash."),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini | claude | openrouter (default: whichever has a key)."),
) -> None:
    """Interactive multi-turn session. Each turn retrieves afresh."""
    rag = _pipeline(index_dir, top_k=top_k, model=model, provider=provider)
    _warn_if_stale(rag)
    stats = rag.stats()
    console.print(
        Panel(
            f"{stats['chunks']} chunks from {stats['sources']} sources. "
            "Ask a question, or /exit to quit.",
            title="RAG chat",
            border_style="green",
        )
    )

    history: list[dict] = []
    while True:
        try:
            question = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not question:
            continue
        if question in {"/exit", "/quit", "/q"}:
            break
        if question == "/clear":
            history.clear()
            console.print("[dim]History cleared.[/dim]")
            continue

        try:
            answer = rag.ask(question, top_k=top_k, history=history)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        console.print(Markdown(answer.text))
        if answer.cited_sources:
            locations = ", ".join(s.location for s in answer.cited_sources)
            console.print(f"[dim]Sources: {escape(locations)}[/dim]")
        console.print()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer.text})
        # Keep the window short: retrieval, not history, carries the context.
        history[:] = history[-8:]


@app.command()
def stats(index_dir: Optional[str] = typer.Option(None, "--index")) -> None:
    """Show what is in the index."""
    rag = _pipeline(index_dir)
    info = rag.stats()
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    for key in ("index_dir", "sources", "chunks", "tokens", "avg_chunk_tokens", "embedder", "dim"):
        table.add_row(key, str(info.get(key)))
    table.add_row("formats", ", ".join(f"{k}:{v}" for k, v in sorted(info["by_format"].items())) or "-")
    table.add_row("retrieval", "dense + bm25" if info["dense"] else "bm25 only")

    answers = (
        f"{info['provider']} ({info['answer_model']})"
        if info.get("provider")
        else "[yellow]no API key - retrieval only[/yellow]"
    )
    table.add_row("answers", answers)
    keys = ", ".join(info["keys_found"]) or "none"
    if info.get("env_file_keys"):
        keys += f"  (from {info['env_filename']}: {', '.join(info['env_file_keys'])})"
    table.add_row("keys found", keys)
    console.print(table)
    if not info["keys_found"]:
        console.print(
            f"[dim]To answer questions, put a key in {info['env_filename']} "
            "(GEMINI_API_KEY=...) -- free key: https://aistudio.google.com/apikey[/dim]"
        )
    if info.get("warning"):
        console.print(f"[yellow]![/yellow] {info['warning']}")
    if not info["chunks"]:
        console.print("[dim]Nothing indexed yet: python -m rag ingest <path>[/dim]")


@app.command()
def sources(index_dir: Optional[str] = typer.Option(None, "--index")) -> None:
    """List indexed files."""
    rag = _pipeline(index_dir)
    if not rag.store.sources:
        console.print("[dim]Nothing indexed.[/dim]")
        return
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2, 0, 0))
    table.add_column("Source")
    table.add_column("Format")
    table.add_column("Chunks", justify="right")
    for source, meta in sorted(rag.store.sources.items()):
        table.add_row(source, str(meta.get("format", "-")), str(meta.get("chunks", 0)))
    console.print(table)


@app.command()
def remove(
    source: str = typer.Argument(..., help="Exact source path to drop from the index."),
    index_dir: Optional[str] = typer.Option(None, "--index"),
) -> None:
    """Remove one source from the index."""
    rag = _pipeline(index_dir)
    removed = rag.remove(source)
    if removed:
        console.print(f"[green]OK[/green] removed {removed} chunks from {source}")
    else:
        console.print(f"[yellow]Not indexed:[/yellow] {source}")
        raise typer.Exit(1)


@app.command()
def reset(
    index_dir: Optional[str] = typer.Option(None, "--index"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete the whole index."""
    rag = _pipeline(index_dir)
    target = rag.store.dir
    if not target.exists():
        console.print("[dim]No index to delete.[/dim]")
        return
    if not yes and not typer.confirm(f"Delete the index at {target}?"):
        raise typer.Abort()
    rag.reset()
    console.print(f"[green]OK[/green] deleted {target}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    index_dir: Optional[str] = typer.Option(None, "--index"),
) -> None:
    """Run the HTTP API (needs: pip install uvicorn)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Install the server extras: pip install uvicorn[/red]")
        raise typer.Exit(1)
    if index_dir:
        import os

        os.environ["RAG_INDEX_DIR"] = index_dir
    console.print(f"Test page: [bold]http://{host}:{port}/[/bold]  (Ctrl+C to stop)")
    uvicorn.run("rag.server:app", host=host, port=port, log_level="warning")


def _force_utf8_output() -> None:
    """Emit UTF-8 even when stdout is a pipe or a file.

    An interactive Windows console already receives Unicode through the console
    API, but a pipe or redirect falls back to the ANSI code page (cp1256 on
    Arabic-locale Windows). Arabic then arrives as mojibake in anything reading
    the output -- `search --json | ...` included -- or fails outright for
    characters the code page lacks.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # replaced or detached streams
            pass


def main() -> None:
    _force_utf8_output()
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


if __name__ == "__main__":
    main()
