"""Command-line interface: `notelight index | search | status | mcp | eval | version`."""

from __future__ import annotations

import codecs
import io
import json
import os
import platform
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import typer

import notelight
from notelight import paths
from notelight.embed import EMBEDDER_NAMES, Embedder, ModelFile, make_embedder, name_for_model_id
from notelight.errors import IndexNotFoundError, NotelightError
from notelight.indexer import SyncStats, sync
from notelight.search import (
    IndexInfo,
    Searcher,
    describe_index,
    list_indexes,
    match_spans,
    query_terms,
    snippet,
)

app = typer.Typer(
    name="notelight",
    help="Private semantic search over your own files, for you and your AI agents.",
    no_args_is_help=True,
    add_completion=False,
)

EmbedderOpt = Annotated[
    str | None,
    typer.Option(
        "--embedder",
        "-e",
        help=f"Embedding model: {', '.join(EMBEDDER_NAMES)}. 'none' is keyword-only (BM25).",
        show_default=False,
    ),
]
OfflineOpt = Annotated[
    bool,
    typer.Option("--offline", help="Never download; fail if the model is not cached."),
]


def _fail(message: str, code: int = 1) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _on_download(file: ModelFile) -> None:
    typer.echo(f"Downloading {file.path} ({file.size / 1e6:.0f} MB, one time)...", err=True)


def _embedder(name: str, offline: bool) -> Embedder | None:
    if name not in EMBEDDER_NAMES:
        _fail(f"unknown embedder {name!r}; choose one of {', '.join(EMBEDDER_NAMES)}", code=2)
    return make_embedder(name, offline=offline, on_download=_on_download)


def _utf8_output() -> None:
    """Write UTF-8 to pipes and files, on every OS.

    On Windows, Python encodes redirected output in the ANSI code page
    (cp1252 and the like), so `notelight search > hits.txt` crashed on a note
    with a character outside it, and `--json` was not UTF-8 as JSON requires.
    A console is unaffected: Python already writes Unicode to it.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and codecs.lookup(stream.encoding).name != "utf-8":
            stream.reconfigure(encoding="utf-8", errors=stream.errors)


@app.callback()
def main() -> None:
    """notelight command-line interface."""
    _utf8_output()


@app.command()
def index(
    folder: Annotated[
        Path,
        typer.Argument(
            help="Folder of Markdown, text, PDF and DOCX files to index.", show_default=False
        ),
    ],
    embedder: EmbedderOpt = None,
    offline: OfflineOpt = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print only errors.")] = False,
) -> None:
    """Create or update the index for FOLDER. Only changed files are re-embedded.

    Without --embedder, an existing index keeps the embedder it was built
    with, and a new index uses bge-small.
    """
    emb = _embedder(embedder or _current_embedder_name(folder) or "bge-small", offline)
    announced = False

    def progress(stage: str, done: int, total: int) -> None:
        nonlocal announced
        if stage == "embed" and not announced and not quiet:
            typer.echo(f"Embedding {total} new chunks...", err=True)
            announced = True

    try:
        stats = sync(folder, emb, progress=progress)
    except NotADirectoryError as e:
        _fail(str(e), code=2)
    except NotelightError as e:
        _fail(str(e))
    if not quiet:
        _print_sync(stats)


def _current_embedder_name(folder: Path) -> str | None:
    """The registry name of the embedder FOLDER's index was built with, or
    None if the folder has no readable index yet."""
    try:
        indexed_with = Searcher(folder, None).manifest.get("embedder")
    except (NotelightError, OSError):
        return None  # not indexed, or unreadable: the sync rebuilds it
    name = name_for_model_id(indexed_with)
    if name is None:
        _fail(
            f"the index for {paths.normalize_folder(folder)} uses embedder {indexed_with!r}, "
            "which this version does not know; pass --embedder to rebuild it",
            code=2,
        )
    return name


def _print_sync(s: SyncStats) -> None:
    for skipped in s.skipped[:10]:
        typer.echo(f"  skipped {skipped.path}: {skipped.reason}", err=True)
    if len(s.skipped) > 10:
        typer.echo(f"  ... and {len(s.skipped) - 10} more skipped", err=True)
    if s.up_to_date:
        typer.echo(f"{s.folder}: up to date ({s.files} files, {s.chunks} chunks)")
        return
    typer.echo(
        f"{s.folder}: {s.files} files ({s.added} added, {s.changed} changed, "
        f"{s.removed} removed), {s.chunks} chunks, {s.embedded} embedded, "
        f"generation {s.generation}, {s.seconds:.1f}s"
    )


def _pick_folder(folder: Path | None) -> Path:
    """The folder to search: the given one, else the only index, else the
    indexed folder containing the current directory."""
    if folder is not None:
        return folder
    indexes = [i for i in list_indexes() if i.error is None]
    if len(indexes) == 1:
        return indexes[0].folder
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if describe_index(paths.index_dir_for(candidate)) is not None:
            return candidate
    if not indexes:
        _fail("nothing is indexed yet; run `notelight index <folder>` first")
    listing = "\n".join(f"  {i.folder}" for i in indexes)
    _fail(f"several folders are indexed; choose one with --folder:\n{listing}", code=2)


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(
            help="What to look for. Quote exact phrases: '\"lock file\" windows'.",
            show_default=False,
        ),
    ],
    folder: Annotated[
        Path | None,
        typer.Option(
            "--folder",
            "-f",
            help="Indexed folder to search. Default: the only index, or the one containing "
            "the current directory.",
            show_default=False,
        ),
    ] = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Number of results.")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    embedder: EmbedderOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """Search an indexed folder (hybrid BM25 + embeddings)."""
    target = _pick_folder(folder)
    try:
        probe = Searcher(target, None)
        indexed_with = probe.manifest.get("embedder")
        name = embedder or name_for_model_id(indexed_with)
        if name is None:
            _fail(
                f"the index uses embedder {indexed_with!r}, which this version does not know; "
                "re-index it or pass --embedder none for keyword search"
            )
        emb = _embedder(name, offline)
        searcher = Searcher(target, emb) if emb is not None else probe
        results = searcher.search(query, k=k)
    except NotelightError as e:
        _fail(str(e), code=2 if isinstance(e, IndexNotFoundError) else 1)

    if as_json:
        payload = {
            "query": query,
            "folder": str(searcher.folder),
            "results": [r.to_dict() for r in results],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not results:
        typer.echo("No results.")
        return
    # Colour only on a terminal (click strips it from pipes), and never with NO_COLOR.
    color = False if os.environ.get("NO_COLOR") else None
    terms = query_terms(query)
    for n, r in enumerate(results, 1):
        heading = _highlight(" > ".join(r.heading_path), terms)
        where = f"{r.location} ({r.pages})" if r.pages else r.location
        typer.secho(f"{n:>2}. {where}", fg=typer.colors.CYAN, nl=False, color=color)
        typer.echo(f"  {heading}" if heading else "", color=color)
        typer.echo(f"    {_highlight(snippet(r.text, terms=terms), terms)}", color=color)


def _highlight(text: str, terms: frozenset[str]) -> str:
    """`text` with the words matching the query in bold yellow."""
    parts, last = [], 0
    for start, end in match_spans(text, terms):
        parts += [text[last:start], typer.style(text[start:end], fg=typer.colors.YELLOW, bold=True)]
        last = end
    parts.append(text[last:])
    return "".join(parts)


def _print_info(i: IndexInfo) -> None:
    typer.secho(str(i.folder), bold=True)
    if i.error is not None:
        typer.echo(f"  error: {i.error}")
        return
    typer.echo(f"  {i.files} files, {i.chunks} chunks, generation {i.generation}")
    typer.echo(f"  embedder: {i.embedder or 'none (keyword-only)'}")
    typer.echo(f"  index: {i.index_dir} ({i.size_bytes / 1e6:.1f} MB)")


@app.command()
def status(
    folder: Annotated[
        Path | None, typer.Argument(help="Show only this folder.", show_default=False)
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print as JSON.")] = False,
) -> None:
    """List indexed folders and their size, embedder and generation."""
    if folder is not None:
        info = describe_index(paths.index_dir_for(folder))
        if info is None:
            _fail(f"{paths.normalize_folder(folder)} is not indexed", code=2)
        infos = [info]
    else:
        infos = list_indexes()
    if as_json:
        typer.echo(json.dumps([i.to_dict() for i in infos], indent=2))
        return
    if not infos:
        typer.echo(f"No indexes yet (data directory: {paths.data_dir()}).")
        return
    for i in infos:
        _print_info(i)


@app.command()
def mcp(
    folders: Annotated[
        list[Path] | None,
        typer.Option(
            "--folder",
            "-f",
            help="Serve only this indexed folder (repeat for several). Default: every indexed "
            "folder, including ones indexed later.",
            show_default=False,
        ),
    ] = None,
    offline: OfflineOpt = False,
) -> None:
    """Run the MCP server over stdio, for Claude, Cursor and other AI agents.

    Searching is read-only: the server never changes an index, so it can run
    while `notelight index` updates one.
    """
    from notelight.mcp_server import serve

    for folder in folders or []:
        if describe_index(paths.index_dir_for(folder)) is None:
            target = paths.normalize_folder(folder)
            _fail(f"{target} is not indexed; run `notelight index {target}` first", code=2)
    serve(folders or None, offline=offline)


@app.command("serve")
def serve_web(
    folders: Annotated[
        list[Path] | None,
        typer.Option(
            "--folder",
            "-f",
            help="Serve only this indexed folder (repeat for several). Default: every indexed "
            "folder, including ones indexed later.",
            show_default=False,
        ),
    ] = None,
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535, help="Port.")] = 8765,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the page in the default browser.")
    ] = False,
    offline: OfflineOpt = False,
) -> None:
    """Run the web UI on http://127.0.0.1:PORT (needs `pip install notelight[web]`).

    It listens on localhost only and, like `notelight mcp`, never changes an index.
    """
    try:
        from notelight.web import run
    except ImportError as e:
        if e.name not in {"fastapi", "uvicorn", "starlette"}:
            raise
        _fail('the web UI needs extra packages; install them with: pip install "notelight[web]"')

    for folder in folders or []:
        if describe_index(paths.index_dir_for(folder)) is None:
            target = paths.normalize_folder(folder)
            _fail(f"{target} is not indexed; run `notelight index {target}` first", code=2)

    def ready(url: str) -> None:
        typer.echo(f"notelight web UI on {url} (Ctrl+C to stop)", err=True)
        if open_browser:
            import webbrowser

            webbrowser.open(url)

    run(folders or None, port=port, offline=offline, on_ready=ready)


@app.command("eval")
def eval_(
    datasets: Annotated[
        list[str] | None,
        typer.Argument(
            help="BEIR dataset names (scifact, nfcorpus, fiqa), notes (short queries over "
            "notelight's own docs, bundled), or folders in the BEIR layout. "
            "Default: the three BEIR datasets.",
            show_default=False,
        ),
    ] = None,
    embedder: EmbedderOpt = None,
    methods: Annotated[
        str | None,
        typer.Option(
            "--methods",
            "-m",
            help="Comma-separated methods to run. Default: all.",
            show_default=False,
        ),
    ] = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Cut-off for recall and nDCG.")] = 10,
    candidates: Annotated[
        int, typer.Option(min=1, help="Binary/ternary candidates re-scored with int8.")
    ] = 200,
    phrase_boost: Annotated[
        float,
        typer.Option(min=0.0, help="Hybrid bonus for query terms in order; 0 turns it off."),
    ] = 1.0,
    coverage_power: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Hybrid weight of a BM25 hit: (share of query terms it contains) ** this. "
            "0 is plain RRF.",
        ),
    ] = 2.0,
    max_queries: Annotated[
        int | None,
        typer.Option(min=1, help="Evaluate only the first N queries.", show_default=False),
    ] = None,
    split: Annotated[
        str,
        typer.Option(
            help="notes only: tune (for choosing settings), heldout (report once) or all."
        ),
    ] = "tune",
    max_chars: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="notes only: chunk size to evaluate. Default: the indexer's (1200).",
            show_default=False,
        ),
    ] = None,
    baseline: Annotated[
        bool, typer.Option("--baseline/--no-baseline", help="Also run the NumPy baseline.")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    offline: OfflineOpt = False,
) -> None:
    """Measure recall@k, nDCG@k and MRR@k of every retrieval method on BEIR
    datasets or the bundled notes benchmark.

    Datasets and the model are downloaded once; corpus embeddings are cached.
    """
    from notelight.eval import (
        BEIR_DATASETS,
        METHODS,
        NOTES_SPLITS,
        evaluate,
        format_markdown,
        load_dataset,
    )
    from notelight.eval.datasets import DatasetSpec

    name = embedder or "bge-small"
    if name == "none":
        _fail("eval needs an embedder; choose bge-small or hashing", code=2)
    emb = _embedder(name, offline)
    assert emb is not None
    chosen = [m.strip() for m in methods.split(",")] if methods else list(METHODS)
    unknown = [m for m in chosen if m not in METHODS]
    if unknown:
        _fail(f"unknown method(s) {', '.join(unknown)}; choose from {', '.join(METHODS)}", code=2)
    if split not in NOTES_SPLITS:
        _fail(f"unknown split {split}; choose from {', '.join(NOTES_SPLITS)}", code=2)

    def on_dataset_download(spec: DatasetSpec) -> None:
        typer.echo(f"Downloading {spec.name} ({spec.size / 1e6:.1f} MB, one time)...", err=True)

    def progress(stage: str, done: int, total: int) -> None:
        if stage == "embed-docs" and (done == total or done % 2048 < 256):
            typer.echo(f"  embedded {done:,}/{total:,} documents", err=True)

    results = []
    for target in datasets or list(BEIR_DATASETS):
        try:
            data = load_dataset(
                target,
                offline=offline,
                on_download=on_dataset_download,
                notes_split=split,
                max_chars=max_chars,
            )
            typer.echo(
                f"{data.name}: {len(data.doc_ids):,} documents, {len(data.queries)} queries",
                err=True,
            )
            results.append(
                evaluate(
                    data,
                    emb,
                    methods=chosen,
                    k=k,
                    candidates=candidates,
                    phrase_boost=phrase_boost,
                    coverage_power=coverage_power,
                    max_queries=max_queries,
                    baseline=baseline,
                    progress=progress,
                )
            )
        except NotelightError as e:
            _fail(str(e))
    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        typer.echo(format_markdown(results))


@app.command()
def version() -> None:
    """Print notelight, Rust core, Python and NumPy versions."""
    typer.echo(f"notelight {notelight.__version__}")
    typer.echo(f"  rust core  {notelight.core_version}")
    typer.echo(f"  python     {platform.python_version()} ({platform.system()})")
    typer.echo(f"  numpy      {np.__version__}")
    typer.echo(f"  data dir   {paths.data_dir()}")


if __name__ == "__main__":
    app()
