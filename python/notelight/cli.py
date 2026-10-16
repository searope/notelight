"""Command-line interface: `notelight index | search | status | mcp | eval | version`."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import typer

import notelight
from notelight import paths
from notelight.embed import EMBEDDER_NAMES, Embedder, ModelFile, make_embedder, name_for_model_id
from notelight.errors import IndexNotFoundError, NotelightError
from notelight.indexer import SyncStats, sync
from notelight.search import IndexInfo, Searcher, describe_index, list_indexes, snippet

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


@app.callback()
def main() -> None:
    """notelight command-line interface."""


@app.command()
def index(
    folder: Annotated[
        Path, typer.Argument(help="Folder of Markdown and text files to index.", show_default=False)
    ],
    embedder: EmbedderOpt = None,
    offline: OfflineOpt = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print only errors.")] = False,
) -> None:
    """Create or update the index for FOLDER. Only changed files are re-embedded."""
    emb = _embedder(embedder or "bge-small", offline)
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
    query: Annotated[str, typer.Argument(help="What to look for.", show_default=False)],
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
    for n, r in enumerate(results, 1):
        heading = " > ".join(r.heading_path)
        typer.secho(f"{n:>2}. {r.location}", fg=typer.colors.CYAN, nl=False)
        typer.echo(f"  {heading}" if heading else "")
        typer.echo(f"    {snippet(r.text)}")


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


@app.command("eval")
def eval_(
    datasets: Annotated[
        list[str] | None,
        typer.Argument(
            help="BEIR dataset names (scifact, nfcorpus, fiqa) or folders in the BEIR layout. "
            "Default: all three.",
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
    max_queries: Annotated[
        int | None,
        typer.Option(min=1, help="Evaluate only the first N queries.", show_default=False),
    ] = None,
    baseline: Annotated[
        bool, typer.Option("--baseline/--no-baseline", help="Also run the NumPy baseline.")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    offline: OfflineOpt = False,
) -> None:
    """Measure recall@k and nDCG@k of every retrieval method on BEIR datasets.

    Datasets and the model are downloaded once; corpus embeddings are cached.
    """
    from notelight.eval import BEIR_DATASETS, METHODS, evaluate, format_markdown, load_dataset
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

    def on_dataset_download(spec: DatasetSpec) -> None:
        typer.echo(f"Downloading {spec.name} ({spec.size / 1e6:.1f} MB, one time)...", err=True)

    def progress(stage: str, done: int, total: int) -> None:
        if stage == "embed-docs" and (done == total or done % 2048 < 256):
            typer.echo(f"  embedded {done:,}/{total:,} documents", err=True)

    results = []
    for target in datasets or list(BEIR_DATASETS):
        try:
            data = load_dataset(target, offline=offline, on_download=on_dataset_download)
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
