"""Command-line interface. Milestone 0 only reports versions; real commands land in M2."""

from __future__ import annotations

import platform

import numpy as np
import typer

import notelight

app = typer.Typer(
    name="notelight",
    help="Private semantic search over your own files, for you and your AI agents.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """notelight command-line interface."""


@app.command()
def version() -> None:
    """Print notelight, Rust core, Python and NumPy versions."""
    typer.echo(f"notelight {notelight.__version__}")
    typer.echo(f"  rust core  {notelight.core_version}")
    typer.echo(f"  python     {platform.python_version()} ({platform.system()})")
    typer.echo(f"  numpy      {np.__version__}")


if __name__ == "__main__":
    app()
