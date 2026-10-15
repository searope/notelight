"""Shared fixtures: an isolated data directory, a notes folder, and embedders."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from notelight.embed import HashingEmbedder
from notelight.paths import HOME_ENV

NOTES = {
    "rust/ownership.md": """\
# Rust

## Ownership

Every value has a single owner. When the owner goes out of scope the value is
dropped. Borrowing lends access without moving ownership.

## Lifetimes

Lifetimes describe how long references stay valid.
""",
    "python/typing.md": """\
# Python typing

Protocols give structural typing: any class with the right methods matches.
Use mypy in strict mode to check annotations.
""",
    "recipes/bread.txt": """\
Sourdough bread

Mix flour, water and starter. Let the dough rise overnight, then bake at 230 C.
""",
}


@pytest.fixture(autouse=True)
def notelight_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Every test gets its own data directory, so no test sees a real index."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv(HOME_ENV, str(home))
    return home


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    """A small folder of Markdown and text notes in nested directories."""
    root = tmp_path / "notes"
    for rel, text in NOTES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class CountingEmbedder(HashingEmbedder):
    """A HashingEmbedder that records how many texts it embedded."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim)
        self.embedded: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        self.embedded.extend(texts)
        return super().embed_documents(texts)


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()
