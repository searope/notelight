"""Find and read the text files in a folder.

Skipped while walking:

- hidden files and directories (names starting with `.`), which covers
  `.git`, `.hg`, `.svn`, `.venv` and editor folders;
- common build and dependency directories (`node_modules`, `target`, ...);
- anything matched by a `.notelightignore` file at the folder root, which
  uses `.gitignore` syntax;
- files with other extensions, and files over `max_bytes` for their type
  (`MAX_FILE_BYTES` for text, `MAX_DOCUMENT_BYTES` for PDF and DOCX).

PDF and DOCX files are always found, even without the extra that reads them:
the indexer then records them as skipped with a reason that says what to
install (see `notelight.documents`).

Paths are reported relative to the folder with `/` separators on every OS, so
an index built on Windows reads the same as one built on Linux.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

__all__ = [
    "DOCUMENT_SUFFIXES",
    "IGNORE_FILE",
    "INDEXED_SUFFIXES",
    "MARKDOWN_SUFFIXES",
    "MAX_DOCUMENT_BYTES",
    "MAX_FILE_BYTES",
    "SKIP_DIRS",
    "TEXT_SUFFIXES",
    "Decoded",
    "SourceFile",
    "decode",
    "max_bytes",
    "walk",
]

IGNORE_FILE = ".notelightignore"
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
TEXT_SUFFIXES = MARKDOWN_SUFFIXES | {".txt"}
#: Document suffix -> the extra that reads it (`pip install notelight[pdf]`).
DOCUMENT_SUFFIXES = {".pdf": "pdf", ".docx": "docx"}
INDEXED_SUFFIXES = TEXT_SUFFIXES | DOCUMENT_SUFFIXES.keys()
MAX_FILE_BYTES = 10 * 1024 * 1024
#: PDFs carry fonts and images, so they get a larger limit than text.
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
SKIP_DIRS = frozenset(
    {"node_modules", "target", "build", "dist", "__pycache__", "venv", "site-packages"}
)
_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class SourceFile:
    """A candidate file found by `walk`."""

    rel_path: str
    """Path relative to the folder root, `/`-separated."""
    abs_path: Path
    size: int
    mtime_ns: int

    @property
    def suffix(self) -> str:
        """The lower-cased file extension, such as `.md` or `.pdf`."""
        return Path(self.rel_path).suffix.lower()

    @property
    def is_markdown(self) -> bool:
        """Whether to chunk with Markdown structure (headings, code fences).

        True for DOCX too: its text is extracted with Markdown headings.
        """
        return self.suffix in MARKDOWN_SUFFIXES or self.suffix == ".docx"


def max_bytes(suffix: str) -> int:
    """Largest file of type `suffix` that is indexed."""
    return MAX_DOCUMENT_BYTES if suffix.lower() in DOCUMENT_SUFFIXES else MAX_FILE_BYTES


def _load_ignore(root: Path) -> pathspec.GitIgnoreSpec | None:
    ignore = root / IGNORE_FILE
    if not ignore.is_file():
        return None
    lines = ignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.GitIgnoreSpec.from_lines(lines)


def walk(root: Path) -> Iterator[SourceFile]:
    """Yield indexable files under `root` in sorted path order."""
    spec = _load_ignore(root)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_dir = here.relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"

        # Prune in place so os.walk does not descend; sort for stable order.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in SKIP_DIRS
            and not (spec is not None and spec.match_file(prefix + d + "/"))
        )
        for name in sorted(filenames):
            suffix = Path(name).suffix.lower()
            if name.startswith(".") or suffix not in INDEXED_SUFFIXES:
                continue
            rel = prefix + name
            if spec is not None and spec.match_file(rel):
                continue
            path = here / name
            try:
                st = path.stat()
            except OSError:
                continue  # vanished or unreadable; the next sync will see it
            if st.st_size > max_bytes(suffix):
                continue
            yield SourceFile(rel, path, st.st_size, st.st_mtime_ns)


@dataclass(frozen=True)
class Decoded:
    """Result of `decode` or of document extraction: the text, or why the
    file was skipped."""

    text: str | None
    encoding: str | None = None
    skip_reason: str | None = None


def decode(data: bytes) -> Decoded:
    """Decode file bytes as text.

    UTF-8 (with or without a BOM) is tried first, then Windows-1252, which
    covers most legacy Western text. Files containing NUL bytes are treated as
    binary and skipped. Line endings are normalized to `\\n`; this does not
    change line numbers.
    """
    if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        return Decoded(None, skip_reason="binary")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return Decoded(text.replace("\r\n", "\n").replace("\r", "\n"), encoding=encoding)
    return Decoded(None, skip_reason="not UTF-8 or Windows-1252 text")
