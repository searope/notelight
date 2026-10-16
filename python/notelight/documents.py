"""Text extraction for PDF and DOCX files (optional extras).

`pip install notelight[pdf]` adds pypdf, `notelight[docx]` adds python-docx,
and `notelight[documents]` adds both. Without the extra, these files are
skipped with a reason that says what to install.

The extracted text is what gets chunked and cited, and `read_context`
extracts it again the same way, so line numbers in citations always refer to
it, not to the original file:

- **PDF**: each page's text, ending in a newline; pages are joined with a
  form feed (`\\f`), so every page after the first starts with `\\f` on its
  first line. Pages never share a line, so a line number maps to exactly one
  page (`page_starts`, `page_range`).
- **DOCX**: one paragraph per line with a blank line between paragraphs.
  Paragraphs styled `Heading 1` to `Heading 6` become Markdown headings
  (`# ...`), so the Markdown chunker gives them heading paths. Table rows
  become `cell | cell` lines. A body line that Markdown would read as a heading
  or a code fence is escaped with a backslash.

Extraction is versioned (`extractor_id`): the id includes `EXTRACT_VERSION`
and the library version, and it is part of the chunk cache key, so upgrading
either re-extracts the file instead of mixing old chunks with new line
numbers.
"""

from __future__ import annotations

import bisect
import io
import re
from typing import TYPE_CHECKING

from notelight.ingest import DOCUMENT_SUFFIXES, MARKDOWN_SUFFIXES, Decoded, decode

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument

__all__ = [
    "EXTRACT_VERSION",
    "PAGE_BREAK",
    "content_mode",
    "extract",
    "extractor_id",
    "is_missing_extra_reason",
    "load_text",
    "missing_extra",
    "page_range",
    "page_starts",
]

#: Bump when the extracted text for the same file and library could change.
EXTRACT_VERSION = 1
#: Separates PDF pages in extracted text; starts each page after the first.
PAGE_BREAK = "\f"

_LABELS = {".pdf": "PDF", ".docx": "DOCX"}
_HEADING_STYLE = re.compile(r"^Heading ([1-9])$")
# Lines the Markdown chunker would take as structure: an ATX heading or a fence.
_MARKDOWN_SYNTAX = re.compile(r"^ {0,3}(#{1,6}(\s|$)|```|~~~)")
# Characters that would shift line numbers or page breaks if left in.
_CONTROL = str.maketrans({"\x00": None, "\f": None, "\r": None, "\v": "\n"})


def _missing_reason(suffix: str) -> str:
    return (
        f"{_LABELS[suffix]} support is not installed; "
        f"install notelight[{DOCUMENT_SUFFIXES[suffix]}]"
    )


def missing_extra(suffix: str) -> str | None:
    """Why a `suffix` file cannot be read right now, or None if it can.

    Returns None for suffixes that are not documents.
    """
    suffix = suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf  # noqa: F401
        except ImportError:
            return _missing_reason(suffix)
    elif suffix == ".docx":
        try:
            import docx  # noqa: F401
        except ImportError:
            return _missing_reason(suffix)
    return None


def is_missing_extra_reason(reason: str) -> bool:
    """Whether `reason` is the skip reason `missing_extra` gives for some suffix."""
    return reason in {_missing_reason(s) for s in DOCUMENT_SUFFIXES}


def extractor_id(suffix: str) -> str:
    """Identifies how a `suffix` file is turned into text, for cache keys.

    Raises:
        ImportError: the extra for `suffix` is not installed.
        ValueError: `suffix` is not a document suffix.
    """
    suffix = suffix.lower()
    if suffix == ".pdf":
        import pypdf

        return f"pdf{EXTRACT_VERSION}:pypdf-{pypdf.__version__}"
    if suffix == ".docx":
        from importlib.metadata import version

        import docx  # noqa: F401

        return f"docx{EXTRACT_VERSION}:python-docx-{version('python-docx')}"
    raise ValueError(f"not a document suffix: {suffix}")


def extract(suffix: str, data: bytes) -> Decoded:
    """Extract the text of a PDF or DOCX file from its bytes.

    Never raises for bad input: encrypted, image-only, empty and malformed
    files come back with `text=None` and a `skip_reason`.

    Raises:
        ImportError: the extra for `suffix` is not installed (check
            `missing_extra` first).
        ValueError: `suffix` is not a document suffix.
    """
    suffix = suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    raise ValueError(f"not a document suffix: {suffix}")


def content_mode(suffix: str) -> str:
    """How a file with `suffix` becomes chunks: `md`, `txt`, or the document
    extractor's id. Part of the chunk cache key.

    Raises:
        ImportError: the extra for a document `suffix` is not installed.
    """
    suffix = suffix.lower()
    if suffix in DOCUMENT_SUFFIXES:
        return extractor_id(suffix)
    return "md" if suffix in MARKDOWN_SUFFIXES else "txt"


def load_text(suffix: str, data: bytes) -> Decoded:
    """The indexed text of a file: extracted for documents, decoded otherwise.

    Raises:
        ImportError: the extra for a document `suffix` is not installed.
    """
    if suffix.lower() in DOCUMENT_SUFFIXES:
        return extract(suffix, data)
    return decode(data)


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").translate(_CONTROL)


def _extract_pdf(data: bytes) -> Decoded:
    import pypdf
    from pypdf.errors import DependencyError

    # pypdf raises many exception types on malformed input; a bad file must
    # never stop a sync, so anything it raises becomes a skip reason.
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                # Many PDFs are encrypted only to restrict printing or
                # copying; they open with an empty user password.
                opened = reader.decrypt("")
            except DependencyError:
                return Decoded(None, skip_reason="encrypted PDF (its cipher needs pypdf[crypto])")
            if opened == pypdf.PasswordType.NOT_DECRYPTED:
                return Decoded(None, skip_reason="encrypted PDF (password required)")
        pages = [_clean(page.extract_text() or "") for page in reader.pages]
    except Exception as e:
        return Decoded(None, skip_reason=f"unreadable PDF: {type(e).__name__}: {e}")
    if not any(p.strip() for p in pages):
        return Decoded(None, skip_reason="PDF has no text layer (scanned or image-only?)")
    return Decoded(PAGE_BREAK.join(p.rstrip("\n") + "\n" for p in pages))


def _extract_docx(data: bytes) -> Decoded:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
        lines = _docx_lines(document)
    except Exception as e:
        return Decoded(None, skip_reason=f"unreadable DOCX: {type(e).__name__}: {e}")
    if not any(line.strip() for line in lines):
        return Decoded(None, skip_reason="DOCX has no text")
    return Decoded("".join(line + "\n" for line in lines))


def _docx_lines(document: DocxDocument) -> list[str]:
    """Body paragraphs and tables in document order, as lines with a blank
    line after each paragraph or table."""
    from docx.table import Table

    lines: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            rows = [
                " | ".join(" ".join(_clean(cell.text).split()) for cell in row.cells)
                for row in item.rows
            ]
            rows = [_escape(r) for r in rows if r.strip(" |")]
            if rows:
                lines.extend([*rows, ""])
            continue
        text = _clean(item.text)
        if not text.strip():
            continue
        style = item.style.name if item.style is not None else None
        match = _HEADING_STYLE.match(style or "")
        if match is not None:
            level = min(int(match.group(1)), 6)
            lines.extend(["#" * level + " " + " ".join(text.split()), ""])
        else:
            lines.extend([*(_escape(line) for line in text.split("\n")), ""])
    return lines


def _escape(line: str) -> str:
    """Keep a body line from being read as a Markdown heading or fence."""
    return "\\" + line if _MARKDOWN_SYNTAX.match(line) else line


def page_starts(text: str) -> list[int]:
    """The 1-based line on which each page of extracted PDF text starts.

    Text without page breaks is one page starting on line 1.
    """
    starts = [1]
    for number, line in enumerate(text.split("\n"), start=1):
        if number > 1 and line.startswith(PAGE_BREAK):
            starts.append(number)
    return starts


def page_range(starts: list[int], line_start: int, line_end: int) -> tuple[int, int]:
    """The 1-based first and last page that lines `line_start..=line_end` touch."""
    first = bisect.bisect_right(starts, line_start)
    last = bisect.bisect_right(starts, line_end)
    return max(first, 1), max(last, 1)
