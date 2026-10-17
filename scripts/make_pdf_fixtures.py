"""Write the small PDFs that test PDF extraction in the browser.

    python scripts/make_pdf_fixtures.py

The PDFs go to web-demo/tests/fixtures/pdf/ (a few KiB in all, committed),
with pdf.json beside them: for each file, the text its lines spell, joined
the way `notelight.documents` joins pages (each page ends in a newline,
every page after the first starts with a form feed), and what pypdf extracts
from it (or its skip reason). web-demo/tests/documents.test.mjs requires
pdf.js to give the expected text and pages, and the same skip reasons.

The files are written by hand, object by object, so their text is known
exactly: Helvetica text shown with `Tj` and moved down a line with `T*`.

- report.pdf: four pages, the third blank (as in tests/test_documents.py);
- field-guide.pdf: five pages of a few hundred words each, so a 1200-character
  chunker cuts them into chunks on single pages and across page breaks;
- owner-only.pdf: report.pdf encrypted (RC4-128) with an empty user password,
  which opens without asking (many PDFs restrict only printing or copying);
- locked.pdf: the same with the user password "secret", which is skipped;
- cjk.pdf: Japanese in a non-embedded CID font with the predefined CMap
  UniJIS-UCS2-H, which pdf.js reads from the vendored cmaps/ folder;
- scanned.pdf: two pages that only draw an image, so no text layer;
- broken.pdf: a PDF header followed by garbage.

The encrypted files use a fixed file identifier, so the output is the same on
every run. Needs pypdf with its crypto extra (`pip install notelight[pdf]
cryptography`).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pypdf
from pypdf.generic import ArrayObject, ByteStringObject

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web-demo" / "tests" / "fixtures" / "pdf"
PAGE_BREAK = "\f"

REPORT = [
    ["Quarterly report", "Revenue grew in every region."],
    ["Glaciers retreat as summers warm.", "Meltwater feeds the rivers."],
    [],
    ["Appendix: sourdough needs a starter."],
]

_TOPICS = [
    ("Ferns", "fronds", "spores gather under each frond in brown clusters called sori"),
    ("Mosses", "cushions", "a moss cushion holds water like a sponge after rain"),
    ("Lichens", "crusts", "lichen is a partnership of a fungus and an alga on bare rock"),
    ("Liverworts", "thalli", "liverworts spread as flat green thalli along stream banks"),
    ("Horsetails", "rhizomes", "horsetails rise from rhizomes and their stems feel like sandpaper"),
]


def _field_guide() -> list[list[str]]:
    pages = []
    for number, (title, word, fact) in enumerate(_TOPICS, start=1):
        lines = [f"Chapter {number}: {title}"]
        for i in range(1, 15):
            lines.append(f"Observation {number}.{i}: the {word} by the trail; {fact}.")
        pages.append(lines)
    return pages


CJK = [["日本語のテキスト", "東京の天気は晴れ"], ["二ページ目"]]

HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
# A CID font the reader must not have embedded, so text maps through the
# predefined CMaps (UniJIS-UCS2-H, then Adobe-Japan1-UCS2 for Unicode).
HEISEI = (
    b"<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiKakuGo-W5 /Encoding /UniJIS-UCS2-H "
    b"/DescendantFonts [<< /Type /Font /Subtype /CIDFontType0 /BaseFont /HeiseiKakuGo-W5 "
    b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> /DW 1000 >>] >>"
)


def _literal(text: str) -> str:
    """A PDF literal string (Latin-1 text)."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})"


def _ucs2(text: str) -> str:
    """A PDF hex string of UTF-16BE code units (for UniJIS-UCS2-H)."""
    return "<" + text.encode("utf-16-be").hex().upper() + ">"


def make_pdf(pages: list[list[str]], *, image: bool = False, cjk: bool = False) -> bytes:
    """A minimal valid PDF: one text line per entry on each page (in Heisei
    Kaku Gothic with `cjk`, else Helvetica), or only an image (a 2x2 grey
    square) on every page with `image`."""
    first_page = 5
    page_ids = [first_page + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    pixels = b"\x00\x80\x80\xff"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        HEISEI if cjk else HELVETICA,
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length 4 >>\nstream\n" + pixels + b"\nendstream",
    ]
    for page_id, lines in zip(page_ids, pages, strict=True):
        if image:
            content = b"q 200 0 0 200 72 500 cm /Im1 Do Q"
        else:
            show = _ucs2 if cjk else _literal
            shows = " ".join(f"{show(line)} Tj T*" for line in lines)
            content = f"BT /F1 10 Tf 13 TL 56 760 Td {shows} ET".encode() if lines else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> /XObject << /Im1 4 0 R >> >> "
            f"/Contents {page_id + 1} 0 R >>".encode()
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def encrypt(data: bytes, user_password: str) -> bytes:
    """`data` encrypted with RC4-128 (no random IV) and a fixed file id."""
    writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(data)))
    file_id = ByteStringObject(bytes(range(16)))
    writer._ID = ArrayObject([file_id, file_id])
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def joined(pages: list[list[str]]) -> str:
    """The text `documents._extract_pdf` makes of these lines."""
    return PAGE_BREAK.join("".join(line + "\n" for line in lines) or "\n" for lines in pages)


def pypdf_text(data: bytes) -> dict[str, str | None]:
    from notelight import documents

    decoded = documents.extract(".pdf", data)
    return {"text": decoded.text, "skipReason": decoded.skip_reason}


def main() -> None:
    report = make_pdf(REPORT)
    guide = _field_guide()
    files: dict[str, tuple[bytes, str | None]] = {
        "report.pdf": (report, joined(REPORT)),
        "field-guide.pdf": (make_pdf(guide), joined(guide)),
        "owner-only.pdf": (encrypt(report, ""), joined(REPORT)),
        "locked.pdf": (encrypt(report, "secret"), None),
        "cjk.pdf": (make_pdf(CJK, cjk=True), joined(CJK)),
        "scanned.pdf": (make_pdf([[], []], image=True), None),
        "broken.pdf": (b"%PDF-1.4\ngarbage\n", None),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (data, text) in files.items():
        (OUT / name).write_bytes(data)
        manifest[name] = {"bytes": len(data), "expected": text, "pypdf": pypdf_text(data)}
    text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    (OUT / "pdf.json").write_text(text, encoding="utf-8", newline="\n")
    total = sum(len(d) for d, _ in files.values())
    print(
        f"wrote {len(files)} PDFs ({total / 1024:.1f} KiB) and pdf.json to {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
