"""Vendor pdf.js into web-demo/vendor/pdfjs/ for PDF text extraction in the browser.

    python scripts/vendor_pdfjs.py

The browser app reads the text of PDFs with pdf.js, and the page loads
nothing from a CDN: like onnxruntime-web (scripts/vendor_ort.py), pdf.js is
served from `web-demo/vendor/` next to the page. This script downloads one
pinned npm release of `pdfjs-dist`, checks the tarball against the SHA-512
integrity npm publishes for it, and extracts:

- `legacy/build/pdf.min.mjs` and `legacy/build/pdf.worker.min.mjs` (the API
  and the parser it normally runs in a worker; the indexing worker imports
  both, see web-demo/documents.js), each checked against its own SHA-256.
  The legacy build, because the modern one uses JavaScript too new for
  Node 24 and for browsers a year or two old (`Map.getOrInsertComputed`);
- `cmaps/` (169 packed character maps for CJK fonts, fetched only by PDFs
  that use them), checked as one SHA-256 over the sorted file names and
  their SHA-256s;
- `LICENSE` (Apache-2.0), also in `web-demo/THIRD-PARTY-NOTICES.md`.

The standard font files and the image decoders (`standard_fonts/`, `wasm/`)
are left out: they draw pages and decode images, and the app only reads text.
The files are 3 MB, so they are not committed (like `pkg/`).

To move to another release, change VERSION, then copy the new `dist.integrity`
from `npm view pdfjs-dist@<version> dist.integrity` and the SHA-256s from a
trusted extraction (the script prints any that differ). Changing VERSION
changes the extractor id (`pdf1:pdfjs-<version>`) in web-demo/documents.js,
which must be bumped with it, so browser indexes re-extract their PDFs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web-demo" / "vendor" / "pdfjs"

VERSION = "6.3.289"
TARBALL = f"https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-{VERSION}.tgz"
# `npm view pdfjs-dist@6.3.289 dist.integrity`
INTEGRITY = "sha512-ZHjSVpDa3D6izMq8/04lvkhkATUmL9px6ChPaXc1k6nU2Mrhlg1/7F0bdUqCwUjw3NsPTfPZsMDUU6ZIcRaeQw=="  # noqa: E501
#: package path -> SHA-256; written flat into OUT.
FILES = {
    "legacy/build/pdf.min.mjs": "f401927e692efc7735e0cd528c490d0dd31b7f0972c122b7040df805be45cce4",
    "legacy/build/pdf.worker.min.mjs": (
        "a33cfe728c584fdba4fcc1fd54bcdc2f9f2f13889ddbb5b2bd1d0f8cbe49b84e"
    ),
    "LICENSE": "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594",
}
CMAPS = "cmaps/"
#: `cmaps_digest` of the release's cmaps/ folder.
CMAPS_DIGEST = "d674956416fc6e50acfbee981906e4db64d506dd1107c399b77ff050531fc5ee"


def cmaps_digest(files: dict[str, bytes]) -> str:
    """One SHA-256 over `name<TAB>sha256` lines, sorted by name."""
    lines = "".join(f"{n}\t{hashlib.sha256(files[n]).hexdigest()}\n" for n in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def flat(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def up_to_date(out: Path) -> bool:
    """Whether every file is already present with its pinned SHA-256."""
    for name, digest in FILES.items():
        path = out / flat(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    cmaps = out / "cmaps"
    if not cmaps.is_dir():
        return False
    return cmaps_digest({p.name: p.read_bytes() for p in cmaps.iterdir()}) == CMAPS_DIGEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT, help="where to write the files")
    args = parser.parse_args()
    out: Path = args.out_dir

    if up_to_date(out):
        print(f"pdf.js {VERSION} is already in {out}")
        return

    print(f"downloading {TARBALL}")
    with urllib.request.urlopen(TARBALL, timeout=120) as response:
        data = response.read()
    algorithm, expected = INTEGRITY.split("-", 1)
    actual = base64.b64encode(hashlib.new(algorithm, data).digest()).decode()
    if actual != expected:
        sys.exit(f"{TARBALL} failed its integrity check: expected {expected}, got {actual}")

    files: dict[str, bytes] = {}
    cmaps: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix("package/")
            if name in FILES or (name.startswith(CMAPS) and member.isfile()):
                extracted = tar.extractfile(member)
                if extracted is None:
                    sys.exit(f"{member.name} is not a file")
                (files if name in FILES else cmaps)[name] = extracted.read()
    bad = []
    for name, digest in FILES.items():
        if name not in files:
            sys.exit(f"package/{name} is not in the tarball")
        got = hashlib.sha256(files[name]).hexdigest()
        if got != digest:
            bad.append(f"  {name}: expected {digest}, got {got}")
    cmaps = {flat(n): b for n, b in cmaps.items()}
    got = cmaps_digest(cmaps)
    if got != CMAPS_DIGEST:
        bad.append(f"  {CMAPS} ({len(cmaps)} files): expected {CMAPS_DIGEST}, got {got}")
    if bad:
        sys.exit("files differ from their pinned SHA-256:\n" + "\n".join(bad))

    if out.exists():
        shutil.rmtree(out)
    (out / "cmaps").mkdir(parents=True)
    for name, content in files.items():
        (out / flat(name)).write_bytes(content)
    for name, content in cmaps.items():
        (out / "cmaps" / name).write_bytes(content)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote pdf.js {VERSION} to {out.relative_to(ROOT)} ({total / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
