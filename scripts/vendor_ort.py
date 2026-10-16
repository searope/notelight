"""Vendor ONNX Runtime Web into web-demo/vendor/ for the browser embedder.

    python scripts/vendor_ort.py

The demo embeds queries with bge-small in the browser, and the page loads
nothing from a CDN: the runtime is served from `web-demo/vendor/` next to the
page. This script downloads one pinned npm release of `onnxruntime-web`,
checks the tarball against the SHA-512 integrity npm publishes for it, and
extracts the three files the WASM backend needs, each checked against its own
SHA-256. The files are 14 MB, so they are not committed (like `pkg/`); the
licence is, in `web-demo/THIRD-PARTY-NOTICES.md`.

To move to another release, change VERSION, then copy the new `dist.integrity`
from `npm view onnxruntime-web@<version> dist.integrity` and the files'
SHA-256s from a trusted extraction (the script prints any that differ).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web-demo" / "vendor" / "onnxruntime-web"

VERSION = "1.30.0"
TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz"
# `npm view onnxruntime-web@1.30.0 dist.integrity`
INTEGRITY = "sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=="  # noqa: E501
# The WASM-only ES module, the Emscripten glue it imports from its own folder,
# and the SIMD + threads WebAssembly binary (it runs single-threaded unless the
# page is cross-origin isolated).
FILES = {
    "ort.wasm.min.mjs": "219e6a1fc8a9938268d18efca3c91d310bd2f4a59bbd13744df5b2b7fc6cee3b",
    "ort-wasm-simd-threaded.mjs": (
        "e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b"
    ),
    "ort-wasm-simd-threaded.wasm": (
        "3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2"
    ),
}


def up_to_date(out: Path) -> bool:
    """Whether every file is already present with the pinned SHA-256."""
    return all(
        (out / name).is_file() and hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
        for name, digest in FILES.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT, help="where to write the files")
    args = parser.parse_args()
    out: Path = args.out_dir

    if up_to_date(out):
        print(f"onnxruntime-web {VERSION} is already in {out}")
        return

    print(f"downloading {TARBALL}")
    with urllib.request.urlopen(TARBALL, timeout=120) as response:
        data = response.read()
    algorithm, expected = INTEGRITY.split("-", 1)
    actual = base64.b64encode(hashlib.new(algorithm, data).digest()).decode()
    if actual != expected:
        sys.exit(f"{TARBALL} failed its integrity check: expected {expected}, got {actual}")

    out.mkdir(parents=True, exist_ok=True)
    bad = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name, digest in FILES.items():
            member = tar.extractfile(f"package/dist/{name}")
            if member is None:
                sys.exit(f"package/dist/{name} is not in the tarball")
            content = member.read()
            got = hashlib.sha256(content).hexdigest()
            if got != digest:
                bad.append(f"  {name}: expected {digest}, got {got}")
                continue
            (out / name).write_bytes(content)
    if bad:
        sys.exit("files differ from their pinned SHA-256:\n" + "\n".join(bad))
    total = sum((out / n).stat().st_size for n in FILES)
    print(f"wrote onnxruntime-web {VERSION} to {out.relative_to(ROOT)} ({total / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
