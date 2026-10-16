"""Assemble the browser demo into a folder a static host can serve.

    python scripts/build_wasm.py     # web-demo/pkg/
    python scripts/vendor_ort.py     # web-demo/vendor/
    python scripts/build_site.py     # _site/ (or --out DIR)

The GitHub Pages workflow (.github/workflows/pages.yml) runs the same three
steps and uploads the folder. The site is the page, its scripts, the wasm
package, the vendored onnxruntime-web and the committed sample index; tests
and Node-only benchmarks are left out. Nothing in it calls home: the only
network request the page makes is the one-time model download from Hugging
Face at the pinned revision, verified by SHA-256 (see web-demo/embed.js).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "web-demo"

#: Files and folders copied from web-demo/ (everything the page loads).
INCLUDE = [
    "index.html",
    "app.js",
    "style.css",
    "embed.js",
    "tokenizer.js",
    "ingest.js",
    "indexer.js",
    "store.js",
    "index-worker.js",
    "coi-sw.js",
    "bench.js",
    "bench-queries.json",
    "THIRD-PARTY-NOTICES.md",
    "sample-index",
    "pkg",
    "vendor",
]
#: Built or fetched by other scripts; missing ones are an error, not a warning.
REQUIRED = {
    "pkg": "python scripts/build_wasm.py",
    "vendor": "python scripts/vendor_ort.py",
}


def build(out: Path) -> int:
    for name, how in REQUIRED.items():
        if not (DEMO / name).is_dir():
            raise SystemExit(f"web-demo/{name}/ is missing; run: {how}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in INCLUDE:
        src = DEMO / name
        if src.is_dir():
            # wasm-bindgen writes a .gitignore into pkg/; a site has no use for it.
            shutil.copytree(src, out / name, ignore=shutil.ignore_patterns(".gitignore"))
        else:
            shutil.copy2(src, out / name)
    # Serve files as they are: no Jekyll processing on GitHub Pages.
    (out / ".nojekyll").touch()
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder")
    args = parser.parse_args()
    size = build(args.out)
    print(f"site in {args.out}: {size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
