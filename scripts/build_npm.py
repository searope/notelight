"""Build the notelight-wasm npm package into dist/npm/ and check what it ships.

    python scripts/build_npm.py           # dist/npm/notelight-wasm/ + `npm pack --dry-run` check
    python scripts/build_npm.py --pack    # also writes dist/npm/notelight-wasm-<version>.tgz
    python scripts/build_npm.py --print-version

The package is the same `--target web` build the browser demo loads
(`scripts/build_wasm.py`), plus a Node entry (`crates/notelight-wasm/npm/node.js`),
the npm README, both licences and a package.json generated here. Its version is
Cargo's `[workspace.package]` version, so there is one version for the crates,
the Python package and the npm package. `npm pack --dry-run` must list exactly
the expected files, so nothing ships by accident.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_wasm import ROOT, build, report_size

NAME = "notelight-wasm"
SOURCES = ROOT / "crates" / "notelight-wasm" / "npm"
DIST = ROOT / "dist" / "npm"

# What the tarball contains; `npm pack --dry-run` must agree exactly.
GLUE = [
    "notelight_wasm.js",
    "notelight_wasm.d.ts",
    "notelight_wasm_bg.wasm",
    "notelight_wasm_bg.wasm.d.ts",
]
EXTRA = {
    "node.js": SOURCES / "node.js",
    "README.md": SOURCES / "README.md",
    "LICENSE-MIT": ROOT / "LICENSE-MIT",
    "LICENSE-APACHE": ROOT / "LICENSE-APACHE",
}
EXPECTED = sorted([*GLUE, *EXTRA, "package.json"])


def cargo_version() -> str:
    """The `[workspace.package]` version in the root Cargo.toml."""
    toml = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = re.search(r"^\[workspace\.package\]\n(.*?)(?=^\[)", toml, re.M | re.S)
    match = section and re.search(r'^version = "([^"]+)"$', section.group(1), re.M)
    if not match:
        raise SystemExit("no version in Cargo.toml [workspace.package]")
    return match.group(1)


def manifest(version: str) -> dict[str, object]:
    return {
        "name": NAME,
        "version": version,
        "description": (
            "notelight's Rust retrieval core as WebAssembly: open a notelight index "
            "and run BM25 or hybrid (BM25 + quantized vector) search, or build an "
            "index, in the browser or Node."
        ),
        "license": "MIT OR Apache-2.0",
        "author": "Sergey Nefedyev",
        "homepage": "https://searope.github.io/notelight/",
        # Trusted publishing checks this against the publishing repository.
        "repository": {
            "type": "git",
            "url": "git+https://github.com/searope/notelight.git",
            "directory": "crates/notelight-wasm",
        },
        "bugs": {"url": "https://github.com/searope/notelight/issues"},
        "keywords": [
            "search",
            "semantic-search",
            "hybrid-search",
            "bm25",
            "vector-search",
            "embeddings",
            "rag",
            "notes",
            "webassembly",
            "wasm",
            "rust",
        ],
        "type": "module",
        "main": "./notelight_wasm.js",
        "types": "./notelight_wasm.d.ts",
        "exports": {
            ".": {
                "types": "./notelight_wasm.d.ts",
                "node": "./node.js",
                "default": "./notelight_wasm.js",
            },
            "./notelight_wasm_bg.wasm": "./notelight_wasm_bg.wasm",
            "./package.json": "./package.json",
        },
        "files": sorted([*GLUE, "node.js", "LICENSE-MIT", "LICENSE-APACHE"]),
        # node.js instantiates the module on import; nothing else has side effects.
        "sideEffects": ["./node.js"],
        "engines": {"node": ">=22"},
    }


def npm(*args: str, cwd: Path) -> str:
    exe = shutil.which("npm")
    if exe is None:
        raise SystemExit("npm not found; install Node.js")
    return subprocess.run([exe, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def assemble(out: Path, version: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report_size(build(out))
    for path in out.iterdir():
        if path.name not in GLUE:
            path.unlink()  # wasm-bindgen's .gitignore, which would hide everything from npm
    for name, source in EXTRA.items():
        shutil.copyfile(source, out / name)
    text = json.dumps(manifest(version), indent=2) + "\n"
    (out / "package.json").write_text(text, encoding="utf-8", newline="\n")


def check_contents(out: Path) -> None:
    [report] = json.loads(npm("pack", "--dry-run", "--json", cwd=out))
    files = sorted(f["path"] for f in report["files"])
    if files != EXPECTED:
        raise SystemExit(f"npm pack would ship {files}, expected {EXPECTED}")
    print(
        f"{report['id']}: {len(files)} files, {report['size'] / 1024:.0f} KiB packed, "
        f"{report['unpackedSize'] / 1024:.0f} KiB unpacked"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DIST, help="where to write the package")
    parser.add_argument("--pack", action="store_true", help="also write the .tgz to --out-dir")
    parser.add_argument("--print-version", action="store_true", help="print the version and exit")
    args = parser.parse_args()

    version = cargo_version()
    if args.print_version:
        print(version)
        return
    dist = args.out_dir.resolve()
    out = dist / NAME
    assemble(out, version)
    check_contents(out)
    if args.pack:
        [report] = json.loads(npm("pack", "--json", "--pack-destination", str(dist), cwd=out))
        print(dist / report["filename"])


if __name__ == "__main__":
    main()
