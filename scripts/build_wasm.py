"""Build the notelight-wasm module into web-demo/pkg/ and report its size.

    rustup target add wasm32-unknown-unknown
    cargo install wasm-bindgen-cli --version <the wasm-bindgen version in Cargo.lock> --locked
    python scripts/build_wasm.py

Two steps, no wasm-pack: `cargo build` compiles the crate to a .wasm file,
then `wasm-bindgen` generates the JavaScript glue (`notelight_wasm.js`), the
TypeScript types and a processed `notelight_wasm_bg.wasm`. The CLI must be
the exact version of the `wasm-bindgen` crate, so the script checks it first.
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web-demo" / "pkg"
TARGET = "wasm32-unknown-unknown"
PROFILE = "wasm-release"


def locked_version(crate: str) -> str:
    """The version of `crate` that Cargo.lock pins."""
    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    match = re.search(rf'name = "{re.escape(crate)}"\nversion = "([^"]+)"', lock)
    if match is None:
        raise SystemExit(f"{crate} is not in Cargo.lock; run `cargo build` first")
    return match.group(1)


def check_cli(expected: str) -> None:
    exe = shutil.which("wasm-bindgen")
    install = f"cargo install wasm-bindgen-cli --version {expected} --locked"
    if exe is None:
        raise SystemExit(f"wasm-bindgen not found; install it with:\n  {install}")
    found = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
    if found.stdout.split()[-1] != expected:
        raise SystemExit(
            f"{found.stdout.strip()} does not match the wasm-bindgen crate ({expected}); "
            f"install the matching CLI with:\n  {install}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT, help="where to write the package")
    args = parser.parse_args()

    check_cli(locked_version("wasm-bindgen"))
    build = ["cargo", "build", "-p", "notelight-wasm", "--target", TARGET, "--profile", PROFILE]
    subprocess.run(build, cwd=ROOT, check=True)
    wasm = ROOT / "target" / TARGET / PROFILE / "notelight_wasm.wasm"
    bindgen = ["wasm-bindgen", "--target", "web", "--out-dir", str(args.out_dir), str(wasm)]
    subprocess.run(bindgen, cwd=ROOT, check=True)

    final = args.out_dir / "notelight_wasm_bg.wasm"
    raw = final.read_bytes()
    packed = gzip.compress(raw, compresslevel=9)
    name = final.relative_to(ROOT)
    print(f"{name}: {len(raw) / 1024:.0f} KiB, {len(packed) / 1024:.0f} KiB gzipped")


if __name__ == "__main__":
    main()
