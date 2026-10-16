"""Rebuild web-demo/sample-index/ from this repository's docs/ folder.

The browser demo and the notelight-wasm tests read this committed fixture: one
generation of a keyword-only index (`--embedder none`), built by the real CLI
so the bytes are exactly what `notelight index` writes. Re-run it when the
index schema changes (the notelight-wasm tests fail until you do) or when you
want the demo to search newer docs.

    python scripts/make_sample_index.py

The index is built in a temporary NOTELIGHT_HOME, so your own indexes are not
touched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "web-demo" / "sample-index"
FILES = ("manifest.json", "docs.bin", "bm25.bin")


def main() -> None:
    with tempfile.TemporaryDirectory() as home:
        env = {**os.environ, "NOTELIGHT_HOME": home}
        cmd = [sys.executable, "-m", "notelight.cli", "index", str(DOCS), "--embedder", "none"]
        subprocess.run(cmd, env=env, check=True)
        # There is exactly one index in the fresh home; CURRENT names its generation.
        (pointer,) = Path(home).glob("indexes/*/*/CURRENT")
        generation = pointer.parent / pointer.read_text(encoding="utf-8").strip()
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        for name in FILES[1:]:
            shutil.copyfile(generation / name, OUT / name)
        # The indexer records the absolute source folder; keep this machine's
        # paths out of the repository. The manifest's own size is not checked.
        manifest = json.loads((generation / FILES[0]).read_text(encoding="utf-8"))
        manifest["metadata"]["source"] = "docs"
        text = json.dumps(manifest, indent=2) + "\n"
        (OUT / FILES[0]).write_text(text, encoding="utf-8", newline="\n")
    total = sum((OUT / n).stat().st_size for n in FILES)
    print(f"wrote {OUT.relative_to(ROOT)} ({total / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
