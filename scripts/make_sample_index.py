"""Rebuild web-demo/sample-index/ and its embedder parity fixture.

The browser demo and the notelight-wasm tests read this committed fixture: one
generation of this repository's docs/ indexed with the default embedder
(bge-small), built by the real CLI so the bytes are exactly what
`notelight index` writes. The same run embeds the queries in QUERIES with the
Python `OnnxEmbedder` and records each query vector and its top-10 hits in
web-demo/tests/fixtures/queries.json, which web-demo/tests/embed.test.mjs
compares with the browser embedder. Re-run it when the index schema changes
(the notelight-wasm tests fail until you do) or when you want the demo to
search newer docs.

    python scripts/make_sample_index.py
    python scripts/make_sample_index.py --hits-only

`--hits-only` keeps the committed index and only re-records the fixture's
hits, after a ranking change that leaves the index format alone.

The index is built in a temporary NOTELIGHT_HOME, so your own indexes are not
touched; the model comes from the usual model cache (downloaded if missing).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "web-demo" / "sample-index"
FIXTURE = ROOT / "web-demo" / "tests" / "fixtures" / "queries.json"
FILES = ("manifest.json", "docs.bin", "bm25.bin", "vectors.bin")
K = 10

# Questions about this repository's docs: some share words with the answer,
# some only meaning.
QUERIES = (
    "how are the model files verified",
    "why is there no approximate nearest neighbour index",
    "memory mapped files",
    "what happens when two processes write the same index",
    "reciprocal rank fusion",
    "how big is the WebAssembly module",
    "page numbers when citing a PDF",
    "does ternary quantization lose quality",
    "can the MCP server read any file on disk",
    "bonus for words in the right order",
    "running on Windows",
    "protecting the local web page from DNS rebinding",
    "how fast is a search",
    "what changes force a full re-embed",
)


def write_fixture(index_root: Path) -> None:
    """Embed QUERIES in Python and record each vector and its hybrid hits on
    the index under `index_root`."""
    from notelight import _core
    from notelight.embed import OnnxEmbedder

    embedder = OnnxEmbedder()
    reader = _core.IndexReader.open(index_root)
    queries = []
    for text in QUERIES:
        vector = embedder.embed_query(text).astype("<f4")
        hits = reader.search(text, vector, k=K)
        queries.append(
            {
                "text": text,
                "vector": base64.b64encode(vector.tobytes()).decode("ascii"),
                "hits": [h.id for h in hits],
            }
        )
    fixture = {
        "comment": "Written by scripts/make_sample_index.py: OnnxEmbedder query vectors "
        "(base64 little-endian float32) and hybrid top-k chunk ids on web-demo/sample-index.",
        "model_id": embedder.model_id,
        "k": K,
        "queries": queries,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8", newline="\n")
    assert np.isfinite(np.frombuffer(base64.b64decode(queries[0]["vector"]), "<f4")).all()


def rewrite_hits() -> None:
    """Re-record the fixture's hits on the committed sample index."""
    with tempfile.TemporaryDirectory() as root:
        # A one-generation index root around the committed files.
        generation = Path(root) / "gen-000001"
        shutil.copytree(OUT, generation)
        (Path(root) / "CURRENT").write_text(generation.name, encoding="utf-8")
        write_fixture(Path(root))
    print(f"wrote {FIXTURE.relative_to(ROOT)} ({FIXTURE.stat().st_size / 1024:.0f} KiB)")


def main() -> None:
    if sys.argv[1:] == ["--hits-only"]:
        rewrite_hits()
        return
    from notelight import paths

    # NOTELIGHT_HOME would also move the model cache; keep using the usual one.
    models = str(paths.model_cache_dir())
    with tempfile.TemporaryDirectory() as home:
        # notelight.paths reads these at call time, in this process and the CLI's.
        os.environ.update({"NOTELIGHT_HOME": home, "NOTELIGHT_MODEL_CACHE": models})
        env = dict(os.environ)
        cmd = [sys.executable, "-m", "notelight.cli", "index", str(DOCS)]
        subprocess.run(cmd, env=env, check=True)
        # There is exactly one index in the fresh home; CURRENT names its generation.
        (pointer,) = Path(home).glob("indexes/*/*/CURRENT")
        generation = pointer.parent / pointer.read_text(encoding="utf-8").strip()
        index_root = pointer.parent
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
        write_fixture(index_root)
    total = sum((OUT / n).stat().st_size for n in FILES)
    print(f"wrote {OUT.relative_to(ROOT)} ({total / 1024:.0f} KiB)")
    print(f"wrote {FIXTURE.relative_to(ROOT)} ({FIXTURE.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
