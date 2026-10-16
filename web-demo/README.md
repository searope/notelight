# notelight in the browser (v2 spike)

A static page that loads a notelight index into WebAssembly and searches it
with the Rust core, entirely client-side. No framework, bundler or CDN.
Design and measurements: `docs/design-notes.md`, "v2 spike: the core in the
browser".

## Run it (PowerShell)

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked   # the version in Cargo.lock
python scripts/build_wasm.py                                # writes web-demo/pkg/
python -m http.server 8000 --bind 127.0.0.1 --directory web-demo
```

Open <http://127.0.0.1:8000> and click **Load the sample index**, or
**Open an index folder…** and pick one of your own: the folder that
`notelight status` prints as `index:`, or a `gen-NNNNNN` folder inside it. The
files are read by the page and never uploaded. An index with embeddings opens
too, but this spike cannot embed queries in the browser, so it ranks by BM25
and phrases only.

A module script cannot load from `file://`, hence the local server.

## Files

| File | What it is |
|---|---|
| `index.html`, `app.js`, `style.css` | the page |
| `bench.js` | the latency benchmark behind the **Benchmark** button |
| `bench-node.mjs` | the same benchmark under Node |
| `bench-queries.json` | queries for the sample index |
| `sample-index/` | this repository's `docs/`, indexed keyword-only; rebuild with `python scripts/make_sample_index.py` |
| `pkg/` | built by `scripts/build_wasm.py`, not committed |

## Measuring

```powershell
cargo run --release -p notelight-wasm --example latency -- web-demo\sample-index web-demo\bench-queries.json 200
node web-demo\bench-node.mjs web-demo\sample-index web-demo\bench-queries.json 200
```

In the page, **Benchmark** runs the same queries. For another index, serve a
folder that contains both the demo and the index, and pass
`?index=<url of a generation folder>&bench=<url of a queries file>&reps=5`.
A queries file is `{"queries": [{"text": "...", "vector": [...] | null}]}`;
vectors enable the hybrid text + vector row.
