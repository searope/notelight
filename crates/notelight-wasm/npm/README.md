# notelight-wasm

The retrieval core of [notelight](https://github.com/searope/notelight)
(private semantic search over your own files), compiled from Rust to
WebAssembly. It opens a notelight index and searches it with BM25, or with
hybrid search (BM25 plus quantized vectors, fused with Reciprocal Rank Fusion,
with exact phrases ranked first), and it can chunk documents and build an
index in memory. It works in browsers, bundlers and Node. The .wasm is about
320 KiB (about 130 KiB gzipped), with no dependencies.

**[Live demo](https://searope.github.io/notelight/)**: this package plus a
query embedder, indexing and searching a folder of notes entirely in the
browser.

> **Pre-1.0.** The API may change in a minor version; see
> [API stability](#api-stability). The index format is versioned, and an
> index from another schema version is refused with an error.

## What it does and does not include

- **Included:** opening an index (`Index`), BM25 search (`searchBm25`),
  hybrid search with a query vector you supply (`search`), stored chunks
  (`chunk`), the chunker the Python indexer uses (`chunkText`), and an
  in-memory index writer (`IndexWriter`). The code is the same Rust core the
  Python package runs.
- **Not included:** an embedding model. Hybrid search needs a query vector
  from the model the index was built with (`index.embedder`). The demo's
  [`embed.js`](https://github.com/searope/notelight/blob/main/web-demo/embed.js)
  embeds queries in the browser with bge-small through onnxruntime-web; it is
  not part of this package, because it brings a 127 MiB model and an ONNX
  runtime that many users of a search core will not want. Without a vector,
  `search` ranks by BM25 and phrases alone.
- **Not included:** reading folders, file storage or the incremental indexer
  (the demo's `ingest.js`, `indexer.js` and `store.js`).

## Install

```bash
npm install notelight-wasm
```

## Initialise

The package is ES modules only. Call the default export once before using
anything else:

```js
import init, { Index } from "notelight-wasm";
await init();
```

- **Browsers and bundlers** (Vite, webpack 5, Rollup, esbuild with a loader
  for `.wasm`): `init()` fetches `notelight_wasm_bg.wasm` from next to the JS
  file (`new URL(..., import.meta.url)`), which bundlers turn into an asset
  URL. The server must send `.wasm` as `application/wasm`. To serve the file
  from somewhere else, pass it: `await init({ module_or_path: url })`. Its
  path inside the package is exported as `notelight-wasm/notelight_wasm_bg.wasm`.
- **Node 22+** (and Deno and Bun, which read the `node` export condition):
  the module is loaded from disk when you import the package, so `await init()`
  is a no-op there and can stay in shared code.
- `initSync({ module })` takes the .wasm bytes or a compiled
  `WebAssembly.Module`, for workers and other places that cannot await.

## Search an index built by the Python CLI

`notelight index <folder>` writes generations of an index, and
`notelight status` prints where (`index:`). The live generation is the
`gen-NNNNNN` folder named in the `CURRENT` file. Pass its files to `Index`:

```js
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import init, { Index } from "notelight-wasm";

await init();
const root = process.argv[2]; // the `index:` folder that `notelight status` prints
const gen = join(root, (await readFile(join(root, "CURRENT"), "utf8")).trim());
const read = (name) => readFile(join(gen, name));
const index = new Index(
  await read("manifest.json"),
  await read("docs.bin"),
  await read("bm25.bin"),
  await read("vectors.bin").catch(() => null), // absent for a keyword-only index
);

// Keyword search: BM25 alone.
for (const { id, score } of index.searchBm25("lock file", 5)) {
  const chunk = index.chunk(id);
  console.log(score.toFixed(2), `${chunk.path}:${chunk.lineStart}`, chunk.headingPath.join(" > "));
}

// Hybrid search: BM25 + vectors. The query vector must come from the model
// the index was built with (index.embedder, index.dim).
console.log(index.embedder, index.dim); // "BAAI/bge-small-en-v1.5@5c38ec7c405e" 384
const vector = await embedQuery("how do I release a hotfix?"); // your embedder, a Float32Array
for (const hit of index.search("how do I release a hotfix?", vector, 10)) {
  console.log(hit.score, hit.bm25Rank, hit.vectorRank, index.chunk(hit.id).path);
}
index.free(); // or let the FinalizationRegistry release it
```

In a browser, get the same four files with `fetch(...).then((r) => r.arrayBuffer())`
and wrap them in `new Uint8Array(...)`, or read them from a directory picker
or IndexedDB.

For an index built with the CLI's default model (bge-small-en-v1.5),
`embedQuery` must match the Python embedder: prefix the text with
`"Represent this sentence for searching relevant passages: "`, take the CLS
token of the last hidden state and L2-normalize it. `web-demo/embed.js` does
exactly that and is tested in CI against the Python embedder's vectors; its
results match the CLI's top 10. A vector of the wrong length throws; a vector
from another model of the same length gives poor results without an error,
so check `index.embedder`.

## Build an index in memory

```js
import init, { chunkText, Index, IndexWriter } from "notelight-wasm";
await init();

// dim 0 and no embedder: a keyword-only index. With embeddings, pass the
// dimension and the model id, and a Float32Array per chunk to `add`.
const writer = new IndexWriter(0, null, JSON.stringify({ chunker: { max_chars: 1200 } }));
const text = "# Release\n\n## Hotfixes\n\nTag the commit, build the wheels, publish.\n";
for (const c of chunkText(text, 1200, true)) { // true: Markdown-aware
  writer.add("runbooks/release.md", c.headingPath, c.lineStart, c.lineEnd, c.text, null);
}
const gen = writer.build(1); // consumes the writer
const index = new Index(gen.manifest, gen.docs, gen.bm25, gen.vectors ?? null);
console.log(index.searchBm25("hotfix", 3));
```

`gen.manifest`, `gen.docs`, `gen.bm25` and `gen.vectors` are the bytes of
`manifest.json`, `docs.bin`, `bm25.bin` and `vectors.bin`, in the
[documented format](https://github.com/searope/notelight/blob/main/docs/index-format.md)
the CLI writes, so you can store them and reopen them later. `chunkText`
splits text exactly as `notelight index` does: by Markdown headings for `.md`
files (`markdown: true`), by blank-line paragraphs otherwise. Text must have
`\n` line endings.

## API

| | |
|---|---|
| `new Index(manifest, docs, bm25, vectors?)` | open one generation from its file bytes; throws on invalid or mismatched files |
| `index.searchBm25(text, k)` | `{ id, score }[]` |
| `index.search(text, vector \| null, k)` | `{ id, score, bm25Rank, vectorRank }[]`; ranks are 1-based or `null` |
| `index.chunk(id)` | `{ path, headingPath, lineStart, lineEnd, text }` |
| `index.length`, `dim`, `embedder`, `hasVectors`, `generation` | from the manifest |
| `chunkText(text, maxChars, markdown)` | `{ headingPath, lineStart, lineEnd, text }[]` |
| `new IndexWriter(dim, embedder?, metadataJson?)` | `.add(path, headingPath, lineStart, lineEnd, text, vector?)`, `.length`, `.build(generation)` |
| `version()` | the core's version |

TypeScript declarations are included (`notelight_wasm.d.ts`). Objects that
wrap Rust memory (`Index`, `IndexWriter`, `Generation`) have `free()` and
`[Symbol.dispose]()`, so `using index = new Index(...)` works; the
declarations therefore need TypeScript 5.2+ with `"esnext.disposable"` (or
`"esnext"`) in `lib`, or `skipLibCheck`.

## API stability

Before 1.0, a minor version (0.x → 0.y) may change this API: function and
argument names (the TypeScript argument names are still snake_case), the
shape of results, and the index schema version, which makes older indexes
fail to open until they are rebuilt with a matching `notelight index`. Patch
versions do not change the API. The package version is the same as the
Python package's, and an index opens only in a package that reads its
schema version; the
[changelog](https://github.com/searope/notelight/blob/main/CHANGELOG.md)
lists every change.

## Links

- [Live demo](https://searope.github.io/notelight/) and its
  [source](https://github.com/searope/notelight/tree/main/web-demo)
- [notelight on PyPI](https://pypi.org/project/notelight/): the CLI, the
  indexer with its embedding model, an MCP server for AI agents
- [How it works](https://github.com/searope/notelight/blob/main/docs/how-it-works.md)
  and the [index format](https://github.com/searope/notelight/blob/main/docs/index-format.md)

## License

MIT OR Apache-2.0, at your option.
