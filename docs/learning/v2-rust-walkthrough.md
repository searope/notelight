# v2 spike: Rust walkthrough

The v2 spike compiles the search core to WebAssembly and calls it from a web
page. Almost no core code changed. The new Rust is the binding crate,
`crates/notelight-wasm`, and a few lines of Cargo configuration. This guide
explains the ideas behind them in C# and Python terms: what a `cdylib` is,
how `cfg(target_arch)` picks code per platform, what `#[wasm_bindgen]`
generates, and how values cross between JavaScript and Rust. It takes about
25 minutes.

**Suggested order:** `crates/notelight-wasm/Cargo.toml` →
`crates/notelight-wasm/src/lib.rs` → `web-demo/pkg/notelight_wasm.js` (after
`python scripts/build_wasm.py`; skim it, it is generated) → `web-demo/app.js`
→ `crates/notelight-core/src/index/storage.rs`.

## 1. A target is a (CPU, OS) pair, and wasm32 is a new CPU

`rustc` compiles for a *target triple*: `x86_64-pc-windows-msvc` on your PC,
`aarch64-apple-darwin` on an M-series Mac. `wasm32-unknown-unknown` reads as
"the wasm32 instruction set, no vendor, no operating system". It is like
asking .NET to compile for a CPU that has no Windows, no files, no threads and
no clock underneath it; only what the host (the browser) hands in through
imports.

```powershell
rustup target add wasm32-unknown-unknown
cargo build -p notelight-core --target wasm32-unknown-unknown
```

Two things follow from "no OS":

- `std` still exists, but the OS parts are stubs: `std::fs::read` compiles and
  returns an error at run time, `std::thread::spawn` panics. That is why the
  core compiled without changes: the filesystem code is still there, the
  browser just never calls it.
- `usize` is 32 bits. Every `as usize` on a value read from a file is a
  possible truncation on wasm32 that never happens on 64-bit Windows. The core
  already decodes offsets as `u64` and checks them before converting
  (milestone 1), which is exactly what makes it safe here. The CI job runs the
  core's unit tests on `wasm32-wasip1` (wasm32 *with* a minimal OS interface,
  WASI) to prove that.

## 2. `cfg(target_arch = "wasm32")`: conditional compilation

`storage.rs` has a variant that only exists on native targets:

```rust
pub enum Storage {
    Owned(Vec<u8>),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
}
```

and `Cargo.toml` only pulls memmap2 in for those targets:

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
memmap2 = "0.9"
```

This is C#'s `#if WINDOWS ... #endif`, with two differences that matter:

- The condition is evaluated by the compiler from the target, not from a
  symbol you define in the project file. `target_arch`, `target_os`,
  `target_pointer_width` and `windows`/`unix` are all built in.
- It applies to *items*: a whole variant, function, `impl`, `use` or `match`
  arm disappears, not just lines of text. Code that is compiled out is still
  parsed, so it must be syntactically valid.

There is also a value form, `cfg!(...)`, that evaluates to a `bool` while
compiling *both* branches:

```rust
if !mmap || len == 0 || cfg!(target_arch = "wasm32") {
    return std::fs::read(path).map(Self::Owned)...;
}
```

Use `#[cfg]` when the other branch would not compile on this target (it names
`memmap2`), and `cfg!` when both branches compile and you just want one of them
taken; the optimizer removes the dead one.

The same mechanism works for dev-dependencies. Proptest needs a clock and
processes, so the core lists it under
`[target.'cfg(not(target_arch = "wasm32"))'.dev-dependencies]`, and the wasm
tests only need the `--lib` unit tests, which do not use it.

## 3. `crate-type = ["cdylib", "rlib"]`: what the compiler produces

A Rust library normally compiles to an `rlib`: Rust-only metadata plus code,
which only another Rust crate can link. It is like a .NET assembly that only
.NET can load. `notelight-wasm/Cargo.toml` asks for two outputs:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

- **`cdylib`**, a "C dynamic library": a self-contained module with a C-style
  export table and no Rust metadata. On Windows that is a `.dll`, which is
  what `notelight-py` builds for Python to load (renamed `.pyd`). On wasm32 it
  is a `.wasm` module whose exports a JavaScript engine can call. Everything
  the crate depends on, the core and `std` included, is linked in, and the
  linker drops whatever no export reaches.
- **`rlib`**, so that Rust code can still use the crate: its own native unit
  tests, the `examples/latency.rs` benchmark, and `tests/node.rs`.

`notelight-py` only lists `cdylib`, because nothing in Rust links against it.

## 4. `#[wasm_bindgen]`: a code generator on both sides

A `.wasm` module can only import and export functions over numbers (`i32`,
`i64`, `f32`, `f64`). Strings, arrays and objects do not exist at that level.
`wasm-bindgen` closes the gap, in two halves that must agree, which is why
the CLI version has to match the crate version exactly:

1. The `#[wasm_bindgen]` attribute is a *procedural macro*: Rust code that runs
   at compile time and rewrites the item it is attached to, much like a C#
   source generator. On

   ```rust
   #[wasm_bindgen]
   impl Index {
       #[wasm_bindgen(js_name = searchBm25, unchecked_return_type = "Bm25Hit[]")]
       pub fn search_bm25(&self, text: &str, k: usize) -> Result<JsValue, JsError> { ... }
   }
   ```

   it emits a raw export, roughly
   `extern "C" fn index_searchBm25(this: u32, text_ptr: u32, text_len: u32, k: u32) -> ...`,
   that unpacks its numeric arguments and calls your method. It also embeds a
   description of the signature in the `.wasm` file.
2. The `wasm-bindgen` CLI reads those descriptions and writes
   `notelight_wasm.js`: a JavaScript `class Index` whose `searchBm25(text, k)`
   encodes `text` into WebAssembly memory, calls the raw export, and turns
   the result back into JavaScript values. It also writes `notelight_wasm.d.ts`
   with TypeScript types for all of it.

The page then sees an ordinary ES module:

```js
import init, { Index } from "./pkg/notelight_wasm.js";
await init();                                  // fetch + compile the .wasm
const index = new Index(manifest, docs, bm25, null);
index.searchBm25("lock file", 10);             // [{id: 12, score: 7.3}, ...]
```

The attribute arguments are the API's surface: `constructor` makes `new Index(...)`
call `Index::new`, `getter` turns `fn length(&self)` into a property
(`index.length`), and `js_name` renames `search_bm25` to the JavaScript
convention `searchBm25`.

## 5. How values cross the boundary

WebAssembly has its own linear memory, a single `ArrayBuffer` that Rust sees
as its whole address space. JavaScript can read and write it; Rust cannot see
JavaScript's heap at all. So every non-number either gets **copied** into or
out of that memory, or stays in JavaScript and is referred to by **handle**.

| Rust parameter or result | What JavaScript passes or gets | What happens |
|---|---|---|
| `u32`, `usize`, `f32`, `bool` | `number`, `boolean` | passed directly |
| `&str` | `string` | UTF-16 is encoded as UTF-8 into wasm memory; Rust borrows it for the call |
| `String` (result) | `string` | UTF-8 is decoded from wasm memory, then freed |
| `&[u8]`, `Vec<u8>` | `Uint8Array` | the bytes are copied into wasm memory |
| `Option<Vec<f32>>` | `Float32Array`, plain array, `null` or `undefined` | copied, or `None` |
| `JsValue` (result) | any JavaScript value | a handle into a table of JS objects |
| `Result<T, JsError>` | `T`, or a thrown `Error` | `Err` becomes an exception |
| `Index` (a `#[wasm_bindgen]` struct) | a JS object wrapping a pointer | the Rust value lives in wasm memory |

Some consequences you can see in the code:

- **Opening an index copies it.** `Index::new(manifest: &[u8], docs: Vec<u8>, ...)`
  receives copies of the `Uint8Array`s. There is no memory-mapping in a
  browser, and JavaScript's buffers are outside Rust's address space. That is
  the 70 ms for 104 MiB in the design notes. Taking `Vec<u8>` rather than
  `&[u8]` for the big files moves that copy straight into the reader, so it
  happens once; `&[u8]` would be a borrowed copy that `from_parts` would have
  to copy again into a `Vec` it can own.
- **Results are built as JavaScript objects.** `search_bm25` returns
  `JsValue`, made by `serde-wasm-bindgen` from ordinary `#[derive(Serialize)]`
  structs. It walks the struct and calls JavaScript to create each object,
  array and number. That is the C# equivalent of returning a DTO that a
  serializer maps to JSON, except the "JSON" is live JS objects, with no string
  in between. `#[serde(rename_all = "camelCase")]` gives JavaScript
  `bm25Rank`, and `Serializer::json_compatible()` makes `None` become `null`
  instead of `undefined`. The alternative, a `#[wasm_bindgen]` struct per
  result, would give each hit a handle into wasm memory that JavaScript must
  `free()`: easy to leak, and awkward to log or `JSON.stringify`.
- **Errors become exceptions.** `JsError::new(&e.to_string())` creates a real
  JavaScript `Error`. In C# terms, the `Result` is the `TryParse` style inside
  Rust and the exception style at the edge, which is what each side expects.
- **The `Index` object has a manual lifetime.** Its Rust value lives in wasm
  memory, and JavaScript holds a pointer to it. `index.free()` drops it, which
  runs Rust's `Drop` and frees the 100 MiB. wasm-bindgen also registers a
  `FinalizationRegistry`, the JavaScript cousin of a .NET finalizer, as a
  safety net; the demo calls `free()` itself before opening another index,
  for the same reason you would `Dispose()` rather than wait for the GC.

## 6. Code that runs in two worlds: the plain-Rust twins

Creating a `JsValue` calls into JavaScript. On a native target there is no
JavaScript, so those calls panic. A unit test that calls `search_bm25`
natively would therefore crash. The binding splits each method in two:

```rust
#[wasm_bindgen]
impl Index {
    pub fn search_bm25(&self, text: &str, k: usize) -> Result<JsValue, JsError> {
        to_js(&self.hits_bm25(text, k))          // JS conversion only
    }
}

impl Index {                                      // no #[wasm_bindgen]
    pub fn hits_bm25(&self, text: &str, k: usize) -> Vec<Bm25Hit> { ... }
}
```

A type can have several `impl` blocks, like a C# `partial class` split over
files. Only the first block is exported to JavaScript. The second holds the
plain-Rust twins that do the work, which the native tests and the latency
example call. `tests/node.rs` then tests the exported methods for real, as
WebAssembly under Node, and is compiled only for that target:

```rust
#![cfg(target_arch = "wasm32")]
```

An inner attribute (`#!`) at the top of a file applies to the whole file, so on
Windows `cargo test` sees an empty test file instead of one that does not
compile.

## 7. Profiles: tuning one build without touching the others

`Cargo.toml` defines a custom profile for the browser build:

```toml
[profile.wasm-release]
inherits = "release"
lto = "fat"
strip = true
```

A profile is a named set of compiler settings, like a build configuration
(`Debug`/`Release`) in a .csproj. `inherits` copies `release` and overrides two
settings: whole-program link-time optimization, so the linker can drop more of
`std`, and stripping symbol names out of the `.wasm`. The result is 245 KiB
(102 KiB gzipped) instead of 298 KiB. The Python wheels keep using `release`
unchanged. `scripts/build_wasm.py` selects it with
`cargo build --profile wasm-release`.

## Try it

```powershell
rustup target add wasm32-unknown-unknown wasm32-wasip1
cargo install wasm-bindgen-cli --version 0.2.129 --locked   # must match Cargo.lock
python scripts/build_wasm.py
python -m http.server 8000 --directory web-demo
# open http://localhost:8000 and click "Load the sample index"

cargo test --target wasm32-unknown-unknown -p notelight-wasm --test node
cargo test --target wasm32-wasip1 -p notelight-core --lib
```

Then, as an exercise: add a `#[wasm_bindgen(getter)] fn embedder` style getter
that returns the manifest's `created_by`, rebuild, and find it in the
generated `notelight_wasm.d.ts`.
