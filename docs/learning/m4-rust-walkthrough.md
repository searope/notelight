# Milestone 4: Rust walkthrough

Milestone 4 is the MCP server (`python/notelight/mcp_server.py`), and **no
Rust changed**. The server is the first long-lived process that uses the core,
though, and it relies on three properties of the existing Rust code that a
one-shot CLI never needed: one reader shared by several threads, searches that
run without the GIL, and readers that keep working while a new generation is
published. This guide explains where each property comes from, in C# and
Python terms. It takes about 20 minutes.

**Suggested order:** `crates/notelight-py/src/index.rs` (`PyIndexReader`) →
`crates/notelight-core/src/index/mod.rs` (`CURRENT`, `current_generation_dir`)
→ `Searcher.is_live` and `NotesService.searcher` in Python.

## 1. `frozen`: a Python object that is safe to share between threads

The MCP SDK runs each synchronous tool in a worker thread
(`anyio.to_thread.run_sync`), so two `search` calls can use the same
`IndexReader` at the same time. The binding declares it like this:

```rust
#[pyclass(module = "notelight._core", name = "IndexReader", frozen)]
pub struct PyIndexReader {
    inner: index::IndexReader,
}
```

Without `frozen`, PyO3 guards every Python object with a runtime borrow flag,
like a `RefCell`: a method taking `&mut self` marks the object as mutably
borrowed, and a second thread calling in at that moment gets
`RuntimeError: Already borrowed`. `frozen` promises that no method ever takes
`&mut self`. PyO3 then drops the flag and requires the struct to be `Sync`, and
the compiler checks that promise: if `IndexReader` held a `Cell` or an
`Rc`, the `#[pyclass]` would not compile.

In C# terms, `frozen` is an immutable class whose thread safety the compiler
verifies, instead of a comment saying "this type is thread-safe". `Sync` means
"`&T` may be used from several threads at once", which for `IndexReader` holds
because everything it owns is read-only after `open`: the manifest, and the
memory-mapped files behind `Storage`.

`IndexWriter`, by contrast, is not `frozen`: `add` takes `&mut self`. That is
fine, because only the indexer uses it, under the writer lock.

## 2. `py.detach`: letting other threads run during a search

```rust
fn search(&self, py: Python<'_>, text: &str, vector: Option<PyReadonlyArray1<'_, f32>>, ...)
    -> PyResult<Vec<PyHit>>
{
    let vector = vector.as_ref().map(|v| v.as_slice()).transpose()?;
    ...
    let hits = py
        .detach(|| self.inner.search(text, vector, &opts))
        .map_err(to_py)?;
```

`py.detach(f)` releases the GIL, runs `f`, and takes the GIL back (it was
called `allow_threads` in older PyO3). While the closure runs, other Python
threads continue: another tool call can embed its query or read a file while
this one scores BM25 postings. In CPython C-API terms it is
`Py_BEGIN_ALLOW_THREADS` … `Py_END_ALLOW_THREADS`; in C#, the nearest picture is
leaving a `lock` block before doing slow work.

The closure must not touch Python objects, and Rust enforces it: `detach`
requires the closure to be `Ungil` (on stable Rust that means `Send`), and
the `Python<'_>` token, which every Python API call needs, is not `Send`, so
it cannot be moved into the closure. So the arguments are turned into plain
Rust views *before* the call: `&str` for the text, and `&[f32]` for the
vector. `as_slice` does not copy; it borrows the NumPy array's buffer, and the
`PyReadonlyArray1` wrapper marks the array as borrowed so no other Rust code
can write to it meanwhile. This is what "the binding layer stays thin" buys: all
the heavy work is plain Rust on plain data, so all of it can run unlocked.

The Python side adds one lock of its own. `OnnxEmbedder` loads its ONNX
session on first use, which is not safe to do from two threads at once, so
`NotesService` wraps each embedder in `_LockedEmbedder`. Embedding a query
takes milliseconds; the search after it stays parallel.

## 3. Generations: why an open reader never breaks, and how to notice it is old

`notelight index` never changes files a reader has open. It writes a complete
new `gen-NNNNNN/` directory and then renames a new `CURRENT` file over the old
one (see `docs/index-format.md`, "Commit protocol"). A reader resolves
`CURRENT` once, in `IndexReader::open`, and maps that generation's files:

```rust
pub fn open(root: &Path, opts: OpenOptions) -> Result<Self> {
    Self::open_generation(&current_generation_dir(root)?, opts)
}
```

So a server started before a re-index keeps answering, from the old
generation, until it reopens. That is snapshot isolation, as in a database:
each reader sees one consistent version, never a half-written one.

To pick up new notes, the server needs to notice that `CURRENT` moved on.
`IndexReader.dir` is the generation directory the reader opened, so
`Searcher.is_live()` compares its name with the contents of `CURRENT`, one
small file read per call. When they differ, `NotesService.searcher` opens a
new `Searcher` and drops the old one.

Dropping matters on Windows. A memory-mapped file cannot be deleted while it
is mapped, so the commit that superseded our generation could not remove it
(the collect step ignores that failure). Once the last Python reference to the
old reader is gone, Rust's `Drop` unmaps the files (the `Drop` of `memmap2::Mmap`
calls `UnmapViewOfFile` on Windows, `munmap` elsewhere), and the next commit deletes the directory. This is
deterministic, like `using`/`Dispose` in C#, but without writing anything: a
value is dropped when its owner goes out of scope, and here the owner is a
Python object whose reference count hits zero.

## Summary

| Need of a long-lived server | Where it comes from |
|---|---|
| several threads share one reader | `#[pyclass(frozen)]` + `IndexReader: Sync` (checked by the compiler) |
| searches run in parallel | `py.detach` around the Rust search (GIL released) |
| re-indexing never breaks a running search | immutable generations + atomic rename of `CURRENT` |
| the server sees new notes | `Searcher.is_live()` compares `IndexReader.dir` with `CURRENT` |
| old generations get cleaned up on Windows | dropping the old reader unmaps its files (`Drop`) |
