# Milestone 2: Rust walkthrough

Milestone 2 is mostly Python, so the Rust in it is small: the PyO3 binding
crate in `crates/notelight-py/src/`, plus one fix in the core prompted by a new
clippy lint. This guide covers the ideas that crate uses, mapped to C# and
Python. It takes about 45 minutes.

**Suggested order:** `lib.rs` → `errors.rs` → `chunking.rs` → `index.rs`, then
`python/notelight/_core.pyi` next to `index.rs`.

## 1. A Rust struct as a Python class (`#[pyclass]`)

```rust
#[pyclass(module = "notelight._core", name = "ChunkRecord", frozen, get_all, eq, skip_from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyChunkRecord {
    path: String,
    heading_path: Vec<String>,
    ...
}
```

`#[pyclass]` is an *attribute macro*: it generates code at compile time, much
like a C# source generator. The options are:

| Option | Effect in Python |
|---|---|
| `name = "ChunkRecord"` | the class name Python sees (the Rust name keeps a `Py` prefix) |
| `frozen` | instances are immutable, like a `@dataclass(frozen=True)` |
| `get_all` | a read-only property for every field |
| `eq` | `__eq__` from Rust's `PartialEq` (the derived, field-by-field one) |
| `skip_from_py_object` | do not generate "copy a Python object into a Rust value"; methods borrow `&PyChunkRecord` instead |

`frozen` has a second benefit: PyO3 can hand out `&self` without runtime
borrow tracking, because nothing can mutate the object.

The `Py` prefix keeps the binding types separate from the core's types. Each
has a `From` impl that converts one into the other, which is Rust's version of
an implicit conversion operator, except you call it explicitly
(`PyChunkRecord::from(record)` or `record.into()`).

## 2. Methods and Python signatures (`#[pymethods]`)

```rust
#[pymethods]
impl PyIndexReader {
    #[staticmethod]
    #[pyo3(signature = (root, mmap = true))]
    fn open(py: Python<'_>, root: PathBuf, mmap: bool) -> PyResult<Self> { ... }

    #[pyo3(signature = (text, vector = None, k = 10, depth = 50, candidates = 200, rrf_k = 60.0))]
    fn search(&self, py: Python<'_>, text: &str, vector: Option<PyReadonlyArray1<'_, f32>>, ...)
```

- `signature = (...)` gives Python defaults and keyword arguments. Rust has
  neither, so PyO3 generates the argument parsing.
- Argument types do the conversion: `PathBuf` accepts `str` or any
  `os.PathLike`, `Option<T>` accepts `None`, `Vec<String>` accepts a list of
  str, and `PyReadonlyArray1<f32>` accepts a 1-D float32 NumPy array **without
  copying**. A float64 array is a `TypeError`, which is why the indexer always
  passes float32.
- `PyResult<T>` is `Result<T, PyErr>`. Returning `Err` raises the exception
  in Python.
- `py: Python<'_>` is a token that proves the GIL is held. You can't create
  one yourself, so a function that takes one can only be called while holding
  the GIL.

## 3. Releasing the GIL (`py.detach`)

```rust
let hits = py
    .detach(|| self.inner.search(text, vector, &opts))
    .map_err(to_py)?;
```

`detach` releases the GIL, runs the closure, and takes the GIL back, like
`Py_BEGIN_ALLOW_THREADS` in C. Other Python threads run in the meantime, so an
MCP server (milestone 4) can answer several queries in parallel.

The compiler checks that this is safe. The closure must be `Send` (Rust's
marker for "may cross threads"), and Python object handles are not `Send`, so
touching one inside `detach` is a compile error. In C# terms, it's as if the
compiler rejected any `async` lambda that captured a UI control.

`text: &str` and the NumPy slice *are* allowed inside: they are plain borrowed
bytes, and PyO3 keeps the Python objects alive until the call returns.

## 4. Single-use objects: `Option::take`

The core's `IndexWriter::commit(self)` takes `self` **by value**: committing
consumes the writer, so the compiler stops you from using it afterwards. Python
can't express that, so the binding stores `Option<IndexWriter>` and moves it
out on commit:

```rust
fn commit(&mut self, py: Python<'_>, root: PathBuf) -> PyResult<PyCommitInfo> {
    let writer = self.inner.take()  // leaves None behind
        .ok_or_else(|| PyRuntimeError::new_err("this IndexWriter was already committed"))?;
    let info = py.detach(|| writer.commit(&root)).map_err(to_py)?;
    ...
}
```

`Option::take` is "swap with `None` and give me the old value". A compile-time
guarantee in Rust becomes a clear runtime error in Python.

## 5. Mapping errors (`errors.rs`)

```rust
create_exception!(_core, IndexFormatError, NotelightError, "...");
create_exception!(_core, SchemaError, IndexFormatError, "...");

pub fn to_py(err: Error) -> PyErr {
    let msg = err.to_string();
    match err {
        Error::Io { .. } => PyOSError::new_err(msg),
        Error::Format(_) => IndexFormatError::new_err(msg),
        Error::Schema { .. } => SchemaError::new_err(msg),
        Error::Dimension { .. } | Error::Invalid(_) => PyValueError::new_err(msg),
    }
}
```

- `create_exception!` defines a Python exception class with a base class, like
  `class SchemaError(IndexFormatError): ...`. Python code can subclass these
  (`notelight/errors.py` adds `IndexBusyError` and others under
  `NotelightError`).
- The `match` is **exhaustive**. If the core adds a variant, this function
  stops compiling until someone decides how Python should see it. A C# `switch`
  on a sealed hierarchy only warns.
- `{ .. }` ignores a variant's fields, and `|` matches several variants in
  one arm.

`.map_err(to_py)?` turns a core `Result` into a `PyResult` and returns early
on error. It's the line you'll see at almost every boundary.

## 6. Don't let Rust panics reach Python

The core's `chunk_text` asserts `max_chars > 0`, because zero is a programming
error inside Rust. From Python it's just a bad argument, and a panic would
surface as `pyo3_runtime.PanicException`, so the binding checks first:

```rust
if max_chars == 0 {
    return Err(PyValueError::new_err("max_chars must be positive"));
}
```

Rule of thumb: validate at the boundary for anything a Python caller can pass,
and keep the core's asserts for its own invariants.

## 7. Passing a dict through JSON

`IndexWriter(dim, embedder, metadata)` stores `metadata` in the manifest as
`serde_json::Value`. Rather than walking Python dicts by hand, the binding
calls Python's own `json.dumps`, then parses the string with `serde_json`:

```rust
let text: String = py.import("json")?.call_method1("dumps", (dict,))?.extract()?;
serde_json::from_str(&text)
```

Calling Python from Rust is ordinary method calls on `Bound<PyAny>` values.
`(dict,)` is a one-element tuple, the same trailing-comma rule as Python.
`.extract()` converts back to a Rust type. The manifest getter does the
reverse with `json.loads`.

## 8. Fixed-size chunks (`as_chunks`, in `quant.rs`)

Clippy 1.99 added `chunks_exact_to_as_chunks`, which fired on the Hamming
distance loop. The fix shows the difference between slices and arrays:

```rust
// before: each chunk is a &[u8] slice; its length is only known at runtime
let mut words_a = a.chunks_exact(8);
... u64::from_le_bytes(chunk.try_into().expect("chunk of 8 bytes")) ...

// after: each chunk is a &[u8; 8] array; the length is part of the type
let (words_a, rest_a) = a.as_chunks::<8>();
... u64::from_le_bytes(*x) ...
```

`[u8; 8]` is a fixed-size array type (like `fixed byte[8]` in C#), so
`from_le_bytes` accepts it directly and the `expect` that could panic is gone.
`::<8>` is the turbofish syntax for passing a generic argument explicitly; here
it's a *const generic*, a number rather than a type. A criterion A/B run showed
no slowdown (about 11% faster at 10k vectors).

## 9. Testing a binding

The binding has no Rust tests of its own. It's tested from Python, which is
how it's used: `tests/test_index_bindings.py` covers the round trip, every
error mapping (including a manifest edited to an unknown schema version), and
the single-use writer. The type stubs in `python/notelight/_core.pyi` are
checked by `mypy --strict` wherever the package uses them, so a stub that
disagrees with real use fails CI.

## Exercises

1. In `dict_to_json_map`, move the `json.dumps` call inside a
   `py.detach(|| ...)` closure. What does the compiler say, and which trait is
   missing?
2. Add a `Error::NotFound` variant to the core without updating `to_py`. Read
   the error. Then map it to `FileNotFoundError`.
3. Time `IndexReader.search` from two Python threads at once, with and without
   `py.detach`. What changes?
