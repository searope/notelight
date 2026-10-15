//! `IndexWriter`, `IndexReader` and the record types they exchange.

use std::collections::BTreeMap;
use std::path::PathBuf;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use notelight_core::index::{self, ChunkRecord, OpenOptions, SearchOpts, WriterOptions};
use notelight_core::vectors::VectorMode;

use crate::errors::to_py;

/// One stored chunk: where it came from and its text.
#[pyclass(
    module = "notelight._core",
    name = "ChunkRecord",
    frozen,
    get_all,
    eq,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyChunkRecord {
    path: String,
    heading_path: Vec<String>,
    line_start: u32,
    line_end: u32,
    text: String,
}

#[pymethods]
impl PyChunkRecord {
    #[new]
    #[pyo3(signature = (path, heading_path, line_start, line_end, text))]
    fn new(
        path: String,
        heading_path: Vec<String>,
        line_start: u32,
        line_end: u32,
        text: String,
    ) -> Self {
        Self {
            path,
            heading_path,
            line_start,
            line_end,
            text,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkRecord(path={:?}, lines={}-{}, heading_path={:?})",
            self.path, self.line_start, self.line_end, self.heading_path
        )
    }
}

impl PyChunkRecord {
    fn to_core(&self) -> ChunkRecord {
        ChunkRecord {
            path: self.path.clone(),
            heading_path: self.heading_path.clone(),
            line_start: self.line_start,
            line_end: self.line_end,
            text: self.text.clone(),
        }
    }
}

impl From<ChunkRecord> for PyChunkRecord {
    fn from(r: ChunkRecord) -> Self {
        Self {
            path: r.path,
            heading_path: r.heading_path,
            line_start: r.line_start,
            line_end: r.line_end,
            text: r.text,
        }
    }
}

/// One hybrid search result. Ranks are 1-based, `None` if the chunk was not
/// in that retriever's list.
#[pyclass(
    module = "notelight._core",
    name = "Hit",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy)]
pub struct PyHit {
    id: u32,
    score: f32,
    bm25_rank: Option<u32>,
    vector_rank: Option<u32>,
}

#[pymethods]
impl PyHit {
    fn __repr__(&self) -> String {
        format!(
            "Hit(id={}, score={:.5}, bm25_rank={}, vector_rank={})",
            self.id,
            self.score,
            py_rank(self.bm25_rank),
            py_rank(self.vector_rank)
        )
    }
}

/// Format an optional rank the way Python prints it.
fn py_rank(rank: Option<u32>) -> String {
    rank.map_or_else(|| "None".to_owned(), |r| r.to_string())
}

/// Result of a successful commit.
#[pyclass(
    module = "notelight._core",
    name = "CommitInfo",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyCommitInfo {
    generation: u64,
    dir: PathBuf,
    n_chunks: u32,
}

#[pymethods]
impl PyCommitInfo {
    fn __repr__(&self) -> String {
        format!(
            "CommitInfo(generation={}, n_chunks={}, dir='{}')",
            self.generation,
            self.n_chunks,
            self.dir.display()
        )
    }
}

/// Collects chunks in memory, then writes them as one index generation.
///
/// A writer is single-use: after `commit` it is empty and further calls raise
/// `RuntimeError`.
#[pyclass(module = "notelight._core", name = "IndexWriter")]
pub struct PyIndexWriter {
    inner: Option<index::IndexWriter>,
}

impl PyIndexWriter {
    fn writer(&mut self) -> PyResult<&mut index::IndexWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("this IndexWriter was already committed"))
    }
}

#[pymethods]
impl PyIndexWriter {
    /// `dim = 0` builds a keyword-only index. `metadata` must be
    /// JSON-serializable; it is stored in the manifest as is.
    #[new]
    #[pyo3(signature = (dim, embedder = None, metadata = None))]
    fn new(
        py: Python<'_>,
        dim: usize,
        embedder: Option<String>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let metadata = match metadata {
            Some(dict) => dict_to_json_map(py, dict)?,
            None => BTreeMap::new(),
        };
        Ok(Self {
            inner: Some(index::IndexWriter::new(WriterOptions {
                dim,
                embedder,
                metadata,
                ..WriterOptions::default()
            })),
        })
    }

    /// Add one chunk and return its id. `embedding` (float32, shape `(dim,)`)
    /// is required unless the index is keyword-only.
    #[pyo3(signature = (record, embedding = None))]
    fn add(
        &mut self,
        record: &PyChunkRecord,
        embedding: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<u32> {
        let embedding = embedding.as_ref().map(|e| e.as_slice()).transpose()?;
        self.writer()?
            .add(&record.to_core(), embedding)
            .map_err(to_py)
    }

    /// Write the index under `root` as a new generation and make it live.
    /// Runs without holding the GIL.
    fn commit(&mut self, py: Python<'_>, root: PathBuf) -> PyResult<PyCommitInfo> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("this IndexWriter was already committed"))?;
        let info = py.detach(|| writer.commit(&root)).map_err(to_py)?;
        Ok(PyCommitInfo {
            generation: info.generation,
            dir: info.dir,
            n_chunks: info.n_chunks,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.as_ref().map_or(0, index::IndexWriter::len)
    }
}

/// A read-only view of the live index generation under a root directory.
///
/// The files are memory-mapped by default. Re-open to see newer commits.
#[pyclass(module = "notelight._core", name = "IndexReader", frozen)]
pub struct PyIndexReader {
    inner: index::IndexReader,
}

#[pymethods]
impl PyIndexReader {
    /// Open the live generation under `root`.
    #[staticmethod]
    #[pyo3(signature = (root, mmap = true))]
    fn open(py: Python<'_>, root: PathBuf, mmap: bool) -> PyResult<Self> {
        let inner = py
            .detach(|| index::IndexReader::open(&root, OpenOptions { mmap }))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Hybrid search: BM25 over `text`, plus vector search when `vector`
    /// (float32, shape `(dim,)`) is given, fused with reciprocal rank fusion.
    #[pyo3(signature = (text, vector = None, k = 10, depth = 50, candidates = 200, rrf_k = 60.0))]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the Python keyword arguments"
    )]
    fn search(
        &self,
        py: Python<'_>,
        text: &str,
        vector: Option<PyReadonlyArray1<'_, f32>>,
        k: usize,
        depth: usize,
        candidates: usize,
        rrf_k: f32,
    ) -> PyResult<Vec<PyHit>> {
        let vector = vector.as_ref().map(|v| v.as_slice()).transpose()?;
        let opts = SearchOpts {
            k,
            depth,
            candidates,
            rrf_k,
        };
        let hits = py
            .detach(|| self.inner.search(text, vector, &opts))
            .map_err(to_py)?;
        Ok(hits
            .into_iter()
            .map(|h| PyHit {
                id: h.id,
                score: h.score,
                bm25_rank: h.bm25_rank,
                vector_rank: h.vector_rank,
            })
            .collect())
    }

    /// Keyword search only: the `k` best `(id, bm25_score)` pairs.
    fn search_bm25(&self, py: Python<'_>, text: &str, k: usize) -> Vec<(u32, f32)> {
        py.detach(|| self.inner.search_bm25(text, k))
    }

    /// Vector search only: the `k` best `(id, cosine)` pairs. `mode` is one of
    /// `VECTOR_MODES`; the default is what hybrid search uses.
    #[pyo3(signature = (vector, k, candidates = 200, mode = "ternary+int8"))]
    fn search_vector(
        &self,
        py: Python<'_>,
        vector: PyReadonlyArray1<'_, f32>,
        k: usize,
        candidates: usize,
        mode: &str,
    ) -> PyResult<Vec<(u32, f32)>> {
        let mode: VectorMode = mode.parse().map_err(to_py)?;
        let vector = vector.as_slice()?;
        py.detach(|| self.inner.search_vector_with(vector, k, candidates, mode))
            .map_err(to_py)
    }

    /// The stored chunk with this id.
    fn chunk(&self, id: u32) -> PyResult<PyChunkRecord> {
        self.inner.chunk(id).map(PyChunkRecord::from).map_err(to_py)
    }

    /// The generation's manifest as a dict (parsed from `manifest.json`).
    #[getter]
    fn manifest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = self.inner.manifest().to_json().map_err(to_py)?;
        let text = String::from_utf8(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.import("json")?.call_method1("loads", (text,))
    }

    /// Directory of the opened generation.
    #[getter]
    fn dir(&self) -> PathBuf {
        self.inner.dir().to_path_buf()
    }

    /// Whether the index stores embeddings (false for keyword-only indexes).
    #[getter]
    fn has_vectors(&self) -> bool {
        self.inner.has_vectors()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Whether `root` holds a committed index.
#[pyfunction]
pub fn index_exists(root: PathBuf) -> bool {
    index::exists(&root)
}

/// Convert a dict to JSON values by round-tripping through Python's `json`,
/// so any JSON-serializable dict works and the error messages are familiar.
fn dict_to_json_map(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
) -> PyResult<BTreeMap<String, serde_json::Value>> {
    let text: String = py
        .import("json")?
        .call_method1("dumps", (dict,))?
        .extract()?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(format!("metadata: {e}")))
}
