//! `chunk_text` and the `Chunk` class.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use notelight_core::chunk::{self, ChunkOpts};

/// A contiguous piece of a document, as produced by `chunk_text`.
///
/// `byte_start`/`byte_end` are UTF-8 byte offsets into the source text (not
/// Python string indices); `line_start`/`line_end` are 1-based and inclusive.
#[pyclass(
    module = "notelight._core",
    name = "Chunk",
    frozen,
    get_all,
    eq,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyChunk {
    text: String,
    heading_path: Vec<String>,
    byte_start: usize,
    byte_end: usize,
    line_start: u32,
    line_end: u32,
}

#[pymethods]
impl PyChunk {
    fn __repr__(&self) -> String {
        format!(
            "Chunk(lines={}-{}, heading_path={:?}, chars={})",
            self.line_start,
            self.line_end,
            self.heading_path,
            self.text.chars().count()
        )
    }
}

impl From<chunk::Chunk> for PyChunk {
    fn from(c: chunk::Chunk) -> Self {
        Self {
            text: c.text,
            heading_path: c.heading_path,
            byte_start: c.byte_start,
            byte_end: c.byte_end,
            line_start: c.line_start,
            line_end: c.line_end,
        }
    }
}

/// Split `text` into chunks of at most `max_chars` characters, following
/// Markdown headings and code fences when `markdown` is true.
#[pyfunction]
#[pyo3(signature = (text, max_chars = 1200, markdown = true))]
pub fn chunk_text(
    py: Python<'_>,
    text: &str,
    max_chars: usize,
    markdown: bool,
) -> PyResult<Vec<PyChunk>> {
    // The core asserts this; check first so Python gets a ValueError, not a panic.
    if max_chars == 0 {
        return Err(PyValueError::new_err("max_chars must be positive"));
    }
    let opts = ChunkOpts {
        max_chars,
        markdown,
    };
    Ok(py.detach(|| {
        chunk::chunk_text(text, &opts)
            .into_iter()
            .map(PyChunk::from)
            .collect()
    }))
}
