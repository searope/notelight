//! Python bindings for `notelight-core`, imported as `notelight._core`.
//!
//! Keep this layer thin: convert types, release the GIL for heavy work, and
//! delegate everything else to the core crate.

// PyO3 extracts arguments by value; borrowing them is not an option here.
#![allow(clippy::needless_pass_by_value)]

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use notelight_core::quant;

/// Quantize a batch of float32 vectors, shape `(n, dim)`, to packed binary
/// codes of shape `(n, ceil(dim / 8))`.
#[pyfunction]
fn quantize_binary<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let view = vectors.as_array();
    let (n, dim) = view.dim();
    let code_len = quant::binary_code_len(dim);
    // Borrow the input as one contiguous row-major slice (copies only if the
    // caller passed a strided view), then quantize with the GIL released.
    let contiguous = view.as_standard_layout();
    let data = contiguous
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("input must be convertible to a contiguous array"))?;
    let flat: Vec<u8> = py.detach(|| {
        if dim == 0 {
            return Vec::new();
        }
        data.chunks_exact(dim)
            .flat_map(quant::quantize_binary)
            .collect()
    });
    let arr = numpy::ndarray::Array2::from_shape_vec((n, code_len), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Hamming distance between two packed binary codes of equal length.
#[pyfunction]
fn hamming(a: PyReadonlyArray1<'_, u8>, b: PyReadonlyArray1<'_, u8>) -> PyResult<u32> {
    let (a, b) = (a.as_slice()?, b.as_slice()?);
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "codes must have equal length, got {} and {}",
            a.len(),
            b.len()
        )));
    }
    Ok(quant::hamming(a, b))
}

/// The compiled Rust extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", notelight_core::VERSION)?;
    m.add_function(wrap_pyfunction!(quantize_binary, m)?)?;
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    Ok(())
}
