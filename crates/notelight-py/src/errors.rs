//! Python exceptions for `notelight_core::Error`.
//!
//! | Rust variant        | Python exception                          |
//! |---------------------|-------------------------------------------|
//! | `Io`                | `OSError` (built in)                      |
//! | `Format`            | `IndexFormatError` (corrupt index)        |
//! | `Schema`            | `SchemaError`, a subclass of the above    |
//! | `Dimension`         | `ValueError` (built in)                   |
//! | `Invalid`           | `ValueError` (built in)                   |
//!
//! Both index errors mean "rebuild the index", so callers can catch
//! `IndexFormatError` alone.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::{PyErr, create_exception};

use notelight_core::Error;

create_exception!(
    _core,
    NotelightError,
    pyo3::exceptions::PyException,
    "Base class for notelight errors raised by the Rust core."
);
create_exception!(
    _core,
    IndexFormatError,
    NotelightError,
    "The index on disk is corrupt or not a notelight index. Rebuild it."
);
create_exception!(
    _core,
    SchemaError,
    IndexFormatError,
    "The index was written by an incompatible version of notelight. Rebuild it."
);

/// Convert a core error into the matching Python exception.
pub fn to_py(err: Error) -> PyErr {
    let msg = err.to_string();
    match err {
        Error::Io { .. } => PyOSError::new_err(msg),
        Error::Format(_) => IndexFormatError::new_err(msg),
        Error::Schema { .. } => SchemaError::new_err(msg),
        Error::Dimension { .. } | Error::Invalid(_) => PyValueError::new_err(msg),
    }
}

/// Register the exception classes on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("NotelightError", py.get_type::<NotelightError>())?;
    m.add("IndexFormatError", py.get_type::<IndexFormatError>())?;
    m.add("SchemaError", py.get_type::<SchemaError>())?;
    Ok(())
}
