//! Error type for the core crate.

use std::path::PathBuf;

/// Errors produced while building, writing or reading an index.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An I/O operation on `path` failed.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// The file or directory involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A file is corrupt, truncated, or not a notelight index file.
    #[error("invalid index data: {0}")]
    Format(String),

    /// The index was written by an incompatible schema version.
    #[error(
        "index schema version {found} is not supported (this build reads {expected}); rebuild the index"
    )]
    Schema {
        /// Version found on disk.
        found: u32,
        /// Version this build understands.
        expected: u32,
    },

    /// A vector had the wrong number of dimensions.
    #[error("expected {expected}-dimensional vector, got {found}")]
    Dimension {
        /// Dimension the index was built with.
        expected: usize,
        /// Dimension that was supplied.
        found: usize,
    },

    /// The caller used the API incorrectly (for example, missing embeddings).
    #[error("{0}")]
    Invalid(String),
}

impl Error {
    pub(crate) fn format(msg: impl Into<String>) -> Self {
        Self::Format(msg.into())
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Result alias for this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
