//! Byte storage for index files: memory-mapped on native targets, owned
//! buffers on WebAssembly or when mapping is disabled.

use std::fs::File;
use std::path::Path;

use crate::error::{Error, Result};

/// The bytes of one index file.
#[derive(Debug)]
pub enum Storage {
    /// File contents read into memory.
    Owned(Vec<u8>),
    /// A read-only memory map of the file.
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
}

impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(m) => m,
        }
    }
}

impl Storage {
    /// Open `path`, memory-mapping it when `mmap` is true and the platform allows it.
    ///
    /// # Errors
    /// Fails if the file cannot be opened, read or mapped.
    pub fn open(path: &Path, mmap: bool) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        let len = file.metadata().map_err(|e| Error::io(path, e))?.len();
        // Mapping a zero-length file fails on some platforms; nothing to map anyway.
        if !mmap || len == 0 || cfg!(target_arch = "wasm32") {
            return std::fs::read(path)
                .map(Self::Owned)
                .map_err(|e| Error::io(path, e));
        }
        Self::map(&file, path)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(unsafe_code)]
    fn map(file: &File, path: &Path) -> Result<Self> {
        // SAFETY: memory-mapping is unsafe because the program would observe
        // changes if another process modified or truncated the file while it
        // is mapped. Index files are written once into a fresh generation
        // directory, synced, and never modified afterwards: writers always
        // create a new generation instead of editing files in place. External
        // tools editing files inside an index directory are not supported.
        let map = unsafe { memmap2::Mmap::map(file) }.map_err(|e| Error::io(path, e))?;
        Ok(Self::Mapped(map))
    }

    #[cfg(target_arch = "wasm32")]
    fn map(_file: &File, path: &Path) -> Result<Self> {
        std::fs::read(path)
            .map(Self::Owned)
            .map_err(|e| Error::io(path, e))
    }

    /// Whether the bytes are memory-mapped.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        !matches!(self, Self::Owned(_))
    }
}
