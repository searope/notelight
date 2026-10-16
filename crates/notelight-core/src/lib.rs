//! `notelight-core`: the retrieval engine behind notelight.
//!
//! This crate has no Python, ML or async dependencies. It is wrapped for
//! Python by `notelight-py` (`PyO3`) and, in v2, compiled to WebAssembly.
//!
//! - [`chunk`]: structure-aware splitting of documents into chunks
//! - [`tokenize`]: terms for keyword search
//! - [`bm25`]: keyword search over a serialized inverted index
//! - [`quant`] and [`vectors`]: binary + int8 vector search
//! - [`fuse`]: reciprocal rank fusion of the two
//! - [`phrase`]: quoted phrases and the in-order bonus applied after fusion
//! - [`index`]: the on-disk index (manifest, files, atomic generations)

pub(crate) mod bin_format;
pub mod bm25;
pub mod chunk;
pub mod error;
pub mod fuse;
pub mod index;
pub mod phrase;
pub mod quant;
pub mod tokenize;
pub mod topk;
pub mod vectors;

pub use error::{Error, Result};

/// Version of the core crate, exposed so Python can report it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
