//! `notelight-core`: the retrieval engine behind notelight.
//!
//! This crate has no Python, ML or async dependencies. It is wrapped for
//! Python by `notelight-py` (`PyO3`) and, in v2, compiled to WebAssembly.
//!
//! Milestone 0 ships only the quantization primitives below, enough to prove
//! the Rust → Python pipeline end to end. Chunking, BM25, search and the
//! on-disk index format arrive in milestone 1.

pub mod quant;

/// Version of the core crate, exposed so Python can report it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
