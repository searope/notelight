//! `manifest.json`: what an index generation contains and how it was built.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bm25::Bm25Params;
use crate::error::{Error, Result};

/// Value of [`Manifest::format`], so foreign JSON files are rejected early.
pub const FORMAT_NAME: &str = "notelight-index";

/// Schema version written by this build. Bump on any incompatible change.
pub const SCHEMA_VERSION: u32 = 1;

/// Description of one index generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Always [`FORMAT_NAME`].
    pub format: String,
    /// On-disk schema version; must equal [`SCHEMA_VERSION`] to be read.
    pub schema_version: u32,
    /// Software that wrote the index, e.g. `notelight-core 0.1.0`.
    pub created_by: String,
    /// Monotonic generation number within the index directory.
    pub generation: u64,
    /// Number of chunks (documents) in every file of this generation.
    pub n_chunks: u32,
    /// Embedding dimension, or 0 for a keyword-only index.
    pub dim: u32,
    /// Identifier of the embedding model, chosen by the caller (for example
    /// `"BAAI/bge-small-en-v1.5@<revision>"`). Readers compare it with their
    /// own model so mismatched vectors are never searched silently.
    pub embedder: Option<String>,
    /// BM25 parameters used at build time.
    pub bm25: Bm25Settings,
    /// Size in bytes of each data file, checked when the index is opened.
    pub files: BTreeMap<String, u64>,
    /// Free-form settings recorded by the caller (chunking options, source root, ...).
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Serializable copy of [`Bm25Params`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bm25Settings {
    /// See [`Bm25Params::k1`].
    pub k1: f32,
    /// See [`Bm25Params::b`].
    pub b: f32,
}

impl From<Bm25Params> for Bm25Settings {
    fn from(p: Bm25Params) -> Self {
        Self { k1: p.k1, b: p.b }
    }
}

impl Manifest {
    /// Parse and validate a manifest.
    ///
    /// # Errors
    /// Fails on malformed JSON, a foreign format, or an unsupported schema version.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let m: Self = serde_json::from_slice(bytes)
            .map_err(|e| Error::format(format!("manifest.json: {e}")))?;
        if m.format != FORMAT_NAME {
            return Err(Error::format(format!(
                "manifest.json: format is {:?}, expected {FORMAT_NAME:?}",
                m.format
            )));
        }
        if m.schema_version != SCHEMA_VERSION {
            return Err(Error::Schema {
                found: m.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(m)
    }

    /// Serialize as pretty-printed JSON.
    ///
    /// # Errors
    /// Fails only if a metadata value cannot be serialized.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        let mut out = serde_json::to_vec_pretty(self)
            .map_err(|e| Error::Invalid(format!("manifest: {e}")))?;
        out.push(b'\n');
        Ok(out)
    }
}
