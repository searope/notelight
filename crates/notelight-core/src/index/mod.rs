//! The on-disk index: an immutable snapshot per *generation*, swapped atomically.
//!
//! ```text
//! <root>/
//!   CURRENT              text file naming the live generation, e.g. "gen-000007"
//!   gen-000007/
//!     manifest.json      what is inside and how it was built
//!     docs.bin           chunk records
//!     bm25.bin           inverted index
//!     vectors.bin        binary + int8 embeddings (absent for keyword-only indexes)
//! ```
//!
//! **Why generations.** Files are never modified after they are written, which
//! is what makes memory-mapping them safe, and it sidesteps a Windows rule:
//! a file that is memory-mapped cannot be replaced or deleted. A writer builds
//! the next generation in a temporary directory, renames it into place, then
//! replaces `CURRENT` by renaming over it. Readers that opened the previous
//! generation keep working; older ones are removed when possible. See
//! `docs/index-format.md` for the full protocol, including crash recovery.
//!
//! Only one writer at a time is supported; the Python layer holds a lock file.

mod docs;
mod manifest;
mod reader;
mod storage;
mod writer;

pub use docs::{ChunkRecord, Docs, DocsBuilder};
pub use manifest::{Bm25Settings, FORMAT_NAME, Manifest, SCHEMA_VERSION};
pub use reader::{DEFAULT_PHRASE_BOOST, Hit, IndexReader, OpenOptions, SearchOpts};
pub use storage::Storage;
pub use writer::{CommitInfo, GenerationFiles, IndexWriter, WriterOptions};

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Name of the pointer file that selects the live generation.
pub const CURRENT_FILE: &str = "CURRENT";
/// Manifest file name inside a generation.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Chunk store file name inside a generation.
pub const DOCS_FILE: &str = "docs.bin";
/// BM25 file name inside a generation.
pub const BM25_FILE: &str = "bm25.bin";
/// Vector file name inside a generation.
pub const VECTORS_FILE: &str = "vectors.bin";

const GEN_PREFIX: &str = "gen-";

/// The text BM25 indexes for a chunk: its heading path, then its text, so a
/// query for a section title matches the chunks under it. Phrase matching
/// reads the same text.
pub(crate) fn search_text(record: &ChunkRecord) -> String {
    let mut text = record.heading_path.join(" ");
    text.push('\n');
    text.push_str(&record.text);
    text
}

/// Directory name for generation `n`, zero-padded so names sort numerically.
#[must_use]
pub fn generation_dir_name(n: u64) -> String {
    format!("{GEN_PREFIX}{n:06}")
}

fn parse_generation(name: &str) -> Option<u64> {
    name.strip_prefix(GEN_PREFIX)?.parse().ok()
}

/// All generation numbers present under `root`, ascending.
fn list_generations(root: &Path) -> Result<Vec<u64>> {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::io(root, e)),
    };
    let mut gens = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(root, e))?;
        if entry.path().is_dir()
            && let Some(n) = entry.file_name().to_str().and_then(parse_generation)
        {
            gens.push(n);
        }
    }
    gens.sort_unstable();
    Ok(gens)
}

/// Path of the live generation directory, read from `CURRENT`.
///
/// # Errors
/// Fails if `root` holds no committed index.
pub fn current_generation_dir(root: &Path) -> Result<PathBuf> {
    let pointer = root.join(CURRENT_FILE);
    let name = std::fs::read_to_string(&pointer).map_err(|e| Error::io(&pointer, e))?;
    let name = name.trim();
    if parse_generation(name).is_none() {
        return Err(Error::format(format!(
            "CURRENT names {name:?}, not a generation"
        )));
    }
    Ok(root.join(name))
}

/// The live generation number, or `None` if `root` has no `CURRENT` yet.
fn read_current(root: &Path) -> Result<Option<u64>> {
    let pointer = root.join(CURRENT_FILE);
    match std::fs::read_to_string(&pointer) {
        Ok(name) => parse_generation(name.trim()).map(Some).ok_or_else(|| {
            Error::format(format!("CURRENT names {:?}, not a generation", name.trim()))
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::io(&pointer, e)),
    }
}

/// Whether `root` contains a committed index.
#[must_use]
pub fn exists(root: &Path) -> bool {
    current_generation_dir(root).is_ok_and(|d| d.join(MANIFEST_FILE).is_file())
}
