//! Building a new index generation and committing it atomically.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::{
    BM25_FILE, CURRENT_FILE, ChunkRecord, DOCS_FILE, DocsBuilder, FORMAT_NAME, MANIFEST_FILE,
    Manifest, SCHEMA_VERSION, VECTORS_FILE, generation_dir_name, list_generations, read_current,
};
use crate::bm25::{Bm25Builder, Bm25Params};
use crate::error::{Error, Result};
use crate::vectors::VectorBuilder;

/// Settings for a new index.
#[derive(Debug, Clone, Default)]
pub struct WriterOptions {
    /// Embedding dimension; 0 builds a keyword-only index.
    pub dim: usize,
    /// Embedding model identifier recorded in the manifest.
    pub embedder: Option<String>,
    /// BM25 parameters.
    pub bm25: Bm25Params,
    /// Free-form settings recorded in the manifest.
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Result of a successful commit.
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// The generation number that is now live.
    pub generation: u64,
    /// Its directory.
    pub dir: PathBuf,
    /// Number of chunks written.
    pub n_chunks: u32,
}

/// The contents of one generation, built in memory by [`IndexWriter::build`].
#[derive(Debug, Clone)]
pub struct GenerationFiles {
    /// The manifest; [`Manifest::to_json`] gives the bytes of `manifest.json`.
    pub manifest: Manifest,
    /// Bytes of `docs.bin`.
    pub docs: Vec<u8>,
    /// Bytes of `bm25.bin`.
    pub bm25: Vec<u8>,
    /// Bytes of `vectors.bin`, absent for a keyword-only index.
    pub vectors: Option<Vec<u8>>,
}

/// Accumulates chunks in memory, then writes them as one generation.
#[derive(Debug)]
pub struct IndexWriter {
    opts: WriterOptions,
    docs: DocsBuilder,
    bm25: Bm25Builder,
    vectors: Option<VectorBuilder>,
}

impl IndexWriter {
    /// Start an empty index.
    #[must_use]
    pub fn new(opts: WriterOptions) -> Self {
        let vectors = (opts.dim > 0).then(|| VectorBuilder::new(opts.dim));
        Self {
            opts,
            docs: DocsBuilder::default(),
            bm25: Bm25Builder::new(),
            vectors,
        }
    }

    /// Add one chunk. `embedding` is required when the index has a dimension
    /// and must be absent otherwise. Returns the chunk id.
    ///
    /// The BM25 text is the heading path followed by the chunk text, so a
    /// query for a section title matches the chunks under it.
    ///
    /// # Errors
    /// Fails if the embedding is missing, unexpected, or malformed.
    pub fn add(&mut self, record: &ChunkRecord, embedding: Option<&[f32]>) -> Result<u32> {
        match (&mut self.vectors, embedding) {
            (Some(v), Some(e)) => {
                v.add(e)?;
            }
            (Some(_), None) => {
                return Err(Error::Invalid(
                    "this index stores embeddings; every chunk needs one".into(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::Invalid(
                    "this index is keyword-only (dim = 0); remove the embedding".into(),
                ));
            }
            (None, None) => {}
        }
        let id = self.docs.add(record)?;
        self.bm25.add_text(&super::search_text(record));
        Ok(id)
    }

    /// Number of chunks added.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether no chunks have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Serialize the index as generation number `generation`, in memory.
    ///
    /// [`Self::commit`] writes these bytes to disk; platforms without a
    /// filesystem (the browser build) keep them and pass them to
    /// [`super::IndexReader::from_parts`].
    ///
    /// # Errors
    /// Fails if counts exceed 32-bit limits or the manifest cannot be built.
    pub fn build(self, generation: u64) -> Result<GenerationFiles> {
        let n_chunks = crate::bin_format::to_u32(self.docs.len(), "chunk count")?;
        let docs = self.docs.finish()?;
        let bm25 = self.bm25.finish(self.opts.bm25)?;
        let vectors = self.vectors.map(VectorBuilder::finish).transpose()?;

        let mut files = BTreeMap::new();
        files.insert(DOCS_FILE.to_string(), docs.len() as u64);
        files.insert(BM25_FILE.to_string(), bm25.len() as u64);
        if let Some(v) = &vectors {
            files.insert(VECTORS_FILE.to_string(), v.len() as u64);
        }
        let manifest = Manifest {
            format: FORMAT_NAME.to_string(),
            schema_version: SCHEMA_VERSION,
            created_by: format!("notelight-core {}", crate::VERSION),
            generation,
            n_chunks,
            dim: crate::bin_format::to_u32(self.opts.dim, "dimension")?,
            embedder: self.opts.embedder,
            bm25: self.opts.bm25.into(),
            files,
            metadata: self.opts.metadata,
        };
        Ok(GenerationFiles {
            manifest,
            docs,
            bm25,
            vectors,
        })
    }

    /// Write the next generation under `root` and make it live.
    ///
    /// 1. Sweep leftovers of earlier failed commits: temporary directories,
    ///    temporary pointer files, and generations newer than `CURRENT` (they
    ///    were renamed into place but never became live). Safe because only
    ///    one writer may commit at a time.
    /// 2. Write every file into a uniquely named temporary directory, fsync,
    ///    and rename it to `gen-NNNNNN`.
    /// 3. Replace `CURRENT` by renaming a temporary pointer file over it.
    /// 4. Delete every generation except the new one and the one that was
    ///    live before this commit. Deletion failures (for example a generation
    ///    still memory-mapped by a reader on Windows) are ignored and retried
    ///    on the next commit.
    ///
    /// # Errors
    /// Fails on I/O errors. The previously live generation stays live, and
    /// everything this commit created is removed.
    pub fn commit(self, root: &Path) -> Result<CommitInfo> {
        fs::create_dir_all(root).map_err(|e| Error::io(root, e))?;
        let live_before = read_current(root)?;
        sweep_leftovers(root, live_before)?;
        let generation = list_generations(root)?
            .last()
            .copied()
            .max(live_before)
            .map_or(1, |g| g + 1);
        let built = self.build(generation)?;
        let n_chunks = built.manifest.n_chunks;
        let manifest = built.manifest;
        let mut files = vec![(DOCS_FILE, built.docs), (BM25_FILE, built.bm25)];
        if let Some(v) = built.vectors {
            files.push((VECTORS_FILE, v));
        }

        let suffix = unique_suffix();
        let final_dir = root.join(generation_dir_name(generation));
        let tmp_dir = root.join(format!(
            "{TMP_PREFIX}{}-{suffix}",
            generation_dir_name(generation)
        ));
        let pointer_tmp = root.join(format!("{CURRENT_FILE}{POINTER_TMP_INFIX}{suffix}"));
        let mut created_tmp = false;
        let mut renamed = false;

        let result = (|| {
            fs::create_dir(&tmp_dir).map_err(|e| Error::io(&tmp_dir, e))?;
            created_tmp = true;
            for (name, bytes) in &files {
                write_synced(&tmp_dir.join(name), bytes)?;
            }
            // The manifest goes last: a generation without one is never valid.
            write_synced(&tmp_dir.join(MANIFEST_FILE), &manifest.to_json()?)?;
            sync_dir(&tmp_dir)?;
            rename_with_retry(&tmp_dir, &final_dir)?;
            renamed = true;
            sync_dir(root)?;

            write_synced(
                &pointer_tmp,
                format!("{}\n", generation_dir_name(generation)).as_bytes(),
            )?;
            rename_with_retry(&pointer_tmp, &root.join(CURRENT_FILE))?;
            sync_dir(root)
        })();

        if let Err(e) = result {
            // Undo only what this commit created; the old generation stays live.
            if renamed {
                let _ = fs::remove_dir_all(&final_dir);
            } else if created_tmp {
                let _ = fs::remove_dir_all(&tmp_dir);
            }
            let _ = fs::remove_file(&pointer_tmp);
            return Err(e);
        }

        for old in list_generations(root)? {
            if old != generation && Some(old) != live_before {
                let _ = fs::remove_dir_all(root.join(generation_dir_name(old)));
            }
        }

        Ok(CommitInfo {
            generation,
            dir: final_dir,
            n_chunks,
        })
    }
}

const TMP_PREFIX: &str = ".tmp-";
const POINTER_TMP_INFIX: &str = ".tmp-";

/// Process id plus a nanosecond timestamp: unique even when a container
/// always runs the writer as pid 1.
fn unique_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{}-{nanos}", std::process::id())
}

/// Remove what failed or crashed commits left behind. Errors are ignored:
/// a leftover that cannot be removed now is retried on the next commit.
fn sweep_leftovers(root: &Path, live: Option<u64>) -> Result<()> {
    let entries = fs::read_dir(root).map_err(|e| Error::io(root, e))?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let path = entry.path();
        if name.starts_with(TMP_PREFIX) && path.is_dir() {
            let _ = fs::remove_dir_all(&path);
        } else if name.starts_with(&format!("{CURRENT_FILE}{POINTER_TMP_INFIX}")) {
            let _ = fs::remove_file(&path);
        }
    }
    for g in list_generations(root)? {
        // Generations newer than CURRENT were never live.
        if live.is_none_or(|l| g > l) {
            let _ = fs::remove_dir_all(root.join(generation_dir_name(g)));
        }
    }
    Ok(())
}

/// `fs::rename`, retried briefly on Windows when another process (antivirus,
/// search indexer, a reader opening `CURRENT`) momentarily holds the file.
fn rename_with_retry(from: &Path, to: &Path) -> Result<()> {
    let mut delay = std::time::Duration::from_millis(5);
    for attempt in 0..8 {
        match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < 7 && is_transient(&e) => {
                std::thread::sleep(delay);
                delay *= 2;
            }
            Err(e) => return Err(Error::io(to, e)),
        }
    }
    unreachable!("the last attempt returns")
}

fn is_transient(e: &std::io::Error) -> bool {
    // ERROR_ACCESS_DENIED (5) and ERROR_SHARING_VIOLATION (32) are what
    // Windows reports while another handle is open on the file.
    cfg!(windows) && matches!(e.raw_os_error(), Some(5 | 32))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = File::create(path).map_err(|e| Error::io(path, e))?;
    f.write_all(bytes).map_err(|e| Error::io(path, e))?;
    f.sync_all().map_err(|e| Error::io(path, e))
}

/// Persist directory entries (renames) on Unix. Windows has no equivalent
/// for directories; NTFS journals metadata changes itself.
#[cfg_attr(
    not(unix),
    allow(
        clippy::unnecessary_wraps,
        reason = "fallible on Unix, a no-op elsewhere"
    )
)]
fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| Error::io(dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}
