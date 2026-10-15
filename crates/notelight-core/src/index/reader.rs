//! Opening an index generation and searching it.

use std::path::{Path, PathBuf};

use super::{
    BM25_FILE, ChunkRecord, DOCS_FILE, Docs, MANIFEST_FILE, Manifest, Storage, VECTORS_FILE,
    current_generation_dir,
};
use crate::bm25::Bm25;
use crate::error::{Error, Result};
use crate::fuse::{DEFAULT_RRF_K, rrf};
use crate::vectors::{VectorMode, Vectors};

/// How to open an index.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Memory-map files (default). When `false`, files are read into memory.
    pub mmap: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { mmap: true }
    }
}

/// Parameters for [`IndexReader::search`].
#[derive(Debug, Clone, Copy)]
pub struct SearchOpts {
    /// Number of results to return.
    pub k: usize,
    /// How many results each retriever (BM25, vectors) contributes to fusion.
    pub depth: usize,
    /// Candidates from the ternary stage (see [`VectorMode`]) re-scored with
    /// int8 vectors.
    pub candidates: usize,
    /// RRF constant; see [`crate::fuse`].
    pub rrf_k: f32,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            k: 10,
            depth: 50,
            candidates: 200,
            rrf_k: DEFAULT_RRF_K,
        }
    }
}

/// One hybrid search result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Chunk id; pass to [`IndexReader::chunk`].
    pub id: u32,
    /// Fused RRF score (higher is better).
    pub score: f32,
    /// 1-based rank in the BM25 list, if the chunk appeared there.
    pub bm25_rank: Option<u32>,
    /// 1-based rank in the vector list, if the chunk appeared there.
    pub vector_rank: Option<u32>,
}

/// A read-only, opened index generation.
#[derive(Debug)]
pub struct IndexReader {
    dir: PathBuf,
    manifest: Manifest,
    docs: Docs<Storage>,
    bm25: Bm25<Storage>,
    vectors: Option<Vectors<Storage>>,
}

impl IndexReader {
    /// Open the live generation of the index at `root`.
    ///
    /// # Errors
    /// Fails if there is no committed index, or any file is invalid.
    pub fn open(root: &Path, opts: OpenOptions) -> Result<Self> {
        Self::open_generation(&current_generation_dir(root)?, opts)
    }

    /// Open a specific generation directory.
    ///
    /// # Errors
    /// Fails if any file is missing, has an unexpected size, or is invalid.
    pub fn open_generation(dir: &Path, opts: OpenOptions) -> Result<Self> {
        let manifest_path = dir.join(MANIFEST_FILE);
        let raw = std::fs::read(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
        let manifest = Manifest::from_json(&raw)?;

        let load = |name: &str| -> Result<Storage> {
            let path = dir.join(name);
            let storage = Storage::open(&path, opts.mmap)?;
            let expected = manifest.files.get(name).copied();
            if expected != Some(storage.as_ref().len() as u64) {
                return Err(Error::format(format!(
                    "{name}: size {} does not match manifest ({expected:?})",
                    storage.as_ref().len()
                )));
            }
            Ok(storage)
        };

        let docs = Docs::from_bytes(load(DOCS_FILE)?)?;
        let bm25 = Bm25::from_bytes(load(BM25_FILE)?)?;
        let vectors = if manifest.dim > 0 {
            Some(Vectors::from_bytes(load(VECTORS_FILE)?)?)
        } else {
            None
        };
        Self::assemble(dir.to_path_buf(), manifest, docs, bm25, vectors)
    }

    /// Build a reader from in-memory file contents, for platforms without a
    /// filesystem (the browser build) and for tests.
    ///
    /// # Errors
    /// Fails if any part is invalid or the parts disagree.
    pub fn from_parts(
        manifest_json: &[u8],
        docs: Vec<u8>,
        bm25: Vec<u8>,
        vectors: Option<Vec<u8>>,
    ) -> Result<Self> {
        let manifest = Manifest::from_json(manifest_json)?;
        let docs = Docs::from_bytes(Storage::Owned(docs))?;
        let bm25 = Bm25::from_bytes(Storage::Owned(bm25))?;
        let vectors = vectors
            .map(|v| Vectors::from_bytes(Storage::Owned(v)))
            .transpose()?;
        Self::assemble(PathBuf::new(), manifest, docs, bm25, vectors)
    }

    fn assemble(
        dir: PathBuf,
        manifest: Manifest,
        docs: Docs<Storage>,
        bm25: Bm25<Storage>,
        vectors: Option<Vectors<Storage>>,
    ) -> Result<Self> {
        let n = manifest.n_chunks as usize;
        if docs.len() != n || bm25.len() != n {
            return Err(Error::format("chunk counts disagree between files"));
        }
        match &vectors {
            Some(v) if v.len() != n || v.dim() != manifest.dim as usize => {
                return Err(Error::format("vector file disagrees with manifest"));
            }
            None if manifest.dim > 0 => {
                return Err(Error::format("manifest has a dimension but no vectors"));
            }
            _ => {}
        }
        Ok(Self {
            dir,
            manifest,
            docs,
            bm25,
            vectors,
        })
    }

    /// The generation's manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Directory of the opened generation (empty for [`Self::from_parts`]).
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Number of chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index has no chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Whether the index stores embeddings.
    #[must_use]
    pub fn has_vectors(&self) -> bool {
        self.vectors.is_some()
    }

    /// Fetch the record for chunk `id`.
    ///
    /// # Errors
    /// Fails if `id` is out of range or the record is corrupt.
    pub fn chunk(&self, id: u32) -> Result<ChunkRecord> {
        self.docs.get(id)
    }

    /// Keyword search: the `k` best `(id, bm25_score)` pairs.
    #[must_use]
    pub fn search_bm25(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        self.bm25.search(query, k)
    }

    /// Vector search in the default [`VectorMode`] (ternary candidates,
    /// int8 rescoring): the `k` best `(id, cosine)` pairs.
    ///
    /// # Errors
    /// Fails if the index is keyword-only or the query has the wrong dimension.
    pub fn search_vector(
        &self,
        query: &[f32],
        k: usize,
        candidates: usize,
    ) -> Result<Vec<(u32, f32)>> {
        self.search_vector_with(query, k, candidates, VectorMode::default())
    }

    /// Vector search with an explicit [`VectorMode`], for comparing
    /// quantization strategies. Scores approximate cosine similarity, except
    /// in [`VectorMode::Binary`] (see there).
    ///
    /// # Errors
    /// Fails if the index is keyword-only or the query is invalid for the mode.
    pub fn search_vector_with(
        &self,
        query: &[f32],
        k: usize,
        candidates: usize,
        mode: VectorMode,
    ) -> Result<Vec<(u32, f32)>> {
        self.vectors
            .as_ref()
            .ok_or_else(|| Error::Invalid("this index is keyword-only".into()))?
            .search_with(query, k, candidates, mode)
    }

    /// Hybrid search: BM25 and (when `query_vec` is given) vector results,
    /// fused with reciprocal rank fusion.
    ///
    /// # Errors
    /// Fails if `query_vec` is given for a keyword-only index or has the
    /// wrong dimension.
    pub fn search(
        &self,
        query_text: &str,
        query_vec: Option<&[f32]>,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let depth = opts.depth.max(opts.k);
        let bm25_ids: Vec<u32> = self
            .search_bm25(query_text, depth)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let vec_ids: Vec<u32> = match query_vec {
            Some(q) => self
                .search_vector(q, depth, opts.candidates)?
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            None => Vec::new(),
        };

        let rank_in =
            |list: &[u32], id: u32| list.iter().position(|&x| x == id).map(|p| p as u32 + 1);
        Ok(rrf(&[&bm25_ids, &vec_ids], opts.rrf_k, opts.k)
            .into_iter()
            .map(|(id, score)| Hit {
                id,
                score,
                bm25_rank: rank_in(&bm25_ids, id),
                vector_rank: rank_in(&vec_ids, id),
            })
            .collect())
    }
}
