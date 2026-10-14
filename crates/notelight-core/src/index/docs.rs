//! The chunk store: one JSON record per chunk, randomly accessible by id.
//!
//! File layout (little-endian, header described in `docs/index-format.md`):
//!
//! ```text
//! header   magic "NLDS", version 1, a = n, b = 0
//! offsets  [u64; n + 1]   byte ranges into records
//! records  JSON objects, concatenated
//! ```
//!
//! JSON keeps records readable and lets fields be added without breaking
//! older readers (unknown fields are ignored). Records are small and only the
//! top-k hits are decoded per query, so parsing cost does not matter.

use serde::{Deserialize, Serialize};

use crate::bin_format::{self, HEADER_LEN, to_u32, u64_at};
use crate::error::{Error, Result};

const MAGIC: [u8; 4] = *b"NLDS";
const VERSION: u16 = 1;

/// Everything stored about one chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// Source file path, relative to the indexed folder, with `/` separators.
    pub path: String,
    /// Enclosing headings, outermost first.
    #[serde(default)]
    pub heading_path: Vec<String>,
    /// 1-based first line of the chunk in the source file.
    pub line_start: u32,
    /// 1-based last line of the chunk (inclusive).
    pub line_end: u32,
    /// The chunk text.
    pub text: String,
}

/// Serializes chunk records.
#[derive(Debug, Default)]
pub struct DocsBuilder {
    offsets: Vec<u64>,
    records: Vec<u8>,
}

impl DocsBuilder {
    /// Append a record. Returns its id.
    ///
    /// # Errors
    /// Fails if the record cannot be serialized or the store exceeds 32-bit ids.
    pub fn add(&mut self, record: &ChunkRecord) -> Result<u32> {
        let id = to_u32(self.offsets.len(), "chunk count")?;
        self.offsets.push(self.records.len() as u64);
        serde_json::to_writer(&mut self.records, record)
            .map_err(|e| Error::Invalid(format!("chunk record: {e}")))?;
        Ok(id)
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Serialize the store.
    ///
    /// # Errors
    /// Fails if counts exceed 32-bit limits.
    pub fn finish(self) -> Result<Vec<u8>> {
        let n = to_u32(self.offsets.len(), "chunk count")?;
        let mut w = bin_format::Writer::with_header(MAGIC, VERSION, n, 0);
        for &o in &self.offsets {
            w.u64(o);
        }
        w.u64(self.records.len() as u64);
        w.bytes(&self.records);
        Ok(w.into_inner())
    }
}

/// Read-only chunk store backed by its serialized bytes.
#[derive(Debug)]
pub struct Docs<B> {
    bytes: B,
    n: usize,
    records_at: usize,
}

impl<B: AsRef<[u8]>> Docs<B> {
    /// Open a chunk store, validating its offset table.
    ///
    /// # Errors
    /// Fails if the bytes are not a well-formed chunk store.
    pub fn from_bytes(bytes: B) -> Result<Self> {
        let data = bytes.as_ref();
        let header = bin_format::read_header(data, MAGIC, VERSION, "docs")?;
        let n = header.a as usize;
        // u64 math: `8 * (n + 1)` must not wrap on 32-bit targets.
        let records_at = HEADER_LEN as u64 + 8 * (u64::from(header.a) + 1);
        if (data.len() as u64) < records_at {
            return Err(Error::format("docs: truncated offset table"));
        }
        let records_at = records_at as usize;
        let mut prev = 0u64;
        for i in 0..=n {
            let o = u64_at(data, HEADER_LEN + 8 * i);
            if o < prev {
                return Err(Error::format("docs: offsets are not monotonic"));
            }
            prev = o;
        }
        if (records_at as u64).checked_add(prev) != Some(data.len() as u64) {
            return Err(Error::format("docs: size does not match offsets"));
        }
        Ok(Self {
            bytes,
            n,
            records_at,
        })
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Decode record `id`.
    ///
    /// # Errors
    /// Fails if `id` is out of range or the record is not valid JSON.
    pub fn get(&self, id: u32) -> Result<ChunkRecord> {
        let i = id as usize;
        if i >= self.n {
            return Err(Error::Invalid(format!("chunk id {id} out of range")));
        }
        let data = self.bytes.as_ref();
        let start = self.records_at + u64_at(data, HEADER_LEN + 8 * i) as usize;
        let end = self.records_at + u64_at(data, HEADER_LEN + 8 * (i + 1)) as usize;
        serde_json::from_slice(&data[start..end])
            .map_err(|e| Error::format(format!("docs: record {id}: {e}")))
    }
}
