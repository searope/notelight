//! Little-endian helpers for the binary index files.
//!
//! Every binary file starts with a 16-byte header:
//!
//! | bytes | field   | notes                               |
//! |-------|---------|-------------------------------------|
//! | 0..4  | magic   | identifies the file type            |
//! | 4..6  | version | per-file format version (`u16`)     |
//! | 6..8  | flags   | reserved, zero                      |
//! | 8..16 | payload | two `u32` fields defined per file   |
//!
//! Values are read with `from_le_bytes` on demand instead of casting byte
//! slices to `&[u32]`. That needs no `unsafe`, no alignment guarantees, and
//! behaves the same on every platform, including WebAssembly.

use crate::error::{Error, Result};

/// Size of the common file header in bytes.
pub const HEADER_LEN: usize = 16;

/// Append-only little-endian buffer used by the index writers.
#[derive(Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    /// Start a file with the common header.
    pub fn with_header(magic: [u8; 4], version: u16, a: u32, b: u32) -> Self {
        let mut w = Self::default();
        w.buf.extend_from_slice(&magic);
        w.u16(version);
        w.u16(0);
        w.u32(a);
        w.u32(b);
        w
    }

    /// Append a `u16`.
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a `u32`.
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a `u64`.
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append an `f32`.
    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append raw bytes.
    pub fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// Pad with zeros to a multiple of 4 bytes, so sections stay word-aligned.
    pub fn pad4(&mut self) {
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
    }

    /// Finish and return the buffer.
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }
}

/// Parsed common header.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    /// First payload field.
    pub a: u32,
    /// Second payload field.
    pub b: u32,
}

/// Validate the header of `bytes` and return its payload fields.
pub fn read_header(bytes: &[u8], magic: [u8; 4], version: u16, what: &str) -> Result<Header> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::format(format!("{what}: file is too short")));
    }
    if bytes[0..4] != magic {
        return Err(Error::format(format!("{what}: wrong magic bytes")));
    }
    let found = u16::from_le_bytes([bytes[4], bytes[5]]);
    if found != version {
        return Err(Error::format(format!(
            "{what}: unsupported version {found} (expected {version})"
        )));
    }
    Ok(Header {
        a: u32_at(bytes, 8),
        b: u32_at(bytes, 12),
    })
}

/// Read a `u32` at byte offset `pos`. Callers validate bounds up front.
#[inline]
pub fn u32_at(bytes: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(bytes[pos..pos + 4].try_into().expect("4 bytes"))
}

/// Read a `u64` at byte offset `pos`. Callers validate bounds up front.
#[inline]
pub fn u64_at(bytes: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(bytes[pos..pos + 8].try_into().expect("8 bytes"))
}

/// Read an `f32` at byte offset `pos`. Callers validate bounds up front.
#[inline]
pub fn f32_at(bytes: &[u8], pos: usize) -> f32 {
    f32::from_le_bytes(bytes[pos..pos + 4].try_into().expect("4 bytes"))
}

/// Convert a length to `u32`, failing cleanly instead of truncating.
pub fn to_u32(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::format(format!("{what} exceeds u32::MAX")))
}
