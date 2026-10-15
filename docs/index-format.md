# Index format (schema version 2)

An index is a directory of immutable **generations**. A generation is written
once, never modified, and replaced as a whole by the next one.

```
<root>/
  CURRENT              "gen-000007\n": the live generation
  gen-000006/          previous generation, kept for readers that still have it open
  gen-000007/
    manifest.json
    docs.bin
    bm25.bin
    vectors.bin        only when manifest.dim > 0
```

## Commit protocol

1. **Sweep.** Remove what failed commits left behind: `.tmp-*` directories,
   `CURRENT.tmp-*` files, and generation directories numbered higher than
   `CURRENT` (renamed into place but never made live). If `CURRENT` is missing,
   every generation is treated as an orphan; the index is a cache and is
   rebuilt.
2. **Write.** Write every file into `<root>/.tmp-gen-NNNNNN-<pid>-<nanos>/` and
   `fsync` each one. `manifest.json` goes last. Then rename the directory to
   `gen-NNNNNN`.
3. **Publish.** Write `CURRENT.tmp-<pid>-<nanos>`, `fsync` it, and rename it
   over `CURRENT`. On Linux and macOS this replacement is atomic. On Windows,
   replacing a file by rename is also all-or-nothing, but it fails while
   another process holds the file open without delete sharing (antivirus,
   search indexers). The writer retries for about a second before giving up.
   Either way, a reader sees the old generation or the new one, never a mix.
4. **Collect.** Delete every generation except the new one and the one that was
   live before this commit, which readers may still have open. Deletion
   failures are ignored: on Windows a generation that a reader still has
   memory-mapped cannot be deleted, so it is retried on the next commit.

If any step fails, the writer removes what it created (the temporary directory,
or the renamed generation if the failure happened in step 3), and the previous
generation stays live. After a crash, step 1 of the next commit cleans up.

Only one writer may commit at a time. The Python layer enforces this with a
lock file.

## manifest.json

```json
{
  "format": "notelight-index",
  "schema_version": 2,
  "created_by": "notelight-core 0.1.0",
  "generation": 7,
  "n_chunks": 1834,
  "dim": 384,
  "embedder": "BAAI/bge-small-en-v1.5@<revision>",
  "bm25": { "k1": 1.2, "b": 0.75 },
  "files": { "bm25.bin": 412816, "docs.bin": 903112, "vectors.bin": 793504 },
  "metadata": { "chunk_max_chars": 1200 }
}
```

- `schema_version` must match exactly; anything else is rejected with a "rebuild
  the index" error.
- `files` records each file's size. Opening fails if a size differs, which
  catches truncated copies cheaply.
- `embedder` is an opaque string chosen by the caller. The Python layer refuses
  to search with a different model than the one that built the index.
- `metadata` is free-form and is ignored by the core.

## Binary files

All integers and floats are little-endian. Every binary file starts with the
same 16-byte header:

| bytes | field   | type     |
|-------|---------|----------|
| 0..4  | magic   | 4 ASCII bytes identifying the file |
| 4..6  | version | `u16`, per-file format version (1) |
| 6..8  | flags   | `u16`, reserved, 0 |
| 8..12 | a       | `u32`, see each file |
| 12..16| b       | `u32`, see each file |

Readers decode values with `from_le_bytes` at known offsets instead of casting
byte slices to typed slices. That needs no `unsafe` code and no alignment
guarantees, and it works the same on every platform, including WebAssembly.

### docs.bin: chunk records (magic `NLDS`)

```
header   a = n (chunks), b = 0
offsets  [u64; n + 1]    record i spans offsets[i]..offsets[i+1] in `records`
records  JSON objects, concatenated
```

Each record:

```json
{"path": "notes/auth.md", "heading_path": ["Auth"], "line_start": 12, "line_end": 18, "text": "..."}
```

`path` is relative to the indexed folder and uses `/` separators.
`line_start` and `line_end` are 1-based and inclusive. Unknown fields are
ignored, so later versions can add fields without a schema bump.

### bm25.bin: inverted index (magic `NLBM`)

```
header        a = n_docs, b = n_terms
params        avgdl f32, k1 f32, b f32, reserved u32
doc_lens      [u32; n_docs]          tokens per document
term_offsets  [u32; n_terms + 1]     byte ranges into term_bytes
post_offsets  [u32; n_terms + 1]     entry ranges into postings
term_bytes    UTF-8 terms in strictly ascending byte order, padded to 4 bytes
postings      [(doc u32, tf u32); P] grouped by term, ascending doc id
```

Terms are looked up by binary search over `term_bytes`. On open, the reader
checks that the offsets are monotonic, the terms are sorted UTF-8, and every
posting points to an existing document. After that, queries index without
further checks.

The indexed text of a chunk is its heading path joined by spaces, a newline,
then the chunk text.

### vectors.bin: quantized embeddings (magic `NLVC`)

```
header   version 2, a = n, b = dim
binary   [u8; n * ceil(dim / 8)]   sign bits, little-endian bit order, padded to 4 bytes
masks    [u8; n * ceil(dim / 8)]   ternary masks, same bit order, padded to 4 bytes
scales   [f32; n]                  int8 scale per vector
int8     [i8; n * dim]             round(x / scale), clamped to -127..=127
```

Vectors are L2-normalized before quantization. Bit `j` of vector `i` is bit
`j % 8` of byte `i * ceil(dim/8) + j / 8`, which matches
`numpy.packbits(v > 0, bitorder="little")`.

A vector's ternary code (used by the default `ternary+int8` search) is its
sign bits plus its mask. Mask bit `j` is set when
`|c_j| > 0.7 * mean(|c|)`, where `c` are the vector's int8 codes taken as
floats (`quant::ternary_mask`), so a mask can always be recomputed from the
stored codes. Padding bits past `dim` in the last byte of a sign code or a
mask are written as zero and never read: the query's own mask is zero there.

Schema version 1 had no mask plane and derived the masks on first use; a
version 1 index is rejected and rebuilt.

## Versioning rules

- Changing the layout of any file, or the meaning of a field, bumps
  `schema_version`.
- Adding optional fields to `manifest.json` or to chunk records does not.
- Readers never try to migrate old data. An index is a cache of your files, so
  rebuilding it is always safe.
