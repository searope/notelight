//! End-to-end tests of the on-disk index: write, commit, reopen, search.

use std::collections::BTreeMap;
use std::fs;

use notelight_core::Error;
use notelight_core::index::{
    self, ChunkRecord, IndexReader, IndexWriter, OpenOptions, SearchOpts, WriterOptions,
};

fn record(path: &str, heading: &[&str], text: &str) -> ChunkRecord {
    ChunkRecord {
        path: path.into(),
        heading_path: heading.iter().map(|s| (*s).to_string()).collect(),
        line_start: 1,
        line_end: 1 + text.lines().count() as u32,
        text: text.into(),
    }
}

fn corpus() -> Vec<ChunkRecord> {
    vec![
        record(
            "setup.md",
            &["Setup", "Windows"],
            "Install with winget and restart.",
        ),
        record(
            "setup.md",
            &["Setup", "Linux"],
            "Install with apt on Debian or Ubuntu.",
        ),
        record(
            "notes/auth.md",
            &["Auth"],
            "Fixed the auth timeout by raising the token TTL.",
        ),
        record("notes/db.md", &[], "Postgres connection pool sizing notes."),
    ]
}

/// Embeddings that put each chunk on its own axis, so vector search is checkable.
fn one_hot(i: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.01; dim];
    v[i] = 1.0;
    v
}

fn write_index(root: &std::path::Path, dim: usize) -> index::CommitInfo {
    let mut w = IndexWriter::new(WriterOptions {
        dim,
        embedder: (dim > 0).then(|| "test-onehot".to_string()),
        metadata: BTreeMap::from([("source".into(), serde_json::json!("fixtures"))]),
        ..Default::default()
    });
    for (i, r) in corpus().iter().enumerate() {
        let emb = (dim > 0).then(|| one_hot(i, dim));
        w.add(r, emb.as_deref()).unwrap();
    }
    w.commit(root).unwrap()
}

#[test]
fn keyword_only_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let info = write_index(dir.path(), 0);
    assert_eq!((info.generation, info.n_chunks), (1, 4));
    assert!(index::exists(dir.path()));

    let r = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();
    assert_eq!(r.len(), 4);
    assert!(!r.has_vectors());
    assert_eq!(r.chunk(2).unwrap(), corpus()[2]);
    assert_eq!(r.manifest().metadata["source"], "fixtures");

    let hits = r.search_bm25("auth timeout", 3);
    assert_eq!(hits[0].0, 2);
    // Headings are indexed too: "windows" only appears in a heading.
    assert_eq!(r.search_bm25("windows", 1)[0].0, 0);
    assert!(r.search_vector(&[1.0], 1, 1).is_err());
}

#[test]
fn hybrid_search_fuses_both_retrievers() {
    let dir = tempfile::tempdir().unwrap();
    write_index(dir.path(), 16);
    let r = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();

    // Vector query aimed at chunk 3, keyword query aimed at chunk 3 too.
    let hits = r
        .search(
            "postgres pool",
            Some(&one_hot(3, 16)),
            &SearchOpts::default(),
        )
        .unwrap();
    assert_eq!(hits[0].id, 3);
    assert_eq!(hits[0].bm25_rank, Some(1));
    assert_eq!(hits[0].vector_rank, Some(1));

    // Keyword-free query still works through vectors alone.
    let hits = r
        .search("zzz", Some(&one_hot(1, 16)), &SearchOpts::default())
        .unwrap();
    assert_eq!(hits[0].id, 1);
    assert_eq!(hits[0].bm25_rank, None);

    assert!(matches!(
        r.search("x", Some(&[1.0; 3]), &SearchOpts::default()),
        Err(Error::Dimension { .. })
    ));
}

#[test]
fn mmap_and_owned_reads_agree() {
    let dir = tempfile::tempdir().unwrap();
    write_index(dir.path(), 16);
    let mapped = IndexReader::open(dir.path(), OpenOptions { mmap: true }).unwrap();
    let owned = IndexReader::open(dir.path(), OpenOptions { mmap: false }).unwrap();
    let q = one_hot(2, 16);
    let opts = SearchOpts::default();
    assert_eq!(
        mapped.search("install", Some(&q), &opts).unwrap(),
        owned.search("install", Some(&q), &opts).unwrap()
    );
}

#[test]
fn from_parts_matches_disk() {
    let dir = tempfile::tempdir().unwrap();
    let info = write_index(dir.path(), 16);
    let read = |name: &str| fs::read(info.dir.join(name)).unwrap();
    let r = IndexReader::from_parts(
        &read(index::MANIFEST_FILE),
        read(index::DOCS_FILE),
        read(index::BM25_FILE),
        Some(read(index::VECTORS_FILE)),
    )
    .unwrap();
    assert_eq!(r.search_bm25("apt", 1)[0].0, 1);
}

#[test]
fn commits_create_generations_and_clean_up_old_ones() {
    let dir = tempfile::tempdir().unwrap();
    let first = write_index(dir.path(), 0);
    let reader_on_first = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();

    let second = write_index(dir.path(), 0);
    assert_eq!(second.generation, 2);
    assert_eq!(
        index::current_generation_dir(dir.path()).unwrap(),
        second.dir
    );
    // The previous generation is kept, so readers that opened it keep working.
    assert!(first.dir.exists());
    assert_eq!(reader_on_first.search_bm25("winget", 1)[0].0, 0);
    drop(reader_on_first);

    let third = write_index(dir.path(), 0);
    assert_eq!(third.generation, 3);
    assert!(!first.dir.exists(), "generation 1 should be removed");
    assert!(second.dir.exists(), "previous generation is kept");
    // No temporary files are left behind.
    let leftovers: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains("tmp"))
        .collect();
    assert!(leftovers.is_empty());
}

#[test]
fn writer_rejects_missing_or_unexpected_embeddings() {
    let r = record("a.md", &[], "text");
    let mut with_dim = IndexWriter::new(WriterOptions {
        dim: 4,
        ..Default::default()
    });
    assert!(with_dim.add(&r, None).is_err());
    let mut keyword_only = IndexWriter::new(WriterOptions::default());
    assert!(keyword_only.add(&r, Some(&[1.0; 4])).is_err());
}

#[test]
fn corrupt_or_foreign_indexes_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let info = write_index(dir.path(), 16);

    // Truncated file: size no longer matches the manifest.
    let bm25 = info.dir.join(index::BM25_FILE);
    let bytes = fs::read(&bm25).unwrap();
    fs::write(&bm25, &bytes[..bytes.len() - 4]).unwrap();
    assert!(matches!(
        IndexReader::open(dir.path(), OpenOptions::default()),
        Err(Error::Format(_))
    ));
    fs::write(&bm25, &bytes).unwrap();

    // Newer schema version.
    let manifest = info.dir.join(index::MANIFEST_FILE);
    let text = fs::read_to_string(&manifest).unwrap();
    fs::write(
        &manifest,
        text.replace("\"schema_version\": 1", "\"schema_version\": 99"),
    )
    .unwrap();
    assert!(matches!(
        IndexReader::open(dir.path(), OpenOptions::default()),
        Err(Error::Schema { found: 99, .. })
    ));

    // No index at all.
    let empty = tempfile::tempdir().unwrap();
    assert!(!index::exists(empty.path()));
    assert!(matches!(
        IndexReader::open(empty.path(), OpenOptions::default()),
        Err(Error::Io { .. })
    ));
}

#[test]
fn empty_index_is_valid() {
    let dir = tempfile::tempdir().unwrap();
    IndexWriter::new(WriterOptions {
        dim: 8,
        ..Default::default()
    })
    .commit(dir.path())
    .unwrap();
    let r = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();
    assert!(r.is_empty());
    assert!(
        r.search("anything", Some(&[1.0; 8]), &SearchOpts::default())
            .unwrap()
            .is_empty()
    );
}

/// Headers that claim absurd sizes must be rejected cleanly, never by
/// overflowing arithmetic or panicking (important on 32-bit WebAssembly).
#[test]
fn hostile_headers_are_rejected_without_panicking() {
    use notelight_core::bm25::Bm25;
    use notelight_core::index::Docs;

    let header = |magic: &[u8; 4], a: u32, b: u32| {
        let mut v = magic.to_vec();
        v.extend_from_slice(&[1, 0, 0, 0]);
        v.extend_from_slice(&a.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
        v
    };
    assert!(Docs::from_bytes(header(b"NLDS", u32::MAX, 0)).is_err());
    assert!(Bm25::from_bytes(header(b"NLBM", u32::MAX, u32::MAX)).is_err());

    // A docs file whose final offset is near u64::MAX.
    let mut docs = header(b"NLDS", 0, 0);
    docs.extend_from_slice(&(u64::MAX - 3).to_le_bytes());
    assert!(Docs::from_bytes(docs).is_err());
}

#[test]
fn leftovers_from_failed_commits_are_swept_and_live_generation_kept() {
    let dir = tempfile::tempdir().unwrap();
    write_index(dir.path(), 0);
    let live = write_index(dir.path(), 0); // gen 2 is live

    // Simulate failed commits: an orphan generation that was renamed into
    // place but never made live, a stale temp dir, and a stale pointer file.
    let orphan = dir.path().join(index::generation_dir_name(3));
    fs::create_dir(&orphan).unwrap();
    let stale_tmp = dir.path().join(".tmp-gen-000003-1-123");
    fs::create_dir(&stale_tmp).unwrap();
    fs::write(stale_tmp.join("docs.bin"), b"partial").unwrap();
    let stale_pointer = dir.path().join("CURRENT.tmp-1-123");
    fs::write(&stale_pointer, b"gen-000003\n").unwrap();

    let next = write_index(dir.path(), 0);
    assert_eq!(next.generation, 3);
    // The generation that was live before this commit is kept for open readers.
    assert!(live.dir.exists());
    assert!(!stale_tmp.exists());
    assert!(!stale_pointer.exists());
    let r = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();
    assert_eq!(r.manifest().generation, 3);
    assert_eq!(r.len(), 4);
}
