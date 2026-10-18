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
    filled_writer(dim).commit(root).unwrap()
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
fn keyword_hits_count_by_the_share_of_terms_they_contain() {
    let dir = tempfile::tempdir().unwrap();
    write_index(dir.path(), 16);
    let r = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();
    // One result per retriever: BM25 finds chunk 0 on "restart" alone (1 of
    // 3 terms), the vector points at chunk 3.
    let one = SearchOpts {
        k: 1,
        depth: 1,
        ..SearchOpts::default()
    };
    let plain = SearchOpts {
        coverage_power: 0.0,
        ..one
    };
    let q = one_hot(3, 16);
    // Plain RRF: two first places tie, and the lower id wins.
    assert_eq!(
        ids(&r.search("restart zebra yak", Some(&q), &plain).unwrap()),
        [0]
    );
    // Weighted: chunk 0 counts (1/3)^2 of a first place.
    let hits = r.search("restart zebra yak", Some(&q), &one).unwrap();
    assert_eq!(ids(&hits), [3]);
    // A hit with every term counts fully, so nothing changes.
    assert_eq!(
        r.search("winget restart", Some(&q), &one).unwrap(),
        r.search("winget restart", Some(&q), &plain).unwrap()
    );
    // Without vectors BM25's order stands.
    let all = SearchOpts::default();
    let no_weights = SearchOpts {
        coverage_power: 0.0,
        ..all
    };
    assert_eq!(
        r.search("install debian zebra", None, &all).unwrap(),
        r.search("install debian zebra", None, &no_weights).unwrap()
    );
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

/// A writer filled with the corpus, not yet committed.
fn filled_writer(dim: usize) -> IndexWriter {
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
    w
}

#[test]
fn build_in_memory_matches_commit() {
    // The browser build has no filesystem: it builds and opens generations
    // from bytes. Those bytes must be exactly what `commit` writes.
    let dir = tempfile::tempdir().unwrap();
    let info = filled_writer(16).commit(dir.path()).unwrap();
    let built = filled_writer(16).build(info.generation).unwrap();
    let read = |name: &str| fs::read(info.dir.join(name)).unwrap();
    assert_eq!(built.docs, read(index::DOCS_FILE));
    assert_eq!(built.bm25, read(index::BM25_FILE));
    assert_eq!(
        built.vectors.as_deref(),
        Some(&read(index::VECTORS_FILE)[..])
    );
    let on_disk: serde_json::Value = serde_json::from_slice(&read(index::MANIFEST_FILE)).unwrap();
    let in_memory: serde_json::Value =
        serde_json::from_slice(&built.manifest.to_json().unwrap()).unwrap();
    assert_eq!(on_disk, in_memory);

    let r = IndexReader::from_parts(
        &built.manifest.to_json().unwrap(),
        built.docs,
        built.bm25,
        built.vectors,
    )
    .unwrap();
    assert_eq!(r.search_bm25("winget", 1)[0].0, 0);
}

#[test]
fn from_parts_checks_sizes_against_manifest() {
    let built = filled_writer(0).build(1).unwrap();
    let manifest = built.manifest.to_json().unwrap();
    let mut short_docs = built.docs.clone();
    short_docs.pop();
    let err = IndexReader::from_parts(&manifest, short_docs, built.bm25.clone(), None).unwrap_err();
    assert!(err.to_string().contains("docs.bin"), "{err}");

    // Vectors handed to a keyword-only manifest have no recorded size.
    let err =
        IndexReader::from_parts(&manifest, built.docs, built.bm25, Some(vec![0; 64])).unwrap_err();
    assert!(err.to_string().contains("vectors.bin"), "{err}");
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
        text.replace(
            &format!("\"schema_version\": {}", index::SCHEMA_VERSION),
            "\"schema_version\": 99",
        ),
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
    assert_eq!(
        r.search("anything", Some(&[1.0; 8]), &SearchOpts::default())
            .unwrap(),
        [] as [notelight_core::index::Hit; 0]
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

/// A keyword-only index over `texts`, one chunk each, with no headings.
fn phrase_index(root: &std::path::Path, texts: &[&str]) -> IndexReader {
    let mut w = IndexWriter::new(WriterOptions::default());
    for (i, text) in texts.iter().enumerate() {
        w.add(&record(&format!("{i}.md"), &[], text), None).unwrap();
    }
    w.commit(root).unwrap();
    IndexReader::open(root, OpenOptions::default()).unwrap()
}

fn ids(hits: &[index::Hit]) -> Vec<u32> {
    hits.iter().map(|h| h.id).collect()
}

const SCATTERED_FIRST: &[&str] = &[
    // Both terms, repeated, but never next to each other: BM25's favourite.
    "The lock is held while the commit runs. A lock protects each file. File \
     handles and lock state are checked; every file is closed.",
    "Other notes about the build cache.",
    // The exact phrase, once.
    "One writer, enforced with a lock file in the index root.",
];

#[test]
fn in_order_terms_get_a_bonus() {
    let dir = tempfile::tempdir().unwrap();
    let r = phrase_index(dir.path(), SCATTERED_FIRST);
    let off = SearchOpts {
        phrase_boost: 0.0,
        ..SearchOpts::default()
    };
    assert_eq!(ids(&r.search("lock file", None, &off).unwrap())[0], 0);
    let hits = r.search("lock file", None, &SearchOpts::default()).unwrap();
    assert_eq!(ids(&hits)[..2], [2, 0]);
    // The bonus is one extra first place: 1 / (rrf_k + 1).
    let base = r.search("lock file", None, &off).unwrap();
    let before = base.iter().find(|h| h.id == 2).unwrap().score;
    assert!((hits[0].score - before - 1.0 / 61.0).abs() < 1e-6);
    // Reversed terms are not in order anywhere, so nothing moves.
    let reversed = r.search("file lock", None, &SearchOpts::default()).unwrap();
    assert_eq!(
        ids(&reversed),
        ids(&r.search("file lock", None, &off).unwrap())
    );
}

#[test]
fn quoted_phrases_rank_first_even_without_the_bonus() {
    let dir = tempfile::tempdir().unwrap();
    let r = phrase_index(dir.path(), SCATTERED_FIRST);
    let off = SearchOpts {
        phrase_boost: 0.0,
        ..SearchOpts::default()
    };
    // Quotes can be curly, and the phrase ignores case and punctuation.
    for query in [r#""lock file""#, "“Lock-File”", r#"commit "lock file""#] {
        let hits = r.search(query, None, &off).unwrap();
        assert_eq!(hits[0].id, 2, "{query}");
    }
    // A quoted phrase nothing contains leaves the fused order alone.
    assert_eq!(
        ids(&r.search(r#""lock commit""#, None, &off).unwrap()),
        ids(&r.search("lock commit", None, &off).unwrap())
    );
}

#[test]
fn every_quoted_phrase_must_match_and_k_still_applies() {
    let dir = tempfile::tempdir().unwrap();
    let r = phrase_index(
        dir.path(),
        &[
            "lock file and commit protocol",
            "lock file only, lock file again",
            "commit protocol only",
        ],
    );
    let hits = r
        .search(
            r#""lock file" "commit protocol""#,
            None,
            &SearchOpts {
                k: 1,
                ..SearchOpts::default()
            },
        )
        .unwrap();
    assert_eq!(ids(&hits), [0]);
}

#[test]
fn phrases_match_headings_too() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = IndexWriter::new(WriterOptions::default());
    w.add(
        &record("a.md", &[], "lock here, file there, lock file nowhere"),
        None,
    )
    .unwrap();
    w.add(&record("b.md", &["Writer lock"], "File handles."), None)
        .unwrap();
    w.commit(dir.path()).unwrap();
    let r = IndexReader::open(dir.path(), OpenOptions::default()).unwrap();
    // "Writer lock" + "\n" + "File handles." reads as "writer lock file handles".
    let hits = r
        .search(r#""writer lock""#, None, &SearchOpts::default())
        .unwrap();
    assert_eq!(hits[0].id, 1);
}
