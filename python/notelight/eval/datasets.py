"""BEIR datasets: pinned downloads, a cache, and a loader for the BEIR layout.

A BEIR dataset is a folder with three parts::

    corpus.jsonl        {"_id": "...", "title": "...", "text": "..."} per line
    queries.jsonl       {"_id": "...", "text": "..."} per line
    qrels/<split>.tsv   query-id <TAB> corpus-id <TAB> score, with a header row

The three datasets notelight reports on are downloaded once from the BEIR
authors' server, checked against a pinned SHA-256, and unpacked into
`paths.eval_cache_dir()`. Any other folder in the same layout can be
evaluated by passing its path instead of a name.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from notelight import paths
from notelight.download import download_verified
from notelight.errors import DatasetNotAvailableError

__all__ = [
    "BEIR_DATASETS",
    "Dataset",
    "DatasetSpec",
    "ensure_dataset",
    "load_beir",
    "load_dataset",
]

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    """A zipped BEIR dataset, pinned by size and SHA-256."""

    name: str
    sha256: str
    size: int
    split: str = "test"
    base_url: str = BEIR_BASE_URL

    @property
    def url(self) -> str:
        """Download URL of the zip file."""
        return f"{self.base_url}/{self.name}.zip"


#: The datasets in the README's results table: small enough to embed on a
#: laptop CPU, and covering science claims, medical questions and finance.
BEIR_DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            "scifact",
            "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
            2_816_079,
        ),
        DatasetSpec(
            "nfcorpus",
            "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
            2_448_432,
        ),
        DatasetSpec(
            "fiqa",
            "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
            17_948_027,
        ),
    )
}


@dataclass
class Dataset:
    """A loaded dataset: the corpus in file order, the queries that have
    relevance judgments in the split, and the judgments themselves."""

    name: str
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    #: query id -> query text, in file order.
    queries: dict[str, str]
    #: query id -> {doc id: graded relevance}; only grades above zero.
    qrels: dict[str, dict[str, int]]
    #: Hash of the corpus, queries and qrels files; keys cached embeddings.
    fingerprint: str = field(default="")


def _safe_extract(archive: Path, dest: Path) -> None:
    """Unzip `archive` into `dest`, refusing members that would land outside it."""
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or ":" in member.filename:
                raise DatasetNotAvailableError(
                    f"{archive.name}: unsafe path {member.filename!r} in archive"
                )
        zf.extractall(dest)


def ensure_dataset(
    spec: DatasetSpec,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    on_download: Callable[[DatasetSpec], None] | None = None,
) -> Path:
    """Return the folder of an unpacked dataset, downloading it if needed.

    The zip is verified against `spec.sha256` before it is unpacked, and the
    folder is unpacked under a temporary name and renamed into place, so a
    folder in the cache is always complete.

    Raises:
        DatasetNotAvailableError: the dataset is not cached and `offline` is
            set, the download failed verification, or the archive is unsafe.
    """
    root = (cache_dir if cache_dir is not None else paths.eval_cache_dir()) / "datasets"
    folder = root / spec.name
    if (folder / "corpus.jsonl").is_file():
        return folder
    archive = root / f"{spec.name}.zip"
    if not (archive.is_file() and archive.stat().st_size == spec.size):
        if offline:
            raise DatasetNotAvailableError(
                f"dataset {spec.name} is not in the cache ({root}) and --offline is set; "
                "run once without --offline to download it"
            )
        if on_download is not None:
            on_download(spec)
        try:
            download_verified(spec.url, archive, spec.sha256, error=DatasetNotAvailableError)
        except OSError as e:
            raise DatasetNotAvailableError(f"downloading {spec.url} failed: {e}") from e
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".unpack-{spec.name}-"))
    try:
        _safe_extract(archive, staging)
        # BEIR zips hold one top-level folder named after the dataset.
        inner = staging / spec.name
        if not (inner / "corpus.jsonl").is_file():
            raise DatasetNotAvailableError(f"{archive.name}: no {spec.name}/corpus.jsonl inside")
        inner.rename(folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return folder


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir(folder: Path, *, split: str = "test", name: str | None = None) -> Dataset:
    """Load a dataset in the BEIR layout.

    Only queries with at least one positive judgment in `split` are kept,
    because recall is undefined without one. Judged documents missing from
    the corpus still count as relevant, as in the BEIR reference evaluation.

    Raises:
        DatasetNotAvailableError: a file is missing or malformed.
    """
    corpus_path = folder / "corpus.jsonl"
    queries_path = folder / "queries.jsonl"
    qrels_path = folder / "qrels" / f"{split}.tsv"
    for p in (corpus_path, queries_path, qrels_path):
        if not p.is_file():
            raise DatasetNotAvailableError(f"{folder} is not a BEIR dataset: {p.name} is missing")
    try:
        corpus = _read_jsonl(corpus_path)
        all_queries = {q["_id"]: q["text"] for q in _read_jsonl(queries_path)}
        qrels: dict[str, dict[str, int]] = {}
        with qrels_path.open(encoding="utf-8", newline="") as f:
            rows = csv.reader(f, delimiter="\t")
            next(rows, None)  # header
            for row in rows:
                if not row:
                    continue
                qid, did, score = row[0], row[1], int(row[2])
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score
        doc_ids = [d["_id"] for d in corpus]
        titles = [d.get("title", "") for d in corpus]
        texts = [d["text"] for d in corpus]
    except (KeyError, IndexError, ValueError) as e:
        raise DatasetNotAvailableError(f"{folder}: malformed BEIR file ({e!r})") from e

    queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
    qrels = {qid: rel for qid, rel in qrels.items() if qid in queries}
    h = hashlib.sha256()
    for p in (corpus_path, queries_path, qrels_path):
        h.update(p.read_bytes())
    return Dataset(
        name=name or folder.name,
        doc_ids=doc_ids,
        titles=titles,
        texts=texts,
        queries=queries,
        qrels=qrels,
        fingerprint=h.hexdigest()[:16],
    )


def load_dataset(
    name_or_path: str,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    on_download: Callable[[DatasetSpec], None] | None = None,
) -> Dataset:
    """Load a pinned BEIR dataset by name, or a BEIR-layout folder by path.

    Raises:
        DatasetNotAvailableError: unknown name, missing folder, or a failed download.
    """
    spec = BEIR_DATASETS.get(name_or_path)
    if spec is not None:
        folder = ensure_dataset(spec, cache_dir=cache_dir, offline=offline, on_download=on_download)
        return load_beir(folder, split=spec.split, name=spec.name)
    folder = Path(name_or_path)
    if not folder.is_dir():
        known = ", ".join(BEIR_DATASETS)
        raise DatasetNotAvailableError(
            f"{name_or_path!r} is neither a known dataset ({known}) nor a folder"
        )
    return load_beir(folder)
