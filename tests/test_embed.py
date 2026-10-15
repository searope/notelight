"""Embedders and pinned model downloads, without network access.

Downloads are exercised against `file://` URLs. The real bge-small model runs
only when NOTELIGHT_MODEL_TESTS=1 (it needs a ~130 MB one-time download; set
NOTELIGHT_MODEL_CACHE to reuse an existing model cache).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from notelight.embed import (
    BGE_SMALL_EN_V15,
    HashingEmbedder,
    ModelFile,
    ModelSpec,
    OnnxEmbedder,
    ensure_model_files,
    make_embedder,
    name_for_model_id,
)
from notelight.errors import ModelNotAvailableError


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    e = HashingEmbedder(dim=32)
    a = e.embed_documents(["the quick brown fox", "lazy dog"])
    b = e.embed_documents(["the quick brown fox", "lazy dog"])
    assert a.shape == (2, 32) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(e.embed_query("lazy dog"), a[1])


def test_hashing_embedder_shared_words_are_closer() -> None:
    e = HashingEmbedder()
    q = e.embed_query("sourdough bread")
    near, far = e.embed_documents(["bake sourdough bread", "rust ownership rules"])
    assert float(q @ near) > float(q @ far)


def test_hashing_embedder_empty_text_is_zero_vector() -> None:
    assert not HashingEmbedder(8).embed_query("").any()


def test_registry_round_trip() -> None:
    assert make_embedder("none") is None
    hashing = make_embedder("hashing")
    assert hashing is not None
    assert name_for_model_id(hashing.model_id) == "hashing"
    assert name_for_model_id(BGE_SMALL_EN_V15.model_id) == "bge-small"
    assert name_for_model_id(None) == "none"
    assert name_for_model_id("someone/else@123") is None
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("nope")


def test_bge_model_id_pins_revision() -> None:
    assert BGE_SMALL_EN_V15.model_id == "BAAI/bge-small-en-v1.5@5c38ec7c405e"


def _fake_repo(tmp_path: Path, onnx: bytes, tokenizer: bytes) -> ModelSpec:
    """A spec whose files are served from a local folder via file:// URLs."""
    rev = "0" * 40
    base = tmp_path / "remote"
    files = []
    for rel, data in (("onnx/model.onnx", onnx), ("tokenizer.json", tokenizer)):
        dest = base / "org" / "model" / "resolve" / rev / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files.append(ModelFile(rel, hashlib.sha256(data).hexdigest(), len(data)))
    return ModelSpec(
        repo="org/model",
        revision=rev,
        onnx=files[0],
        tokenizer=files[1],
        dim=4,
        base_url=base.as_uri(),
    )


def test_download_verifies_and_caches(tmp_path: Path) -> None:
    spec = _fake_repo(tmp_path, b"onnx-bytes", b"{}")
    cache = tmp_path / "cache"
    downloaded: list[str] = []

    model, tok = ensure_model_files(
        spec, cache_dir=cache, on_download=lambda f: downloaded.append(f.path)
    )
    assert model.read_bytes() == b"onnx-bytes" and tok.read_bytes() == b"{}"
    assert downloaded == ["onnx/model.onnx", "tokenizer.json"]

    # Second call: served from the cache, works offline, downloads nothing.
    downloaded.clear()
    assert ensure_model_files(spec, cache_dir=cache, offline=True) == (model, tok)
    assert downloaded == []


def test_offline_without_cache_fails(tmp_path: Path) -> None:
    spec = _fake_repo(tmp_path, b"a", b"b")
    with pytest.raises(ModelNotAvailableError, match="--offline"):
        ensure_model_files(spec, cache_dir=tmp_path / "empty", offline=True)


def test_hash_mismatch_is_rejected_and_not_cached(tmp_path: Path) -> None:
    spec = _fake_repo(tmp_path, b"real", b"{}")
    tampered = dataclasses.replace(spec, onnx=dataclasses.replace(spec.onnx, sha256="0" * 64))
    cache = tmp_path / "cache"
    with pytest.raises(ModelNotAvailableError, match="SHA-256 mismatch"):
        ensure_model_files(tampered, cache_dir=cache)
    leftovers = [p for p in cache.rglob("*") if p.is_file()]
    assert leftovers == []


@pytest.mark.skipif(
    os.environ.get("NOTELIGHT_MODEL_TESTS") != "1",
    reason="set NOTELIGHT_MODEL_TESTS=1 to download and run bge-small",
)
def test_bge_small_end_to_end() -> None:
    e = OnnxEmbedder()
    docs = e.embed_documents(
        [
            "The cat sat on the mat.",
            "Interest rates rose by a quarter point.",
            "A kitten is resting on the rug.",
        ]
    )
    assert docs.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)
    q = e.embed_query("where is the cat sleeping?")
    scores = docs @ q
    assert scores[0] > scores[1] and scores[2] > scores[1]
    # Batching and length-sorting must not change the vectors.
    alone = e.embed_documents(["A kitten is resting on the rug."])[0]
    np.testing.assert_allclose(alone, docs[2], atol=1e-4)


def test_onnx_embedder_loads_lazily(tmp_path: Path) -> None:
    # Constructing never touches the network or disk, so an up-to-date sync
    # with an uncached model and --offline still works; embedding fails clearly.
    e = OnnxEmbedder(BGE_SMALL_EN_V15, cache_dir=tmp_path / "empty", offline=True)
    assert (e.model_id, e.dim) == (BGE_SMALL_EN_V15.model_id, 384)
    assert e.embed_documents([]).shape == (0, 384)
    with pytest.raises(ModelNotAvailableError, match="--offline"):
        e.embed_query("anything")
