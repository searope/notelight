"""Smoke tests for the Rust extension: types, shapes and agreement with NumPy."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

import notelight
from notelight.cli import app


def numpy_reference(vectors: np.ndarray) -> np.ndarray:
    """Pure-NumPy binary quantization with the same little-endian bit order."""
    return np.packbits(vectors > 0, axis=1, bitorder="little")


def test_version_matches_core() -> None:
    assert notelight.__version__ == notelight.core_version


@pytest.mark.parametrize("dim", [8, 10, 384, 1024])
def test_quantize_binary_matches_numpy(dim: int) -> None:
    rng = np.random.default_rng(seed=dim)
    vectors = rng.standard_normal((50, dim)).astype(np.float32)

    codes = notelight.quantize_binary(vectors)

    assert codes.dtype == np.uint8
    assert codes.shape == (50, (dim + 7) // 8)
    np.testing.assert_array_equal(codes, numpy_reference(vectors))


def test_hamming_matches_numpy() -> None:
    rng = np.random.default_rng(seed=7)
    codes = notelight.quantize_binary(rng.standard_normal((2, 384)).astype(np.float32))
    expected = int(np.unpackbits(codes[0] ^ codes[1]).sum())
    assert notelight.hamming(codes[0], codes[1]) == expected


def test_hamming_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        notelight.hamming(np.zeros(4, np.uint8), np.zeros(5, np.uint8))


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"notelight {notelight.__version__}" in result.output
