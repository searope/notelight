"""Type stubs for the compiled Rust extension (crates/notelight-py)."""

import numpy as np
import numpy.typing as npt

__version__: str

def quantize_binary(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Quantize float32 vectors of shape (n, dim) to packed bits, shape (n, ceil(dim / 8))."""

def hamming(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> int:
    """Hamming distance between two packed binary codes of equal length."""
