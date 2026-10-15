"""Downloads verified by SHA-256, shared by the model and dataset caches."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from notelight.errors import NotelightError

__all__ = ["download_verified", "sha256_file"]


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_verified(
    url: str, dest: Path, sha256: str, *, error: type[NotelightError] = NotelightError
) -> None:
    """Stream `url` to a temporary file next to `dest`, verify it, then rename.

    `dest` only ever appears complete and verified: a failed or tampered
    download leaves nothing behind.

    Raises:
        error: the file's SHA-256 is not `sha256`.
        OSError: the download or the write failed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            for block in iter(lambda: resp.read(1 << 20), b""):
                h.update(block)
                out.write(block)
        digest = h.hexdigest()
        if digest != sha256:
            raise error(
                f"{dest.name}: SHA-256 mismatch (expected {sha256}, got {digest}); "
                "the download was corrupted or the file changed upstream"
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
