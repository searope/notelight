"""notelight: private semantic search over your own files, for you and your AI agents."""

from notelight._core import __version__ as core_version
from notelight._core import hamming, quantize_binary

__version__ = core_version

__all__ = ["__version__", "core_version", "hamming", "quantize_binary"]
