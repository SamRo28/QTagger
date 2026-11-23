"""qcclass: dataset generation for quantum circuit classification by algorithm."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qcclass")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"
