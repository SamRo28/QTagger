"""Deterministic seed derivation."""

import hashlib

_SEED_MASK = 0x7FFFFFFF  # 31 bits: valid for numpy, random, networkx and the Qiskit transpiler


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a reproducible 31-bit seed from a base seed and identifying parts.

    The result depends only on the textual value of the arguments, so it is stable across
    processes and Python versions (unlike the built-in ``hash``).
    """
    key = ":".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & _SEED_MASK
