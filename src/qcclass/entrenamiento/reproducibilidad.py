"""Global seeds and random-number-generator states (saved in every checkpoint).

:func:`set_global_seeds` seeds Python's ``random``, NumPy's global generator and torch (CPU
and CUDA). :func:`rng_state` / :func:`restore_rng_state` capture and restore all of them as
plain lists and tensors, so that a checkpoint can be loaded with ``weights_only=True``.
"""

import random

import numpy as np
import torch


def set_global_seeds(seed: int, *, deterministic: bool = False) -> None:
    """Seed every generator; ``deterministic`` also asks torch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def rng_state() -> dict:
    """State of every generator (Python, NumPy, torch CPU and CUDA)."""
    name, keys, position, has_gauss, gauss = np.random.get_state()
    python_version, python_internal, python_gauss = random.getstate()
    return {
        "python": [python_version, list(python_internal), python_gauss],
        "numpy": [name, keys.tolist(), int(position), int(has_gauss), float(gauss)],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Restore the generators saved by :func:`rng_state`."""
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())  # generator states must be CPU byte tensors
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
