"""Binary traces (float16 little-endian) and 2D coordinates of the nodes per stage.

:func:`write_binary` stores tensors one after the other (row-major) and returns the index
(name, byte offset, shape) that goes in the JSON; :func:`read_tensor` reads one back.

:func:`aligned_pca` projects the node vectors of each stage (input attributes, input layer,
every message-passing layer, Jumping Knowledge) to 2D with PCA, and aligns each projection
with the previous one by orthogonal Procrustes (rotation or reflection, no scaling), so that
an animation from one stage to the next does not jump because PCA flipped or rotated its
axes. Every stage is centred and divided by the RMS norm of its points, so the stages share
one scale.
"""

from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from qcclass.demo.esquema import TraceTensor

DTYPE = np.dtype("<f2")


def write_binary(tensors: dict[str, np.ndarray], path: Path) -> list[TraceTensor]:
    """Write ``tensors`` (in order) as float16 little-endian; return their index."""
    index, offset, chunks = [], 0, []
    for name, values in tensors.items():
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float32).astype(DTYPE))
        index.append(TraceTensor(name=name, offset=offset, shape=list(array.shape)))
        chunks.append(array.tobytes(order="C"))
        offset += array.nbytes
    Path(path).write_bytes(b"".join(chunks))
    return index


def read_tensor(path: Path, entry: TraceTensor) -> np.ndarray:
    """Read one tensor of a binary written by :func:`write_binary` (as float32)."""
    count = int(np.prod(entry.shape)) if entry.shape else 1
    data = np.fromfile(path, dtype=DTYPE, count=count, offset=entry.offset)
    return data.reshape(entry.shape).astype(np.float32)


def _project(values: np.ndarray, seed: int) -> tuple[np.ndarray, list[float]]:
    """2D PCA of the rows (padded with zeros when there are fewer than 2 components)."""
    n, features = values.shape
    components = min(2, n, features)
    points = np.zeros((n, 2))
    variance = [0.0, 0.0]
    if components > 0 and n > 1 and float(values.std(axis=0).sum()) > 0:
        pca = PCA(n_components=components, random_state=seed)
        points[:, :components] = pca.fit_transform(values)
        variance[:components] = [float(v) for v in pca.explained_variance_ratio_]
    points -= points.mean(axis=0)
    scale = float(np.sqrt((points**2).sum(axis=1).mean()))
    return (points / scale if scale > 0 else points), variance


def aligned_pca(
    stages: dict[str, np.ndarray], seed: int = 0
) -> list[tuple[str, np.ndarray, list[float]]]:
    """2D coordinates of every stage, each aligned with the previous one (module docstring)."""
    result: list[tuple[str, np.ndarray, list[float]]] = []
    previous = None
    for name, values in stages.items():
        points, variance = _project(np.asarray(values, dtype=np.float64), seed)
        if previous is not None and np.abs(points).sum() > 0 and np.abs(previous).sum() > 0:
            rotation, _ = orthogonal_procrustes(points, previous)
            points = points @ rotation
        result.append((name, points, variance))
        previous = points
    return result
