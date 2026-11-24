"""Feature rungs of the baseline: size only (E7), F3 and Weisfeiler-Lehman (WL).

E7 and F3 come from the F3 Parquet file. WL is computed from the F4 dependency graphs:

* initial node label: gate type, plus for ``rz`` its angle on a grid of pi/``divisiones_pi``
  (``k * pi / divisiones_pi`` within ``tolerancia_angulo``) or the category "other"; the angle
  is recovered from the node's sine and cosine;
* edge label: the four control/target role indicators of F4 (16 values);
* iteration: the new label of a node hashes its label together with the multiset of
  ``(edge label, direction, neighbour label)`` over its outgoing and incoming edges, so the
  direction of the dependency is kept;
* features: for every iteration ``0..h`` the labels are hashed into ``2**bits_hash`` buckets
  (the iteration is part of the hash) and counted per circuit, divided by its number of
  nodes. The hashing never builds a vocabulary, so it cannot look at validation data.

Hashing is deterministic (splitmix64 on 64-bit integers, never Python's ``hash``) and fully
vectorised with NumPy; the multiset is hashed as the wrapping sum of the hashed tuples. The
per-iteration matrices of a set of circuits are cached under ``<resultados>/baseline/cache``,
keyed by the dataset fingerprint, the parameters and the identifiers.
"""

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from qcclass.config import DatasetConfig
from qcclass.formatos.construir import DAG_DIR, FEATURES_FILE
from qcclass.formatos.grafo_dag import load_graph
from qcclass.versionado import version_stamp

logger = logging.getLogger(__name__)

GATES = ("rz", "sx", "x", "cx")  # order of the one-hot columns of F4
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


@dataclass(frozen=True)
class WLSettings:
    """Parameters of the WL features."""

    h_max: int
    bits_hash: int
    divisiones_pi: int
    tolerancia_angulo: float

    @property
    def dimension(self) -> int:
        """Number of hash buckets."""
        return 2**self.bits_hash


@dataclass(frozen=True)
class WLFeatures:
    """Per-iteration WL count matrices (``n x 2**bits``, normalised) and bucket names."""

    ids: tuple[str, ...]
    iterations: tuple[sp.csr_matrix, ...]  # iteration 0 .. h_max
    bucket_names: dict[int, str]  # buckets holding exactly one initial label and nothing else

    def matrix(self, h: int) -> sp.csr_matrix:
        """Features with iterations ``0..h`` (sum of their matrices)."""
        total = self.iterations[0].copy()
        for matrix in self.iterations[1 : h + 1]:
            total = total + matrix
        return total.tocsr()


# --------------------------------------------------------------------------------------------
# E7 and F3
# --------------------------------------------------------------------------------------------


def f3_table(
    config: DatasetConfig, ids: Sequence[str], columns: Sequence[str] | None
) -> pd.DataFrame:
    """F3 features of ``ids`` (in that order); ``columns`` = ``None`` for all of them.

    Raises:
        KeyError: if a circuit or a column is missing from F3.
    """
    features = pd.read_parquet(config.rutas.features / FEATURES_FILE).set_index("id")
    if missing := [i for i in ids if i not in features.index][:5]:
        raise KeyError(f"circuits without F3 features: {missing}")
    selected = features.loc[list(ids)]
    return selected if columns is None else selected[list(columns)]


# --------------------------------------------------------------------------------------------
# WL hashing
# --------------------------------------------------------------------------------------------


def _mix(values: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised (uint64, wrapping arithmetic by design)."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _combine(*parts: np.ndarray) -> np.ndarray:
    """Order-dependent hash of several uint64 arrays (or scalars)."""
    state = np.uint64(0x84222325CBF29CE4)
    for part in parts:
        state = _mix(np.asarray(state, dtype=np.uint64) ^ np.asarray(part, dtype=np.uint64))
    return state


def initial_labels(x: np.ndarray, settings: WLSettings) -> np.ndarray:
    """Initial label code of every node: gate index * (bins + 1) + angle bin (rz only).

    Angle bins are ``0 .. 2 * divisiones_pi - 1`` for multiples of pi/``divisiones_pi`` and
    ``2 * divisiones_pi`` for any other angle; non-``rz`` gates use bin 0.
    """
    gate = x[:, : len(GATES)].argmax(axis=1).astype(np.int64)
    angle = np.arctan2(x[:, len(GATES)], x[:, len(GATES) + 1]).astype(np.float64)
    step = math.pi / settings.divisiones_pi
    multiple = np.round(angle / step)
    on_grid = np.abs(angle - multiple * step) < settings.tolerancia_angulo
    n_bins = 2 * settings.divisiones_pi
    bins = np.where(on_grid, np.mod(multiple, n_bins), n_bins).astype(np.int64)
    bins = np.where(gate == GATES.index("rz"), bins, 0)
    return gate * (n_bins + 1) + bins


def initial_label_name(code: int, settings: WLSettings) -> str:
    """Readable name of an initial label (``rz(3pi/16)``, ``rz(other)``, ``cx``...)."""
    n_bins = 2 * settings.divisiones_pi
    gate, angle_bin = divmod(int(code), n_bins + 1)
    if GATES[gate] != "rz":
        return GATES[gate]
    if angle_bin == n_bins:
        return "rz(otro)"
    k = angle_bin if angle_bin < settings.divisiones_pi else angle_bin - n_bins  # (-pi, pi]
    return f"rz({k}pi/{settings.divisiones_pi})"


def wl_label_sequence(
    x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, settings: WLSettings
) -> list[np.ndarray]:
    """Hashed node labels of iterations ``0..h_max`` of one graph (uint64 arrays)."""
    labels = _combine(np.uint64(0), initial_labels(x, settings).astype(np.uint64))
    sequence = [labels]
    if edge_index.size == 0:
        return sequence * (settings.h_max + 1)
    source, target = edge_index[0], edge_index[1]
    roles = edge_attr.astype(np.int64) @ np.array([8, 4, 2, 1], dtype=np.int64)
    for _ in range(settings.h_max):
        outgoing = _combine(roles, np.uint64(1), labels[target])
        incoming = _combine(roles, np.uint64(2), labels[source])
        aggregate = np.zeros(len(labels), dtype=np.uint64)
        with np.errstate(over="ignore"):
            np.add.at(aggregate, source, outgoing)  # wrapping sum: a multiset hash
            np.add.at(aggregate, target, incoming)
        labels = _combine(labels, aggregate)
        sequence.append(labels)
    return sequence


def _bucket(labels: np.ndarray, iteration: int, bits: int) -> np.ndarray:
    hashed = _combine(np.uint64(iteration + 1), labels)
    return (hashed & np.uint64(2**bits - 1)).astype(np.int64)


# --------------------------------------------------------------------------------------------
# WL features of a set of circuits (with cache)
# --------------------------------------------------------------------------------------------


def _cache_path(
    config: DatasetConfig, ids: Sequence[str], settings: WLSettings, folder: Path
) -> Path:
    key = json.dumps(
        {
            "huella": version_stamp(config)["huella_dataset"],
            "ajustes": settings.__dict__,
            "ids": list(ids),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return folder / f"wl_{digest}.npz"


def _save(path: Path, features: WLFeatures) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": np.array(features.ids), "names": np.array(json.dumps(features.bucket_names))}
    for index, matrix in enumerate(features.iterations):
        arrays |= {
            f"data{index}": matrix.data,
            f"indices{index}": matrix.indices,
            f"indptr{index}": matrix.indptr,
        }
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _load(path: Path, settings: WLSettings) -> WLFeatures:
    content = np.load(path, allow_pickle=False)
    ids = tuple(str(i) for i in content["ids"])
    shape = (len(ids), settings.dimension)
    iterations = tuple(
        sp.csr_matrix(
            (content[f"data{i}"], content[f"indices{i}"], content[f"indptr{i}"]), shape=shape
        )
        for i in range(settings.h_max + 1)
    )
    names = {int(k): v for k, v in json.loads(str(content["names"])).items()}
    return WLFeatures(ids, iterations, names)


def wl_features(
    config: DatasetConfig, ids: Sequence[str], settings: WLSettings, cache_folder: Path
) -> WLFeatures:
    """WL features of ``ids`` (rows in that order), from the cache when available."""
    path = _cache_path(config, ids, settings, cache_folder)
    if path.is_file():
        logger.info("WL features from cache %s", path.name)
        return _load(path, settings)
    rows: list[list[np.ndarray]] = [[] for _ in range(settings.h_max + 1)]
    initial_codes: dict[int, set[int]] = {}  # iteration-0 bucket -> initial label codes
    hit_later = np.zeros(settings.dimension, dtype=bool)  # buckets used by iterations >= 1
    for circuit_id in tqdm(ids, desc="WL"):
        graph = load_graph(config.rutas.graphs / DAG_DIR / f"{circuit_id}.pt")
        x = graph.x.numpy()
        sequence = wl_label_sequence(x, graph.edge_index.numpy(), graph.edge_attr.numpy(), settings)
        for iteration, labels in enumerate(sequence):
            buckets = _bucket(labels, iteration, settings.bits_hash)
            rows[iteration].append(buckets)
            if iteration:
                hit_later[buckets] = True
        if len(x):
            pairs = np.unique(np.stack([rows[0][-1], initial_labels(x, settings)]), axis=1)
            for bucket, code in pairs.T:
                initial_codes.setdefault(int(bucket), set()).add(int(code))
    names = {  # only buckets that hold exactly one initial label and nothing else
        bucket: initial_label_name(next(iter(codes)), settings)
        for bucket, codes in initial_codes.items()
        if len(codes) == 1 and not hit_later[bucket]
    }
    iterations = tuple(_count_matrix(buckets, settings.dimension) for buckets in rows)
    features = WLFeatures(tuple(ids), iterations, names)
    _save(path, features)
    return features


def _count_matrix(buckets_per_row: list[np.ndarray], dimension: int) -> sp.csr_matrix:
    """Normalised bucket counts (one row per circuit; empty circuits give a zero row)."""
    lengths = np.array([len(b) for b in buckets_per_row])
    row_index = np.repeat(np.arange(len(buckets_per_row)), lengths)
    columns = np.concatenate(buckets_per_row) if len(row_index) else np.zeros(0, dtype=np.int64)
    weights = np.repeat(1.0 / np.maximum(lengths, 1), lengths)
    matrix = sp.csr_matrix((weights, (row_index, columns)), shape=(len(buckets_per_row), dimension))
    matrix.sum_duplicates()
    return matrix


def bucket_name(bucket: int, features: WLFeatures) -> str:
    """Readable name of a hashed WL feature: its initial label when the bucket holds only
    that label, otherwise ``wl#<bucket>``."""
    return features.bucket_names.get(bucket, f"wl#{bucket}")
