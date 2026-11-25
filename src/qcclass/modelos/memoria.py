"""GPU memory model of the GNN: peak memory per node of a training micro-batch.

The peak memory of a training step grows with the nodes (and edges, about two per node) of
the micro-batch: ``peak = base + nodes * per_node(config)``, where ``base`` is the memory of
the parameters, gradients and optimiser states. ``per_node`` is a linear function of
:data:`FEATURES` of the configuration, fitted by least squares on measurements of
``qcclass calibrar-gnn`` (coefficients clipped at 0). :meth:`MemoryModel.micro_nodes` picks
the largest micro-batch (the whole budget, half, a quarter...) whose predicted peak, times a
safety margin, fits in the useful memory of the GPU, never below the minimum micro-batch.

The model of each machine is saved as ``<resultados>/medicion/memoria_gnn_<machine>.json``.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qcclass.circuitos import write_text_atomic

FEATURES = (
    "constante",
    "dim",
    "dim_x_capas",
    "dim_x_capas_gatv2",
    "dim_nodo_virtual",
    "dim_atencion",
)
MODEL_FILE = "memoria_gnn_{machine}.json"


def features(model: dict[str, Any]) -> np.ndarray:
    """Feature vector of a model configuration (keys of ``GNNConfig``)."""
    dim, layers = float(model.get("dim", 64)), float(model.get("capas", 3))
    gat = model.get("tipo_capa", "gine") == "gatv2"
    return np.array(
        [
            1.0,
            dim,
            dim * layers,
            dim * layers * gat,
            dim * bool(model.get("nodo_virtual", False)),
            dim * (model.get("agregacion", "media_max") == "atencion"),
        ]
    )


@dataclass(frozen=True)
class MemoryModel:
    """Fitted memory model of one machine."""

    machine: str
    coefficients: tuple[float, ...]  # MB per node, one per feature
    useful_mb: float
    margin: float
    measurements: tuple[dict[str, Any], ...] = ()

    @classmethod
    def fit(
        cls,
        machine: str,
        measurements: list[dict[str, Any]],
        *,
        useful_mb: float,
        margin: float,
    ) -> "MemoryModel":
        """Least squares on ``{"modelo": ..., "mb_por_nodo": ...}`` measurements."""
        x = np.array([features(m["modelo"]) for m in measurements])
        y = np.array([float(m["mb_por_nodo"]) for m in measurements])
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        coefficients = np.clip(coefficients, 0, None)
        return cls(
            machine, tuple(float(c) for c in coefficients), useful_mb, margin, tuple(measurements)
        )

    def per_node_mb(self, model: dict[str, Any]) -> float:
        """Predicted peak MB per node of a training micro-batch."""
        return float(features(model) @ np.array(self.coefficients))

    def peak_mb(self, model: dict[str, Any], nodes: int, base_mb: float = 0.0) -> float:
        """Predicted peak of a micro-batch of ``nodes`` nodes."""
        return base_mb + nodes * self.per_node_mb(model)

    def micro_nodes(
        self, model: dict[str, Any], budget: int, minimum: int, base_mb: float = 0.0
    ) -> int:
        """Largest ``budget / 2**k`` (at least ``minimum``) predicted to fit."""
        nodes = budget
        while (
            nodes > minimum and self.margin * self.peak_mb(model, nodes, base_mb) > self.useful_mb
        ):
            nodes = max(nodes // 2, minimum)
        return nodes

    def save(self, path: Path) -> None:
        """Write the model as JSON."""
        content = {**asdict(self), "caracteristicas": list(FEATURES)}
        write_text_atomic(json.dumps(content, indent=2, ensure_ascii=False) + "\n", path)

    @classmethod
    def load(cls, path: Path) -> "MemoryModel":
        """Read a model written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["caracteristicas"]) != FEATURES:
            raise ValueError(f"{path}: features {data['caracteristicas']} differ from {FEATURES}")
        return cls(
            data["machine"],
            tuple(data["coefficients"]),
            float(data["useful_mb"]),
            float(data["margin"]),
            tuple(data.get("measurements", ())),
        )


def model_path(results: Path, machine: str) -> Path:
    """``<resultados>/medicion/memoria_gnn_<machine>.json``."""
    return Path(results) / "medicion" / MODEL_FILE.format(machine=machine)


def initial_micro_nodes(
    results: Path, machine: str, model: dict[str, Any], budget: int, minimum: int, device: str
) -> int | None:
    """Micro-batch to start with on this machine (``None``: the whole batch; no model or not
    on CUDA). Training still halves it on an out-of-memory error."""
    path = model_path(results, machine)
    if device != "cuda" or not path.is_file():
        return None
    return MemoryModel.load(path).micro_nodes(model, budget, minimum)
