"""F4: gate dependency DAG as a PyTorch Geometric graph.

Nodes are the gates of the canonical circuit in node order (see ``formatos.comun``). There
is an edge ``u -> v`` when ``v`` is the next gate after ``u`` on some qubit (direct
dependency, time direction); two gates sharing both qubits of consecutive CX give a single
edge.

Node attributes (``x``, float32, in this order): one-hot gate type (rz, sx, x, cx);
``sin(theta)`` and ``cos(theta)`` (0 and 1 without angle); relative ASAP layer
(``layer / (depth - 1)``, 0 for depth 1); and the relative index of the gate's qubits
(``qubit / n_qubits``): for CX the control and the target, for single-qubit gates the same
qubit twice.

Edge attributes (``edge_attr``): control/target indicators of the qubit that carries the
dependency, in the source gate and in the destination gate
(``[src_control, src_target, dst_control, dst_target]``, 0 for single-qubit gates). This is
how the role of each qubit in a CX is encoded, since a CX node is both control and target.

Each graph is saved as ``<graphs>/dag/<id>.pt`` (a plain dictionary, loadable with
``torch.load(..., weights_only=True)``); ``indice.csv`` lists them and
:class:`IndexedGraphDataset` loads them lazily.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from qcclass.circuitos import CANONICAL_BASIS
from qcclass.formatos.comun import GateRecord, asap_layers

NODE_FEATURES = (
    *(f"es_{name}" for name in CANONICAL_BASIS),
    "sin_theta",
    "cos_theta",
    "capa_relativa",
    "cubit_a_relativo",
    "cubit_b_relativo",
)
EDGE_FEATURES = ("origen_control", "origen_destino", "fin_control", "fin_destino")
INDEX_FILE = "indice.csv"


def _roles(gate: GateRecord, qubit: int) -> tuple[float, float]:
    """(is control, is target) of ``qubit`` in ``gate``."""
    if gate.name != "cx":
        return 0.0, 0.0
    return float(qubit == gate.qubits[0]), float(qubit == gate.qubits[1])


def _node_features(gates: list[GateRecord], num_qubits: int) -> torch.Tensor:
    """Node attribute matrix (see the module docstring)."""
    layers = asap_layers(gates, num_qubits)
    depth = max(layers) + 1 if layers else 1
    rows = []
    for gate, layer in zip(gates, layers, strict=True):
        one_hot = [float(gate.name == name) for name in CANONICAL_BASIS]
        angle = [math.sin(gate.angle), math.cos(gate.angle)] if gate.angle is not None else [0, 1]
        layer_rel = layer / (depth - 1) if depth > 1 else 0.0
        first, second = gate.qubits[0], gate.qubits[-1]
        rows.append([*one_hot, *angle, layer_rel, first / num_qubits, second / num_qubits])
    return torch.tensor(rows, dtype=torch.float32).reshape(len(gates), len(NODE_FEATURES))


def _edges(gates: list[GateRecord], num_qubits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct dependency edges (deduplicated) and their role attributes."""
    last: list[int | None] = [None] * num_qubits
    attributes: dict[tuple[int, int], list[float]] = {}
    for index, gate in enumerate(gates):
        for qubit in gate.qubits:
            previous = last[qubit]
            if previous is not None:
                roles = [*_roles(gates[previous], qubit), *_roles(gate, qubit)]
                current = attributes.setdefault((previous, index), [0.0] * 4)
                attributes[(previous, index)] = [
                    max(a, b) for a, b in zip(current, roles, strict=True)
                ]
            last[qubit] = index
    pairs = sorted(attributes)
    edge_index = torch.tensor(pairs, dtype=torch.long).reshape(len(pairs), 2).t().contiguous()
    edge_attr = torch.tensor([attributes[p] for p in pairs], dtype=torch.float32)
    return edge_index, edge_attr.reshape(len(pairs), len(EDGE_FEATURES))


def dag_graph(gates: list[GateRecord], num_qubits: int) -> Data:
    """Build the F4 graph (labels are added by the caller)."""
    edge_index, edge_attr = _edges(gates, num_qubits)
    return Data(
        x=_node_features(gates, num_qubits),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(gates),
    )


def save_graph(graph: Data, path: Path) -> None:
    """Save a graph as a plain dictionary of tensors and strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(graph.to_dict(), temporary)
    temporary.replace(path)


def load_graph(path: Path) -> Data:
    """Load a graph saved with :func:`save_graph` (safe ``weights_only`` loading)."""
    return Data.from_dict(torch.load(path, weights_only=True))


class IndexedGraphDataset(Dataset):
    """Lazily loaded graphs of one directory (``dag/`` or ``cubits/``), in index order."""

    def __init__(self, directory: Path, ids: Sequence[str] | None = None, transform=None):
        self.directory = Path(directory)
        index = pd.read_csv(self.directory / INDEX_FILE, dtype=str, keep_default_na=False)
        self.ids = list(index["id"]) if ids is None else list(ids)
        super().__init__(root=None, transform=transform)

    def len(self) -> int:
        """Number of graphs."""
        return len(self.ids)

    def get(self, idx: int) -> Data:
        """Load graph ``idx``."""
        return load_graph(self.directory / f"{self.ids[idx]}.pt")
