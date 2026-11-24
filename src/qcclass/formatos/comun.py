"""Shared helpers for the input formats (F3-F7).

Node order. Every format that lists gates (F4 nodes, F6 tokens, F7 labels) uses the order
of the instructions in the canonical circuit (``circuit.data``). That order is a valid,
deterministic topological order of the gate DAG (an instruction never precedes one it
depends on), and it is stored implicitly: node ``i`` of F4, token ``i`` of F6 and label
``i`` of F7 all refer to ``circuit.data[i]``.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
from qiskit import QuantumCircuit

from qcclass.circuitos import CANONICAL_BASIS
from qcclass.config import DatasetConfig

NO_LABEL = -1  # no algorithm label: family-only circuits
UNKNOWN = -2  # unknown: negative control and reserved algorithms (evaluation only)


@dataclass(frozen=True)
class GateRecord:
    """One gate of a canonical circuit: name, qubit indices and angle (``None`` if none)."""

    name: str
    qubits: tuple[int, ...]
    angle: float | None


def gate_sequence(circuit: QuantumCircuit) -> list[GateRecord]:
    """Gates in node order (``circuit.data`` order); fails on non-canonical gates."""
    records = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in CANONICAL_BASIS:
            raise ValueError(f"non-canonical gate '{name}' in a canonical circuit")
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        params = instruction.operation.params
        angle = float(params[0]) if params else None
        records.append(GateRecord(name, qubits, angle))
    return records


def asap_layers(gates: list[GateRecord], num_qubits: int) -> list[int]:
    """ASAP layer (0-based) of each gate: one more than the last layer on any of its qubits."""
    last = [-1] * num_qubits
    layers = []
    for gate in gates:
        layer = 1 + max(last[q] for q in gate.qubits)
        for q in gate.qubits:
            last[q] = layer
        layers.append(layer)
    return layers


def interaction_graph(gates: list[GateRecord], num_qubits: int) -> nx.Graph:
    """Qubit interaction graph: every qubit is a node, CX gates add weighted edges."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    for gate in gates:
        if gate.name == "cx":
            a, b = gate.qubits
            weight = graph.get_edge_data(a, b, {"weight": 0})["weight"]
            graph.add_edge(a, b, weight=weight + 1)
    return graph


def wrap_angle(angle: float) -> float:
    """Angle mapped to ``[-pi, pi)``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class LabelVocabulary:
    """Indices of the trainable class labels and of the families.

    Target codes: ``NO_LABEL`` (-1) = no algorithm label (family-only circuits: the
    algorithm head ignores them, the family head uses them); ``UNKNOWN`` (-2) = unknown
    (negative control and reserved algorithms), only present in evaluation. Reserved
    algorithms with a family keep their family index.
    """

    classes: tuple[str, ...]
    families: tuple[str, ...]
    reserved: tuple[str, ...]
    unknown_without_family: tuple[str, ...]

    def class_index(self, label: str) -> int:
        """Index of a trainable class label, ``NO_LABEL`` if empty."""
        if not label:
            return NO_LABEL
        if label not in self.classes:
            raise KeyError(f"'{label}' is not a trainable class label")
        return self.classes.index(label)

    def family_index(self, family: str) -> int:
        """Index of a family."""
        return self.families.index(family)

    def targets(self, row: dict) -> tuple[int, int]:
        """(algorithm code, family code) of a metadata row."""
        if row["algoritmo"] in self.unknown_without_family:
            return UNKNOWN, UNKNOWN
        if row["algoritmo"] in self.reserved:
            return UNKNOWN, self.family_index(row["familia"])
        return self.class_index(row["clase"]), self.family_index(row["familia"])

    @classmethod
    def from_json(cls, data: dict) -> "LabelVocabulary":
        """Vocabulary from the content of ``etiquetas.json`` (inverse of :meth:`as_json`)."""
        return cls(
            classes=tuple(data["clases"]),
            families=tuple(data["familias"]),
            reserved=tuple(data["reservados"]),
            unknown_without_family=tuple(data["desconocidos_sin_familia"]),
        )

    def as_json(self) -> dict:
        """Content of ``etiquetas.json``."""
        return {
            "clases": list(self.classes),
            "familias": list(self.families),
            "reservados": list(self.reserved),
            "desconocidos_sin_familia": list(self.unknown_without_family),
            "codigos": {
                str(NO_LABEL): "sin etiqueta de algoritmo (solo familia)",
                str(UNKNOWN): "desconocido (control negativo o reservado)",
            },
        }


def label_vocabulary(
    config: DatasetConfig, reserved: Sequence[str], class_labels: Sequence[str]
) -> LabelVocabulary:
    """Vocabulary from the class labels present in the data.

    Trainable classes exclude the negative control and every reserved algorithm.
    """
    without_family = (*config.control_negativo, *config.reservados_sin_familia)
    with_family = tuple(name for name in reserved if name not in without_family)
    excluded = {*without_family, *with_family}
    classes = sorted({label for label in class_labels if label and label not in excluded})
    return LabelVocabulary(
        classes=tuple(classes),
        families=tuple(sorted(config.familias)),
        reserved=with_family,
        unknown_without_family=tuple(without_family),
    )
