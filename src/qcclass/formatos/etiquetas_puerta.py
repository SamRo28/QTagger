"""F7: per-gate labels with the high-level block each gate comes from.

Applies to circuits with block structure: the ``alg`` level of MQT Bench and the in-house
oracles. The algorithmic circuit (read from QPY, which keeps the original block names) is
stripped of final measurements and barriers; then every top-level instruction is decomposed
*on its own* to the canonical basis with ``optimization_level=0`` and the pieces are
concatenated, recording for each resulting gate the name of the instruction it comes from.

The concatenated circuit is stored as an extra canonical variant
(``variante_comp = etiquetada``), so that the labels and its F4 graph describe exactly the
same circuit; ``<gate_labels>/<id>.json`` holds the labels in F4 node order and the blocks
found.
"""

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile

from qcclass.circuitos import CANONICAL_BASIS
from qcclass.data.deduplicar import LABELLED_VARIANT  # noqa: F401  (re-exported)
from qcclass.data.normalizar import strip_non_unitary


@dataclass(frozen=True)
class LabelledCircuit:
    """Canonical circuit built block by block, with the block of each gate."""

    circuit: QuantumCircuit
    labels: list[str]
    blocks: dict[str, dict[str, int]]


def labelled_canonical(algorithmic: QuantumCircuit) -> LabelledCircuit:
    """Decompose each top-level instruction separately and label every resulting gate."""
    body = strip_non_unitary(algorithmic, drop_idle_qubits=False)
    result = QuantumCircuit(*body.qregs, global_phase=body.global_phase)
    labels: list[str] = []
    blocks: dict[str, dict[str, int]] = {}
    for instruction in body.data:
        name = instruction.operation.name
        qubits = [body.find_bit(qubit).index for qubit in instruction.qubits]
        piece = QuantumCircuit(body.num_qubits)
        piece.append(instruction.operation, qubits)
        flat = transpile(piece, basis_gates=list(CANONICAL_BASIS), optimization_level=0)
        result.global_phase += flat.global_phase
        for gate in flat.data:
            targets = [result.qubits[flat.find_bit(qubit).index] for qubit in gate.qubits]
            result.append(gate.operation, targets)
            labels.append(name)
        stats = blocks.setdefault(name, {"instrucciones": 0, "puertas": 0})
        stats["instrucciones"] += 1
        stats["puertas"] += len(flat.data)
    return LabelledCircuit(circuit=result, labels=labels, blocks=blocks)
