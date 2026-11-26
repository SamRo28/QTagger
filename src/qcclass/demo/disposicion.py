"""Layout of the circuits for the drawings: F1 instructions, F2 gates, blocks and columns.

Columns: every gate goes to the first column free on *all* the qubits it occupies; with
``span=True`` it occupies every qubit between its lowest and highest one (as a drawn box or
a CX line crosses them), with ``span=False`` only its own qubits. The same rule is used for
F1 (one column per top-level instruction) and for F2.

Blocks: as in F7 (:mod:`qcclass.formatos.etiquetas_puerta`), each top-level instruction of the
F1 circuit (without final measurements and barriers) is decomposed on its own to the canonical
basis with ``optimization_level=0``; the pieces, in order, are the F2 gates, and each gate
keeps the index of the instruction (block) it comes from. :func:`blocks_of` returns that
expansion; for the ``etiquetada`` variants it reproduces the stored canonical circuit exactly
(:func:`check_same_gates`).
"""

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile

from qcclass.circuitos import CANONICAL_BASIS
from qcclass.data.normalizar import strip_non_unitary
from qcclass.formatos.comun import GateRecord, gate_sequence


@dataclass(frozen=True)
class Expanded:
    """F1 instructions and the F2 gates they decompose into."""

    instructions: list[tuple[str, tuple[int, ...], list[float]]]
    gates: list[GateRecord]
    block_of_gate: list[int]
    circuit: QuantumCircuit


def _params(operation) -> list[float]:  # noqa: ANN001 - qiskit Operation
    values = []
    for param in operation.params:
        try:
            values.append(float(param))
        except (TypeError, ValueError):
            continue  # matrices or symbolic parameters are not drawn
    return values


def blocks_of(original: QuantumCircuit) -> Expanded:
    """Decompose every top-level instruction separately (see the module docstring)."""
    body = strip_non_unitary(original, drop_idle_qubits=False)
    result = QuantumCircuit(*body.qregs, global_phase=body.global_phase)
    instructions, block_of_gate = [], []
    for index, instruction in enumerate(body.data):
        qubits = tuple(body.find_bit(qubit).index for qubit in instruction.qubits)
        instructions.append((instruction.operation.name, qubits, _params(instruction.operation)))
        piece = QuantumCircuit(body.num_qubits)
        piece.append(instruction.operation, list(qubits))
        flat = transpile(piece, basis_gates=list(CANONICAL_BASIS), optimization_level=0)
        for gate in flat.data:
            targets = [result.qubits[flat.find_bit(qubit).index] for qubit in gate.qubits]
            result.append(gate.operation, targets)
            block_of_gate.append(index)
    return Expanded(instructions, gate_sequence(result), block_of_gate, result)


def check_same_gates(expected: list[GateRecord], found: list[GateRecord]) -> None:
    """Raise ``ValueError`` unless both gate lists match (name, qubits, angle to 1e-9)."""
    if len(expected) != len(found):
        raise ValueError(f"{len(found)} gates instead of {len(expected)}")
    for index, (a, b) in enumerate(zip(expected, found, strict=True)):
        same_angle = (a.angle is None and b.angle is None) or (
            a.angle is not None and b.angle is not None and abs(a.angle - b.angle) < 1e-9
        )
        if a.name != b.name or a.qubits != b.qubits or not same_angle:
            raise ValueError(f"gate {index} differs: {a} / {b}")


def columns(qubit_sets: list[tuple[int, ...]], num_qubits: int, *, span: bool) -> list[int]:
    """Drawing column of every element (see the module docstring)."""
    free = [0] * num_qubits
    result = []
    for qubits in qubit_sets:
        occupied = range(min(qubits), max(qubits) + 1) if span else qubits
        column = max(free[q] for q in occupied)
        for q in occupied:
            free[q] = column + 1
        result.append(column)
    return result


def block_extents(block_of_gate: list[int], gate_columns: list[int]) -> dict[int, tuple[int, int]]:
    """First and last F2 column of the gates of every block."""
    extents: dict[int, tuple[int, int]] = {}
    for block, column in zip(block_of_gate, gate_columns, strict=True):
        low, high = extents.get(block, (column, column))
        extents[block] = (min(low, column), max(high, column))
    return extents
