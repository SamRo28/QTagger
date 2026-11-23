"""Small circuit utilities shared by the pipeline steps."""

import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import ControlFlowOp, Gate, Instruction

NON_GATE_OPERATIONS = frozenset({"measure", "reset", "barrier", "delay"})


def is_dynamic(circuit: QuantumCircuit) -> bool:
    """Return True if the circuit measures, resets or branches before its final measurements."""
    body = circuit.remove_final_measurements(inplace=False)
    return any(
        isinstance(instruction.operation, ControlFlowOp)
        or instruction.operation.name in ("measure", "reset")
        for instruction in body.data
    )


def count_gates(circuit: QuantumCircuit) -> int:
    """Count the gates of a circuit, ignoring measurements, resets, barriers and delays."""
    return sum(1 for inst in circuit.data if inst.operation.name not in NON_GATE_OPERATIONS)


def active_qubits(circuit: QuantumCircuit) -> int:
    """Number of qubits touched by at least one gate (measurements and barriers excluded)."""
    touched = {
        qubit
        for inst in circuit.data
        if inst.operation.name not in NON_GATE_OPERATIONS
        for qubit in inst.qubits
    }
    return len(touched)


def _needs_wrapping(operation: Instruction) -> bool:
    """True for operations with array parameters (e.g. a unitary matrix) and a definition.

    Array-parameter operations without a definition, such as ``PermutationGate``, are
    handled natively by the exporter and are left untouched.
    """
    has_arrays = any(isinstance(param, np.ndarray) for param in operation.params)
    return has_arrays and operation.definition is not None


def _as_named_gate(operation: Instruction) -> Gate:
    """Equivalent parameterless gate with the same name and decomposition."""
    gate = Gate(name=operation.name, num_qubits=operation.num_qubits, params=[])
    gate.definition = make_exportable(operation.definition)
    return gate


def make_exportable(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a circuit that ``qiskit.qasm3.dumps`` can serialise.

    The OpenQASM 3 exporter cannot write gates whose parameters are matrices (such as the
    controlled unitaries used by ``ae``). Each such instruction is replaced by a
    parameterless gate with the same name and the same definition, so the circuit is
    unchanged and the block keeps its name. Circuits without such gates are returned as is.
    """
    if not any(_needs_wrapping(inst.operation) for inst in circuit.data):
        return circuit
    exportable = circuit.copy_empty_like()
    for inst in circuit.data:
        operation = inst.operation
        if _needs_wrapping(operation):
            operation = _as_named_gate(operation)
        exportable.append(operation, inst.qubits, inst.clbits, copy=False)
    return exportable


def write_text_atomic(text: str, path: Path) -> None:
    """Write a text file atomically (temporary file + rename), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_qasm(circuit: QuantumCircuit, path: Path) -> None:
    """Serialise a circuit to OpenQASM 3 and write it atomically."""
    write_text_atomic(qasm3.dumps(make_exportable(circuit)), path)
