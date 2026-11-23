"""Small circuit utilities shared by the pipeline steps."""

import io
import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3, qpy, transpile
from qiskit.circuit import ControlFlowOp, Gate, Instruction
from qiskit.circuit.library import get_standard_gate_name_mapping

NON_GATE_OPERATIONS = frozenset({"measure", "reset", "barrier", "delay"})
CANONICAL_BASIS = ("rz", "sx", "x", "cx")


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


def canonical_gate_count(circuit: QuantumCircuit) -> int:
    """Gates after a plain (``optimization_level=0``) translation to the canonical basis.

    Used as a cheap size estimate before the real normalisation step.
    """
    flat = transpile(circuit, basis_gates=list(CANONICAL_BASIS), optimization_level=0)
    return count_gates(flat)


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


def _write_atomic(data: bytes, path: Path) -> None:
    """Write a file atomically (temporary file + rename), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    """Write a UTF-8 text file atomically."""
    _write_atomic(text.encode("utf-8"), path)


def write_qasm(circuit: QuantumCircuit, path: Path) -> None:
    """Serialise a circuit to OpenQASM 3 and write it atomically."""
    write_text_atomic(qasm3.dumps(make_exportable(circuit)), path)


def qpy_path(qasm_file: Path) -> Path:
    """QPY twin of a QASM file: same name, ``.qpy`` extension."""
    return qasm_file.with_suffix(".qpy")


def write_qpy(circuit: QuantumCircuit, path: Path) -> None:
    """Serialise a circuit to QPY (Qiskit's lossless binary format) and write it atomically."""
    _write_atomic(qpy_bytes(circuit), path)


def read_qpy(path: Path) -> QuantumCircuit:
    """Load the single circuit stored in a QPY file."""
    with path.open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError(f"{path}: expected one circuit, found {len(circuits)}")
    return circuits[0]


def write_circuit(circuit: QuantumCircuit, qasm_file: Path) -> None:
    """Write a circuit as readable OpenQASM 3 plus its lossless QPY twin.

    QPY is the input of the later pipeline steps: it loads orders of magnitude faster than
    ``qasm3.loads`` and keeps gate names and the physical width of mapped circuits. Both
    files are serialised before either is written, so an export error leaves neither.
    """
    text = qasm3.dumps(make_exportable(circuit))
    data = qpy_bytes(circuit)
    write_text_atomic(text, qasm_file)
    _write_atomic(data, qpy_path(qasm_file))


def _qpy_dump(circuit: QuantumCircuit) -> bytes:
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


def _qpy_loads(data: bytes) -> QuantumCircuit:
    return qpy.load(io.BytesIO(data))[0]


def make_qpy_safe(circuit: QuantumCircuit) -> QuantumCircuit:
    """Wrap every top-level non-standard gate with a definition as a plain named gate.

    Qiskit 2.6 writes some library gates (``FullAdderGate``, ``HalfAdderGate``,
    ``ModularAdderGate``, ``MultiplierGate``) to QPY but cannot read them back. The wrapped
    gates keep their name and definition, so the circuit and its blocks are unchanged.
    """
    standard = get_standard_gate_name_mapping()
    safe = circuit.copy_empty_like()
    for inst in circuit.data:
        operation = inst.operation
        if (
            isinstance(operation, Gate)
            and operation.name not in standard
            and operation.definition is not None
            and type(operation) is not Gate
        ):
            operation = _as_named_gate(operation)
        safe.append(operation, inst.qubits, inst.clbits, copy=False)
    return safe


def qpy_bytes(circuit: QuantumCircuit) -> bytes:
    """QPY serialisation that is guaranteed to load back.

    If the plain serialisation cannot be read, library gates are wrapped with
    :func:`make_qpy_safe`; if even that fails, the error is raised.
    """
    data = _qpy_dump(circuit)
    try:
        _qpy_loads(data)
    except Exception:
        data = _qpy_dump(make_qpy_safe(circuit))
        _qpy_loads(data)
    return data
