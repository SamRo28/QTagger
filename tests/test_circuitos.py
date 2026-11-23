"""Tests for the shared circuit utilities and seed derivation."""

import io

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.quantum_info import Operator

from qcclass.circuitos import (
    active_qubits,
    count_gates,
    is_dynamic,
    make_exportable,
    qpy_bytes,
)
from qcclass.semillas import derive_seed


def test_derive_seed_is_deterministic_and_bounded() -> None:
    seed = derive_seed(42, "qaoa", 5, 0)
    assert seed == derive_seed(42, "qaoa", 5, 0)
    assert 0 <= seed < 2**31
    assert len({derive_seed(42, "qaoa", 5, v) for v in range(5)}) == 5
    assert derive_seed(42, "qaoa", 5, 0) != derive_seed(43, "qaoa", 5, 0)


def test_count_and_active_qubits_ignore_non_gates() -> None:
    circuit = QuantumCircuit(4, 4)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.barrier()
    circuit.measure(range(4), range(4))
    assert count_gates(circuit) == 2
    assert active_qubits(circuit) == 2


def test_is_dynamic() -> None:
    assert not is_dynamic(get_benchmark("ghz", BenchmarkLevel.ALG, 4))
    assert is_dynamic(get_benchmark("teleportation", BenchmarkLevel.ALG, 3))


def test_make_exportable_keeps_ae_equivalent() -> None:
    """``ae`` has a controlled unitary with a matrix parameter that QASM 3 cannot export."""
    original = get_benchmark("ae", BenchmarkLevel.ALG, 3, seed=1)
    with pytest.raises(TypeError):
        qasm3.dumps(original)

    exportable = make_exportable(original)
    qasm3.dumps(exportable)  # must not raise

    assert {inst.operation.name for inst in exportable.data} == {
        inst.operation.name for inst in original.data
    }
    unitary_a = original.remove_final_measurements(inplace=False)
    unitary_b = exportable.remove_final_measurements(inplace=False)
    assert Operator(unitary_a).equiv(Operator(unitary_b))


def test_make_exportable_returns_plain_circuits_unchanged() -> None:
    circuit = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    assert make_exportable(circuit) is circuit


@pytest.mark.parametrize(
    ("name", "n"), [("full_adder", 4), ("half_adder", 5), ("modular_adder", 4), ("multiplier", 4)]
)
def test_qpy_round_trip_of_library_gates(name: str, n: int) -> None:
    """Qiskit 2.6 cannot read back some library gates from QPY; qpy_bytes works around it."""
    circuit = get_benchmark(name, BenchmarkLevel.ALG, n)
    loaded = qpy.load(io.BytesIO(qpy_bytes(circuit)))[0]
    assert [i.operation.name for i in loaded.data] == [i.operation.name for i in circuit.data]
    body_a = circuit.remove_final_measurements(inplace=False)
    body_b = loaded.remove_final_measurements(inplace=False)
    assert Operator(body_a).equiv(Operator(body_b))
