"""Tests for the in-house oracle generators (simulation checks for n <= 6)."""

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Operator, Statevector
from typer.testing import CliRunner

from qcclass.circuitos import qpy_path, read_qpy
from qcclass.cli import app
from qcclass.config import DatasetConfig
from qcclass.data.oraculos import (
    ERROR_COLUMNS,
    MCX_M15,
    MCX_VCHAIN,
    InvalidSizeError,
    _mcx,
    bernstein_vazirani,
    deutsch_jozsa,
    errors_path,
    generate_oracles,
    grover,
    grover_constructions,
    mcx_ancillas,
    simon,
)
from qcclass.metadatos import METADATA_COLUMNS, read_table

ConfigFactory = Callable[..., DatasetConfig]  # the ``make_config`` fixture
SEEDS = range(8)


def _input_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    """Probabilities of the measured register (index = integer, qubit 0 = least significant)."""
    state = Statevector(circuit.remove_final_measurements(inplace=False))
    return state.probabilities(qargs=list(range(circuit.num_clbits)))


def _bits_to_int(bits: str) -> int:
    """Bit string written qubit 0 first -> integer."""
    return sum(int(bit) << i for i, bit in enumerate(bits))


def _field(oracle_id: str, prefix: str) -> str:
    """Value of the ``_<prefix>...`` field of an oracle identifier (after ``<alg>_n<n>``)."""
    fields = oracle_id.split("_")[2:]
    return next(part[len(prefix) :] for part in fields if part.startswith(prefix))


@pytest.mark.parametrize("n", range(2, 7))
@pytest.mark.parametrize("seed", SEEDS)
def test_deutsch_jozsa_distinguishes_constant_and_balanced(n: int, seed: int) -> None:
    generated = deutsch_jozsa(n, random.Random(seed))
    probability_all_zero = _input_probabilities(generated.circuit)[0]
    if "const" in generated.oracle_id:
        assert probability_all_zero == pytest.approx(1.0)
    else:
        assert probability_all_zero == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", range(2, 7))
@pytest.mark.parametrize("seed", SEEDS)
def test_bernstein_vazirani_recovers_secret(n: int, seed: int) -> None:
    generated = bernstein_vazirani(n, random.Random(seed))
    secret = _bits_to_int(_field(generated.oracle_id, "s"))
    assert secret != 0
    assert _input_probabilities(generated.circuit)[secret] == pytest.approx(1.0)


@pytest.mark.parametrize("n", [2, 4, 6])
@pytest.mark.parametrize("seed", SEEDS)
def test_simon_outcomes_are_orthogonal_to_secret(n: int, seed: int) -> None:
    generated = simon(n, random.Random(seed))
    secret = _bits_to_int(_field(generated.oracle_id, "s"))
    probabilities = _input_probabilities(generated.circuit)
    for outcome, probability in enumerate(probabilities):
        if probability > 1e-9:
            assert bin(outcome & secret).count("1") % 2 == 0


@pytest.mark.parametrize("n", [2, 3, 5, 6, 8])
@pytest.mark.parametrize("construction", [MCX_VCHAIN, MCX_M15])
@pytest.mark.parametrize("seed", SEEDS)
def test_grover_amplifies_marked_states(n: int, construction: str, seed: int) -> None:
    generated = grover(n, random.Random(seed), construction=construction)
    assert generated.circuit.num_qubits == n
    marked = [_bits_to_int(bits) for bits in _field(generated.oracle_id, "m").split("-")]
    probabilities = _input_probabilities(generated.circuit)
    assert sum(probabilities[state] for state in marked) > 0.5


@pytest.mark.parametrize("num_controls", [3, 4, 5, 6])
@pytest.mark.parametrize("construction", [MCX_VCHAIN, MCX_M15])
def test_mcx_constructions_match_mcx_on_clean_ancillas(
    num_controls: int, construction: str
) -> None:
    """With ancillas in |0>, both MCX blocks act as a plain multi-controlled X."""
    block = _mcx(construction, num_controls)
    ancillas = mcx_ancillas(construction, num_controls)
    assert block.num_qubits == num_controls + 1 + ancillas
    vchain = QuantumCircuit(block.num_qubits)
    vchain.append(block, range(block.num_qubits))
    reference = QuantumCircuit(block.num_qubits)
    reference.append(MCXGate(num_controls), range(num_controls + 1))
    full_a, full_b = Operator(vchain).data, Operator(reference).data
    clean = [i for i in range(2**block.num_qubits) if i >> (num_controls + 1) == 0]
    assert ancillas > 0
    assert np.allclose(full_a[np.ix_(clean, clean)], full_b[np.ix_(clean, clean)])


def test_invalid_sizes() -> None:
    with pytest.raises(InvalidSizeError):
        simon(5, random.Random(0))
    with pytest.raises(InvalidSizeError):
        grover(4, random.Random(0))
    assert grover_constructions(7) == {MCX_VCHAIN: 5}
    assert grover_constructions(6) == {MCX_M15: 5}
    assert set(grover_constructions(9)) == {MCX_VCHAIN, MCX_M15}


def _metadata(config: DatasetConfig) -> pd.DataFrame:
    return read_table(config.rutas.metadata, METADATA_COLUMNS)


def test_generation_metadata_files_and_blocks(make_config: ConfigFactory) -> None:
    config = make_config(oraculos={"por_tamano": 4})
    report = generate_oracles(config, ["dj", "bv", "simon", "grover"], range(2, 7))
    metadata = _metadata(config)

    assert len(report.rows) == len(metadata)
    assert set(metadata["origen"]) == {"propio"}
    assert set(metadata["nivel"]) == {"alg"}
    assert set(metadata["familia"]) == {"oraculos"}
    assert metadata["group_id"].is_unique and metadata["oraculo_id"].is_unique
    per_size = metadata.groupby(["algoritmo", "n_qubits"]).size()
    assert per_size.max() <= 4
    assert per_size[("dj", "6")] == 4
    assert per_size[("bv", "2")] == 1  # a single non-zero secret exists with one input

    for row in metadata.itertuples():
        path = config.rutas.raiz / row.ruta_raw
        assert path.match(f"data/raw/alg/oraculos/{row.algoritmo}/{row.group_id}.qasm")
        assert row.group_id == f"{row.algoritmo}_n{row.n_qubits}_o{row.semilla}"
        names = {inst.operation.name for inst in read_qpy(qpy_path(path)).data}
        assert {"hadamard_inicial", "oraculo"} <= names

    errors = read_table(errors_path(config), ERROR_COLUMNS)
    grover_rows = metadata[metadata["algoritmo"] == "grover"]
    assert set(grover_rows["construccion_mcx"]) <= {"directa", MCX_VCHAIN, MCX_M15}
    assert set(grover_rows.loc[grover_rows["n_qubits"] == "6", "construccion_mcx"]) == {MCX_M15}
    assert set(metadata.loc[metadata["algoritmo"] != "grover", "construccion_mcx"]) == {""}
    assert {("simon", "3"), ("simon", "5"), ("grover", "4")} <= set(
        zip(errors["algoritmo"], errors["n"], strict=True)
    )


def test_generation_is_incremental_and_reproducible(make_config: ConfigFactory) -> None:
    config = make_config(oraculos={"por_tamano": 3})
    first = generate_oracles(config, ["bv"], range(3, 6))
    before = config.rutas.metadata.read_text(encoding="utf-8")
    again = generate_oracles(config, ["bv"], range(3, 6))
    assert len(again.rows) == 0 and again.skipped == len(first.rows)
    forced = generate_oracles(config, ["bv"], range(3, 6), force=True)
    assert len(forced.rows) == len(first.rows)
    assert config.rutas.metadata.read_text(encoding="utf-8") == before


def test_grover_stops_when_nothing_fits(make_config: ConfigFactory) -> None:
    config = make_config(max_nodos=300)
    generate_oracles(config, ["grover"], range(2, 10))
    errors = read_table(errors_path(config), ERROR_COLUMNS)
    assert "omitido_por_tamano" in set(errors["tipo"])
    assert max(_metadata(config)["n_qubits"].astype(int)) <= 5


def test_cli_oracles(make_config: ConfigFactory, tmp_path: Path) -> None:
    make_config(oraculos={"por_tamano": 2})
    config_path = tmp_path / "configs" / "dataset.yaml"
    args = ["oraculos", "--config", str(config_path), "--algoritmos", "dj,bv", "--max", "4"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Oracle circuits written" in result.output


def test_grover_constructions_are_balanced_per_size(make_config: ConfigFactory) -> None:
    config = make_config(oraculos={"por_tamano": 6})
    generate_oracles(config, ["grover"], [9])
    used = _metadata(config)["construccion_mcx"].value_counts()
    assert used[MCX_VCHAIN] == used[MCX_M15] == 3
