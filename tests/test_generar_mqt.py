"""Tests for F1: raw circuit generation from MQT Bench."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from qiskit import qasm3
from typer.testing import CliRunner

from qcclass.circuitos import active_qubits, qpy_path, read_qpy
from qcclass.cli import app
from qcclass.config import DatasetConfig
from qcclass.data.generar_mqt import (
    ERROR_COLUMNS,
    build_selection,
    errors_path,
    generate,
)
from qcclass.metadatos import METADATA_COLUMNS, read_table

ConfigFactory = Callable[..., DatasetConfig]  # the ``make_config`` fixture


def _generate(config: DatasetConfig, algorithms: list[str], low: int, high: int, **kwargs):
    """Generate a small selection of circuits."""
    selection = build_selection(config, algorithms, low, high)
    return generate(config, selection, **kwargs)


def _metadata(config: DatasetConfig) -> pd.DataFrame:
    return read_table(config.rutas.metadata, METADATA_COLUMNS)


def _errors(config: DatasetConfig) -> pd.DataFrame:
    return read_table(errors_path(config), ERROR_COLUMNS)


def test_generated_circuit_reloads_with_expected_qubits(make_config: ConfigFactory) -> None:
    """Every written file loads with ``qasm3.loads`` and has the logical width in metadata."""
    config = make_config()
    report = _generate(config, ["ghz", "qft"], 2, 4)

    metadata = _metadata(config)
    assert len(report.rows) == len(metadata) == 2 * 3 * 4  # algorithms x sizes x levels
    for row in metadata.itertuples():
        circuit = qasm3.loads((config.rutas.raiz / row.ruta_raw).read_text(encoding="utf-8"))
        if row.nivel == "mapped":
            # mapped circuits are written on physical qubits; only n of them are used
            assert active_qubits(circuit) == int(row.n_qubits)
        else:
            assert circuit.num_qubits == int(row.n_qubits)


def test_metadata_and_file_layout(make_config: ConfigFactory) -> None:
    config = make_config()
    _generate(config, ["ghz"], 3, 3)

    metadata = _metadata(config)
    assert list(metadata.columns) == list(METADATA_COLUMNS)
    assert set(metadata["nivel"]) == {"alg", "indep", "nativegates", "mapped"}
    assert set(metadata["group_id"]) == {"ghz_n3_p0"}  # shared by every level
    assert metadata["semilla"].nunique() == 1
    assert set(metadata["origen"]) == {"mqt"}
    assert set(metadata["variante_comp"]) == {""}
    assert set(metadata["ruta_raw"]) == {
        f"data/raw/{level}/preparacion_estados/ghz/ghz_n3_p0.qasm"
        for level in ("alg", "indep", "nativegates", "mapped")
    }
    assert set(metadata["id"]) == {
        f"{level}_ghz_n3_p0" for level in ("alg", "indep", "nativegates", "mapped")
    }


def test_parameter_variants_differ_and_share_seed_across_levels(
    make_config: ConfigFactory,
) -> None:
    config = make_config(variantes_parametros=2)
    _generate(config, ["qaoa"], 3, 3)

    metadata = _metadata(config)
    assert set(metadata["group_id"]) == {"qaoa_n3_p0", "qaoa_n3_p1"}
    seeds = metadata.groupby("group_id")["semilla"].nunique()
    assert (seeds == 1).all()
    assert metadata.drop_duplicates("group_id")["semilla"].nunique() == 2

    alg = metadata[metadata["nivel"] == "alg"].set_index("variante_param")["ruta_raw"]
    text_p0 = (config.rutas.raiz / alg["0"]).read_text(encoding="utf-8")
    text_p1 = (config.rutas.raiz / alg["1"]).read_text(encoding="utf-8")
    assert text_p0 != text_p1


def test_generation_is_incremental(make_config: ConfigFactory) -> None:
    config = make_config()
    first = _generate(config, ["ghz"], 2, 3)
    before = config.rutas.metadata.read_text(encoding="utf-8")

    second = _generate(config, ["ghz"], 2, 3)
    assert len(second.rows) == 0
    assert second.skipped == len(first.rows)
    assert config.rutas.metadata.read_text(encoding="utf-8") == before

    forced = _generate(config, ["ghz"], 2, 3, force=True)
    assert len(forced.rows) == len(first.rows)
    assert config.rutas.metadata.read_text(encoding="utf-8") == before


def test_invalid_sizes_are_logged_once(make_config: ConfigFactory) -> None:
    config = make_config()
    for _ in range(2):  # re-running must replace, not duplicate, the log entries
        _generate(config, ["simon"], 3, 4)

    errors = _errors(config)
    assert len(errors) == 1
    assert errors.iloc[0][["algoritmo", "n", "tipo", "nivel"]].tolist() == [
        "simon",
        "3",
        "generacion",
        "todos",
    ]
    assert set(_metadata(config)["n_qubits"]) == {"4"}


def test_size_probe_stops_growing_circuits(make_config: ConfigFactory) -> None:
    config = make_config(max_nodos=150)
    _generate(config, ["grover"], 2, 6)

    errors = _errors(config).set_index("n")
    too_large = errors.index[errors["tipo"] == "demasiado_grande"].tolist()
    assert len(too_large) == 1
    first = int(too_large[0])
    assert set(errors.loc[errors["tipo"] == "omitido_por_tamano"].index.astype(int)) == set(
        range(first + 1, 7)
    )
    assert max(_metadata(config)["n_qubits"].astype(int)) < first


def test_excluded_and_unknown_algorithms_are_rejected(make_config: ConfigFactory) -> None:
    config = make_config()
    with pytest.raises(ValueError, match="excluded"):
        build_selection(config, ["teleportation"])
    with pytest.raises(ValueError, match="unknown"):
        build_selection(config, ["not_a_benchmark"])


def test_cli_quick_generation(make_config: ConfigFactory, tmp_path: Path) -> None:
    """The quick check of the plan: ``qcclass generar --algoritmos ghz,qft --max 6``."""
    make_config()
    config_path = tmp_path / "configs" / "dataset.yaml"
    result = CliRunner().invoke(
        app, ["generar", "--config", str(config_path), "--algoritmos", "ghz,qft", "--max", "6"]
    )
    assert result.exit_code == 0, result.output
    assert "Circuits written: 40" in result.output
    assert (tmp_path / "data" / "metadata.csv").is_file()


def test_every_raw_circuit_has_a_qpy_twin(make_config: ConfigFactory) -> None:
    """QPY is lossless: same gate names as the source, and mapped circuits keep their width."""
    config = make_config()
    _generate(config, ["ae"], 3, 3)
    for row in _metadata(config).itertuples():
        qasm_file = config.rutas.raiz / row.ruta_raw
        circuit = read_qpy(qpy_path(qasm_file))
        if row.nivel == "mapped":
            assert circuit.num_qubits == 156
            assert active_qubits(circuit) == int(row.n_qubits)
        else:
            assert circuit.num_qubits == int(row.n_qubits)
    alg = read_qpy(config.rutas.raw / "alg" / "qft_fase" / "ae" / "ae_n3_p0.qpy")
    assert "c-unitary" in alg.count_ops()  # original block name, not the exporter's alias


def test_family_less_reserved_algorithm(make_config: ConfigFactory) -> None:
    config = make_config()
    _generate(config, ["qwalk"], 3, 3)
    assert set(_metadata(config)["familia"]) == {"sin_familia"}
