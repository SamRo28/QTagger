"""Tests for the input formats F3-F7."""

import json
import math
import random
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import torch
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qcclass.config import DatasetConfig
from qcclass.data.deduplicar import deduplicate
from qcclass.data.generar_mqt import build_selection, generate
from qcclass.data.normalizar import normalize, strip_non_unitary
from qcclass.data.oraculos import generate_oracles, grover
from qcclass.formatos import caracteristicas, grafo_cubits, grafo_dag, tokens
from qcclass.formatos.comun import gate_sequence
from qcclass.formatos.construir import (
    DAG_DIR,
    FEATURES_FILE,
    QUBIT_DIR,
    TOKENS_FILE,
    build_formats,
)
from qcclass.formatos.etiquetas_puerta import labelled_canonical
from qcclass.metadatos import METADATA_COLUMNS, read_table

ConfigFactory = Callable[..., DatasetConfig]  # the ``make_config`` fixture
RESERVED = ("qwalk", "hrs_cumulative_multiplier", "hhl")


def _known_circuit() -> QuantumCircuit:
    """rz(pi/2) q0; sx q1; cx q0, q1; x q1."""
    circuit = QuantumCircuit(2)
    circuit.rz(math.pi / 2, 0)
    circuit.sx(1)
    circuit.cx(0, 1)
    circuit.x(1)
    return circuit


def test_known_two_qubit_circuit_dag() -> None:
    gates = gate_sequence(_known_circuit())
    graph = grafo_dag.dag_graph(gates, 2)
    expected_x = torch.tensor(
        [
            # rz  sx   x   cx  sin  cos layer qa   qb
            [1, 0, 0, 0, 1, 0, 0.0, 0.0, 0.0],
            [0, 1, 0, 0, 0, 1, 0.0, 0.5, 0.5],
            [0, 0, 0, 1, 0, 1, 0.5, 0.0, 0.5],
            [0, 0, 1, 0, 0, 1, 1.0, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    assert graph.num_nodes == 4
    assert torch.allclose(graph.x, expected_x, atol=1e-7)
    assert graph.edge_index.tolist() == [[0, 1, 2], [2, 2, 3]]
    # [src control, src target, dst control, dst target] of the qubit carrying the edge
    assert graph.edge_attr.tolist() == [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]


def test_known_two_qubit_circuit_other_formats() -> None:
    gates = gate_sequence(_known_circuit())
    qubits = grafo_cubits.qubit_graph(gates, 2)
    assert qubits.x.tolist() == [[1, 1], [2, 1]]
    assert qubits.edge_index.tolist() == [[0, 1], [1, 0]]
    assert qubits.edge_weight.tolist() == [1, 1]

    assert tokens.tokens(gates) == ["rz q0 1.571", "sx q1", "cx q0 q1", "x q1"]

    features = caracteristicas.features(gates, 2)
    assert features["n_puertas"] == 4 and features["profundidad"] == 3
    assert features["n_cx"] == 1 and features["prop_dos_cubits"] == 0.25
    assert features["cx_por_capa_media"] == pytest.approx(1 / 3)
    assert features["cx_por_capa_max"] == 1
    assert (features["grafo_grado_medio"], features["grafo_densidad"]) == (1.0, 1.0)
    assert (features["grafo_diametro"], features["grafo_componentes"]) == (1, 1)


def test_tokens_wrap_angles_and_avoid_negative_zero() -> None:
    circuit = QuantumCircuit(1)
    circuit.rz(2 * math.pi - 0.001, 0)
    circuit.rz(-0.0001, 0)
    circuit.rz(3 * math.pi / 2, 0)
    assert tokens.tokens(gate_sequence(circuit)) == ["rz q0 -0.001", "rz q0 0.000", "rz q0 -1.571"]


def test_disconnected_interaction_graph_has_zero_diameter() -> None:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    features = caracteristicas.features(gate_sequence(circuit), 3)
    assert features["grafo_diametro"] == 0 and features["grafo_componentes"] == 2


def test_graph_save_and_load(tmp_path: Path) -> None:
    graph = grafo_dag.dag_graph(gate_sequence(_known_circuit()), 2)
    graph.id = "known"
    grafo_dag.save_graph(graph, tmp_path / "known.pt")
    loaded = grafo_dag.load_graph(tmp_path / "known.pt")
    assert loaded.id == "known" and torch.equal(loaded.x, graph.x)


@pytest.mark.parametrize("seed", range(3))
def test_labelled_canonical_is_equivalent_and_labelled(seed: int) -> None:
    algorithmic = grover(5, random.Random(seed)).circuit
    labelled = labelled_canonical(algorithmic)
    body = strip_non_unitary(algorithmic, drop_idle_qubits=False)
    assert Operator(body).equiv(Operator(labelled.circuit))
    assert len(labelled.labels) == labelled.circuit.size()
    assert set(labelled.labels) == {"hadamard_inicial", "oraculo", "difusor"}
    assert sum(b["puertas"] for b in labelled.blocks.values()) == len(labelled.labels)
    assert labelled.labels[0] == "hadamard_inicial"


def test_labelled_canonical_keeps_mqt_block_names() -> None:
    labelled = labelled_canonical(get_benchmark("ae", BenchmarkLevel.ALG, 4, seed=1))
    assert {"c-unitary", "qft_dg"} <= set(labelled.blocks)


def test_formats_end_to_end(make_config: ConfigFactory) -> None:
    config = make_config(oraculos={"por_tamano": 2})
    generate(config, build_selection(config, ["ghz", "qft", "randomcircuit"], 2, 4))
    generate_oracles(config, ["grover", "dj"], range(3, 5))
    normalize(config)
    deduplicate(config)
    report = build_formats(config, RESERVED)

    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS).set_index("id")
    labelled = metadata[metadata["variante_comp"] == "etiquetada"]
    usable = metadata[(metadata["ruta_canonical"] != "") & (metadata["exclusion"] == "")]
    assert len(labelled) == report.labelled > 0
    assert set(labelled["exclusion"]) == {"etiquetada"}
    assert report.circuits == len(usable) + len(labelled)
    assert all(row.consistent for row in report.consistency)
    assert any(row.labels_f7 is not None for row in report.consistency)

    features = pd.read_parquet(config.rutas.features / FEATURES_FILE)
    assert len(features) == report.circuits and features["id"].is_unique
    for column in ("n_qubits", "profundidad", "n_puertas", "prop_rz", "n_cx", "grafo_diametro"):
        assert column in features.columns
    lines = (config.rutas.tokens / TOKENS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == report.circuits

    dataset = grafo_dag.IndexedGraphDataset(config.rutas.graphs / DAG_DIR)
    assert len(dataset) == report.circuits
    vocabulary = json.loads((config.rutas.graphs / "etiquetas.json").read_text(encoding="utf-8"))
    assert "randomcircuit" not in vocabulary["clases"]
    for graph in dataset:
        row = metadata.loc[graph.id]
        assert graph.num_nodes == int(row["n_puertas"])
        if row["algoritmo"] == "randomcircuit":
            assert graph.y_algoritmo.item() == graph.y_familia.item() == -2
        else:
            assert vocabulary["clases"][graph.y_algoritmo.item()] == row["clase"]
            assert vocabulary["familias"][graph.y_familia.item()] == row["familia"]
    qubit_graphs = grafo_dag.IndexedGraphDataset(config.rutas.graphs / QUBIT_DIR)
    assert all(g.num_nodes == int(metadata.loc[g.id, "n_qubits"]) for g in qubit_graphs)

    for circuit_id in labelled.index:
        labels = json.loads((config.rutas.gate_labels / f"{circuit_id}.json").read_text())
        assert len(labels["etiquetas"]) == int(labelled.loc[circuit_id, "n_puertas"])


def test_label_codes(make_config: ConfigFactory) -> None:
    """-1: no algorithm label (family only); -2: unknown (negative control, reserved)."""
    config = make_config()
    generate(
        config,
        build_selection(config, ["ghz", "randomcircuit", "qwalk", "hhl", "half_adder"], 3, 3),
    )
    normalize(config, levels=["alg"])
    deduplicate(config)
    build_formats(config, RESERVED)

    vocabulary = json.loads((config.rutas.graphs / "etiquetas.json").read_text(encoding="utf-8"))
    assert vocabulary["clases"] == ["ghz"]
    assert vocabulary["reservados"] == ["hrs_cumulative_multiplier", "hhl"]
    assert set(vocabulary["codigos"]) == {"-1", "-2"}
    family = {name: i for i, name in enumerate(vocabulary["familias"])}
    expected = {
        "ghz": (0, family["preparacion_estados"]),
        "half_adder": (-1, family["aritmetica"]),  # family label only
        "hhl": (-2, family["qft_fase"]),  # reserved, keeps its family
        "qwalk": (-2, -2),  # reserved without family
        "randomcircuit": (-2, -2),  # negative control
    }
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS).set_index("id")
    seen = set()
    for directory in (DAG_DIR, QUBIT_DIR):
        for graph in grafo_dag.IndexedGraphDataset(config.rutas.graphs / directory):
            algorithm = metadata.loc[graph.id, "algoritmo"]
            assert (graph.y_algoritmo.item(), graph.y_familia.item()) == expected[algorithm]
            seen.add(algorithm)
    assert seen == set(expected)
