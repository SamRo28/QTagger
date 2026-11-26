"""Phase 11b: data contract and provisional export for QTaggerDemo."""

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from sklearn.decomposition import PCA
from typer.testing import CliRunner

from qcclass.cli import app
from qcclass.demo import esquema as s
from qcclass.demo.desconocidos import Mahalanobis
from qcclass.demo.disposicion import block_extents, blocks_of, check_same_gates, columns
from qcclass.demo.exportar import (
    BINARY_FILE,
    CIRCUIT_FILE,
    Choice,
    Context,
    export_circuit,
    load_demo_config,
    model_info,
)
from qcclass.demo.modelo import TrainedModel, graph_input, integrated_gradients, run_traced
from qcclass.demo.trazas import aligned_pca, read_tensor, write_binary
from qcclass.formatos.comun import gate_sequence
from qcclass.formatos.etiquetas_puerta import labelled_canonical
from qcclass.formatos.grafo_dag import NODE_FEATURES
from qcclass.modelos.aristas import EDGE_DIM, EdgeTransform
from qcclass.modelos.gnn import CircuitGNN, GNNConfig

CLASSES = ("a", "b", "c")
FAMILIES = ("f", "g")


def _validator(model: type) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(s.json_schema(model))


def test_schema_files_are_up_to_date(repo_root: Path) -> None:
    for name, model in s.SCHEMAS.items():
        stored = (repo_root / s.SCHEMA_DIR / name).read_text(encoding="utf-8")
        assert stored == s.schema_text(model), f"run qcclass esquema-demo ({name})"
        jsonschema.Draft202012Validator.check_schema(json.loads(stored))
    assert s.SCHEMA_VERSION == "0.1.0"


def _blocky_circuit() -> QuantumCircuit:
    inner = QuantumCircuit(2, name="bloque_a")
    inner.h(0)
    inner.cx(0, 1)
    other = QuantumCircuit(1, name="bloque_b")
    other.rx(0.3, 0)
    circuit = QuantumCircuit(3)
    circuit.append(inner.to_gate(), [0, 2])
    circuit.append(other.to_gate(), [1])
    circuit.append(inner.to_gate(), [1, 2])
    circuit.measure_all()
    return circuit


def test_blocks_reproduce_the_labelled_canonical_circuit() -> None:
    circuit = _blocky_circuit()
    expanded = blocks_of(circuit)
    labelled = labelled_canonical(circuit)
    check_same_gates(gate_sequence(labelled.circuit), expanded.gates)
    names = sorted(name for name, _, _ in expanded.instructions)
    assert names == ["bloque_a", "bloque_a", "bloque_b"]
    names = [expanded.instructions[b][0] for b in expanded.block_of_gate]
    assert names == labelled.labels
    with pytest.raises(ValueError, match="differs"):
        check_same_gates(expanded.gates, expanded.gates[::-1])


def test_columns_first_free_column() -> None:
    gates = [(0,), (2,), (0, 2), (1,)]
    assert columns(gates, 3, span=False) == [0, 0, 1, 0]
    assert columns(gates, 3, span=True) == [0, 0, 1, 2]  # the CX line crosses qubit 1
    assert block_extents([0, 0, 1, 1], [0, 3, 2, 5]) == {0: (0, 3), 1: (2, 5)}


def test_binary_round_trip_and_aligned_projection(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tensors = {"a": rng.normal(size=(5, 3)), "b": rng.normal(size=7)}
    index = write_binary(tensors, tmp_path / "t.bin")
    assert [e.offset for e in index] == [0, 30]
    assert (tmp_path / "t.bin").stat().st_size == 2 * (15 + 7)
    for entry in index:
        np.testing.assert_allclose(
            read_tensor(tmp_path / "t.bin", entry), tensors[entry.name], rtol=1e-3, atol=1e-3
        )
    base = rng.normal(size=(40, 6)) * np.array([5, 2, 0.1, 0.1, 0.1, 0.1])
    flipped = base.copy()
    flipped[:, 0] *= -1  # PCA of this one has a reflected first axis
    (_, first, _), (_, second, _) = aligned_pca({"x": base, "y": flipped})
    np.testing.assert_allclose(first, second, atol=1e-9)


def test_mahalanobis_threshold_accepts_the_share_asked() -> None:
    rng = np.random.default_rng(1)
    means = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0]], dtype=float)
    labels = rng.integers(0, 3, 600)
    vectors = means[labels] + rng.normal(size=(600, 3))
    labels[:20] = -1  # family-only rows are skipped
    known = means[rng.integers(0, 3, 400)] + rng.normal(size=(400, 3))
    detector = Mahalanobis.fit(vectors, labels, CLASSES, known, 0.95)
    assert np.mean(detector.scores(known) <= detector.threshold) == pytest.approx(0.95, abs=0.01)
    far = np.array([[30.0, 30.0, 30.0]])
    assert detector.scores(far)[0] > detector.threshold
    assert detector.distances(means).argmin(axis=1).tolist() == [0, 1, 2]


def _trained(tmp_path: Path, **changes) -> TrainedModel:
    torch.manual_seed(0)
    config = GNNConfig(
        node_dim=len(NODE_FEATURES),
        edge_dim=EDGE_DIM,
        num_classes=len(CLASSES),
        num_families=len(FAMILIES),
        dim=16,
        capas=2,
        **changes,
    )
    model = CircuitGNN(config).eval()
    describe = model.describe()
    result = {
        "huella": "0123456789abcdef",
        "bloque": 0,
        "dispositivo": "cpu",
        "precision_mixta": "float32",
        "f1_macro_algoritmo": 0.5,
        "f1_macro_familia": 0.6,
        "historial": [],
        "identidad": {"codigo": "abc", "entrenamiento": {"epocas": 1}},
    }
    return TrainedModel(model, result, describe, tmp_path, EdgeTransform("completo"))


def test_integrated_gradients_complete(tmp_path: Path) -> None:
    trained = _trained(tmp_path)
    circuit = blocks_of(_blocky_circuit()).circuit
    graph = graph_input(gate_sequence(circuit), circuit.num_qubits, trained.transform)
    importance, error = integrated_gradients(trained, graph, 1, steps=128)
    assert importance.shape == (graph.num_nodes,)
    assert error < 0.05  # midpoint sum of a ReLU network


@pytest.mark.parametrize("changes", [{}, {"tipo_capa": "gatv2", "nodo_virtual": True}])
def test_circuit_export_is_valid_and_binary_matches_model(
    tmp_path: Path, repo_root: Path, changes: dict
) -> None:
    trained = _trained(tmp_path, **changes)
    demo = load_demo_config(repo_root / "configs/experimentos/demo_provisional.yaml", repo_root)
    demo = replace(demo, ig_steps=16, output=tmp_path / "out")
    rng = np.random.default_rng(2)
    pooled = 2 * trained.config.dim
    vectors = rng.normal(size=(60, pooled))
    detector = Mahalanobis.fit(vectors, rng.integers(0, 3, 60), CLASSES, vectors, 0.95)
    context = Context(
        config=None,  # only read for dataset circuits
        demo=demo,
        trained=trained,
        model_info=model_info(trained, demo),
        classes=CLASSES,
        families=FAMILIES,
        detector=detector,
        map_pca=PCA(n_components=2).fit(vectors),
        blocks={},
    )
    choice = Choice("unknown_prueba", "unknown", "test", None, _blocky_circuit())
    directory = tmp_path / "out" / "prueba"
    export = export_circuit(choice, context, directory)

    data = json.loads((directory / CIRCUIT_FILE).read_text(encoding="utf-8"))
    _validator(s.CircuitExport).validate(data)
    assert data["provisional"] is True and data["schema_version"] == s.SCHEMA_VERSION
    assert export.canonical.blocks_known
    assert {g.block for g in export.canonical.gates} == {0, 1, 2}
    assert len(export.trace.coordinates) == len(export.trace.stages)

    circuit = blocks_of(_blocky_circuit()).circuit
    graph = graph_input(gate_sequence(circuit), circuit.num_qubits, trained.transform)
    traces = run_traced(trained, graph)
    index = {t.name: t for t in export.trace.tensors}
    for name, expected in (
        ("capa_2", traces["nodos"][2]),
        ("resumen", traces["resumen"][0]),
        ("atributos_entrada", graph.x.numpy()),
    ):
        rebuilt = read_tensor(directory / BINARY_FILE, index[name])
        np.testing.assert_allclose(rebuilt, expected, rtol=2e-3, atol=2e-3)
    assert ("nodo_virtual" in index) == trained.config.nodo_virtual
    if trained.config.tipo_capa == "gatv2":
        assert export.trace.attention is not None


def test_global_example_is_valid(tmp_path: Path, repo_root: Path) -> None:
    trained = _trained(tmp_path)
    demo = load_demo_config(repo_root / "configs/experimentos/demo_provisional.yaml", repo_root)
    example = s.GlobalExport(
        provisional=True,
        generated_at="2026-11-26T00:00:00+00:00",
        versions=s.Versions(
            qcclass_commit="0" * 40,
            code_dirty=False,
            dataset="dataset-v1.1",
            dataset_fingerprint="f" * 64,
            libraries={"torch": "2.15.0"},
        ),
        model=model_info(trained, demo),
        classes=[s.ClassInfo(name="a", family="f", status="trainable")],
        families=[s.FamilyInfo(name="f", classes=["a"], algorithms=["a"])],
        gallery=[],
        selection_criteria=["test"],
        reference_map=s.Section(status="pending"),
        confusion=s.Section(status="pending"),
        baseline_ladder=s.Section(status="pending"),
        training_curves=s.Section(status="pending"),
        optuna_history=s.Section(status="pending", note="phase 12"),
        ablations=s.Section(status="pending"),
        decisions=s.Section(status="pending"),
        dataset=s.Section(status="pending"),
        measurements=s.Section(status="pending"),
    )
    _validator(s.GlobalExport).validate(json.loads(example.model_dump_json()))
    bad = json.loads(example.model_dump_json())
    bad["reference_map"]["status"] = "maybe"
    with pytest.raises(jsonschema.ValidationError):
        _validator(s.GlobalExport).validate(bad)


def test_export_demo_needs_provisional() -> None:
    result = CliRunner().invoke(app, ["exportar-demo"])
    assert result.exit_code == 1
    assert "--provisional" in result.output


class _Linear(torch.nn.Module):
    """``logit_c = sum_nodes x . w_c``: integrated gradients are exactly ``(x - x') . w``."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(weights)

    def forward(self, batch, trazas: bool = False):  # noqa: ANN001, ANN201
        from torch_geometric.utils import scatter

        from qcclass.modelos.gnn import GNNOutput

        per_node = batch.x @ self.weights
        logits = scatter(per_node, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="sum")
        return GNNOutput(logits, torch.zeros(batch.num_graphs, len(FAMILIES)))


@pytest.mark.parametrize("steps", [1, 7, 64])
def test_integrated_gradients_exact_on_a_linear_model(tmp_path: Path, steps: int) -> None:
    torch.manual_seed(3)
    weights = torch.randn(len(NODE_FEATURES), len(CLASSES))
    trained = replace(_trained(tmp_path), model=_Linear(weights).eval())
    circuit = blocks_of(_blocky_circuit()).circuit
    graph = graph_input(gate_sequence(circuit), circuit.num_qubits, trained.transform)
    for target in range(len(CLASSES)):
        importance, error = integrated_gradients(trained, graph, target, steps=steps)
        expected = (graph.x - torch.zeros_like(graph.x)) @ weights[:, target]  # baseline x' = 0
        np.testing.assert_allclose(importance, expected.numpy(), rtol=1e-5, atol=1e-6)
        assert error < 1e-5


def test_demo_reference_is_the_phase_11_reference(repo_root: Path) -> None:
    from qcclass.entrenamiento.gnn import hyperparameter_fingerprint, run_identity
    from qcclass.entrenamiento.gnn_experimento import load_gnn_experiment

    folder = repo_root / "configs" / "experimentos"
    phase11 = load_gnn_experiment(folder / "gnn_referencia.yaml")
    demo = load_gnn_experiment(folder / "demo_referencia.yaml")
    assert demo.mlflow_experiment not in ("gnn_fase11", "gnn_aleatorio_v1")
    assert (demo.model, demo.training, demo.seed) == (phase11.model, phase11.training, phase11.seed)
    identities = [
        run_identity(e.model, e.training, e.seed, device="cpu", micro_nodes=None)
        for e in (phase11, demo)
    ]
    assert hyperparameter_fingerprint(identities[0]) == hyperparameter_fingerprint(identities[1])
    config = load_demo_config(folder / "demo_provisional.yaml", repo_root)
    assert config.experiment.name == "demo_referencia.yaml"
    assert config.family_algorithms == {"oraculos": ("dj", "bv")}
    assert (config.readable_qubits, config.target_gates) == (10, 120)
