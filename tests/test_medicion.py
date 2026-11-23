"""Tests for the node-budget sampler, the reference model and ``qcclass medir`` (Phase M)."""

import json
import random
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch
from typer.testing import CliRunner

from qcclass.cli import app
from qcclass.config import DatasetConfig
from qcclass.formatos.comun import NO_LABEL, UNKNOWN, GateRecord
from qcclass.formatos.grafo_dag import (
    EDGE_FEATURES,
    INDEX_FILE,
    NODE_FEATURES,
    dag_graph,
    save_graph,
)
from qcclass.medicion.medir import (
    MeasurementOptions,
    output_path,
    projection,
    run_measurement,
)
from qcclass.medicion.resumen import write_summary
from qcclass.modelos.lotes import NodeBudgetBatchSampler, budget_batches, graph_sizes
from qcclass.modelos.referencia import (
    Labels,
    MultiTaskLoss,
    ReferenceGNN,
    count_parameters,
)
from qcclass.versionado import access_log_path, write_version

ConfigFactory = Callable[..., DatasetConfig]

CLASSES = ["ghz", "qft"]
FAMILIES = ["preparacion_estados", "qft_fase"]


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------


def _check_budget(batches: list[list[int]], sizes: list[int], budget: int) -> None:
    for batch in batches:
        assert sum(sizes[i] for i in batch) <= budget or len(batch) == 1


def test_no_batch_exceeds_the_budget_except_a_lone_big_graph() -> None:
    rng = random.Random(0)
    sizes = [rng.randint(0, 900) for _ in range(500)] + [2500, 1001, 1000]
    budget = 1000
    sampler = NodeBudgetBatchSampler(sizes, budget, seed=3)
    for _ in range(3):
        batches = list(sampler)
        _check_budget(batches, sizes, budget)
        assert sorted(i for batch in batches for i in batch) == list(range(len(sizes)))
        oversized = [batch for batch in batches if sum(sizes[i] for i in batch) > budget]
        assert sorted(oversized) == [[500], [501]]  # 2500 and 1001 nodes, each alone


def test_sampler_reshuffles_every_epoch_reproducibly() -> None:
    sizes = list(range(1, 200))
    first, second = (
        NodeBudgetBatchSampler(sizes, 500, seed=1),
        NodeBudgetBatchSampler(sizes, 500, seed=1),
    )
    epoch_one, epoch_two = list(first), list(first)
    assert epoch_one != epoch_two
    assert list(second) == epoch_one and list(second) == epoch_two
    limited = NodeBudgetBatchSampler(sizes, 500, seed=1, max_batches=4)
    assert len(limited) == 4 and len(list(limited)) == 4


def test_budget_batches_is_greedy() -> None:
    assert budget_batches([3, 3, 3, 9, 1], [0, 1, 2, 3, 4], 6) == [[0, 1], [2], [3], [4]]


def test_graph_sizes_computed_once_when_missing(tmp_path: Path) -> None:
    gates = [GateRecord("x", (0,), None), GateRecord("cx", (0, 1), None)]
    for name, count in (("a", 1), ("b", 2)):
        save_graph(dag_graph(gates[:count], 2), tmp_path / f"{name}.pt")
    pd.DataFrame({"id": ["a", "b"]}).to_csv(tmp_path / INDEX_FILE, index=False)
    assert graph_sizes(tmp_path).to_dict() == {"a": 1, "b": 2}
    assert list(pd.read_csv(tmp_path / INDEX_FILE).columns) == ["id", "n_nodos"]
    (tmp_path / "a.pt").unlink()  # the second call must not load any graph
    assert graph_sizes(tmp_path).to_dict() == {"a": 1, "b": 2}


# --------------------------------------------------------------------------------------------
# Model and loss
# --------------------------------------------------------------------------------------------


def _random_graph(rng: random.Random, gates: int, label: tuple[int, int], qubits: int = 3):
    records = []
    for _ in range(gates):
        if rng.random() < 0.4:
            a, b = rng.sample(range(qubits), 2)
            records.append(GateRecord("cx", (a, b), None))
        else:
            records.append(GateRecord("rz", (rng.randrange(qubits),), rng.uniform(-3, 3)))
    graph = dag_graph(records, qubits)
    graph.y_algoritmo = torch.tensor([label[0]])
    graph.y_familia = torch.tensor([label[1]])
    return graph


def test_reference_model_shapes_and_parameters() -> None:
    model = ReferenceGNN(len(NODE_FEATURES), len(EDGE_FEATURES), 20, 6)
    rng = random.Random(1)
    batch = Batch.from_data_list([_random_graph(rng, n, (0, 0)) for n in (5, 0, 12)])
    algorithm, family = model(batch)
    assert algorithm.shape == (3, 20) and family.shape == (3, 6)
    # input 9*64+64; 3 x (GIN nn 2*(64*64+64) + edge lin 4*64+64 + BN 2*64); heads
    expected = 640 + 3 * (8320 + 320 + 128) + (8256 + 1300) + (8256 + 390)
    assert count_parameters(model) == expected
    wider = ReferenceGNN(len(NODE_FEATURES), len(EDGE_FEATURES), 20, 6, dim=32, layers=5)
    assert len(wider.convs) == 5 and wider.algorithm_head[-1].in_features == 32


def test_loss_ignores_missing_algorithm_and_rejects_unknown() -> None:
    loss = MultiTaskLoss(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 1.0]))
    logits = (torch.randn(3, 2), torch.randn(3, 2))
    parts = loss(logits, torch.tensor([0, NO_LABEL, 1]), torch.tensor([0, 1, 1]))
    expected = torch.nn.functional.cross_entropy(
        logits[0][[0, 2]], torch.tensor([0, 1]), weight=torch.tensor([1.0, 3.0])
    )
    assert parts.algorithm == pytest.approx(float(expected))
    only_family = loss(logits, torch.tensor([NO_LABEL] * 3), torch.tensor([0, 1, 1]))
    assert only_family.algorithm == 0.0 and torch.isfinite(only_family.total)
    with pytest.raises(ValueError, match="unknown"):
        loss(logits, torch.tensor([0, UNKNOWN, 1]), torch.tensor([0, 1, 1]))


def test_projection() -> None:
    epochs = [{"epoca": 1, "nodos_s": 500.0}, {"epoca": 2, "nodos_s": 1000.0}]
    result = projection(epochs, 3_600_000)  # one hour per epoch
    assert result["fuente"] == "epoca_2" and result["cache_caliente"]
    assert result["entrenamiento_completo_h"] == {"50": 50.0, "100": 100.0, "200": 200.0}
    assert result["optuna_h"] == {"20": 20 * 5 * 100 * 0.8, "50": 50 * 5 * 100 * 0.8}


# --------------------------------------------------------------------------------------------
# Whole measurement on a tiny synthetic dataset
# --------------------------------------------------------------------------------------------


@pytest.fixture
def tiny(make_config: ConfigFactory) -> DatasetConfig:
    """A versioned dataset of 40 DAG graphs with train/val/test and an F7 variant."""
    config = make_config()
    rng = random.Random(7)
    rows, splits = [], []
    directory = config.rutas.graphs / "dag"
    for index in range(40):
        circuit_id = f"c{index:02d}"
        label = (index % 2, index % 2) if index % 5 else (NO_LABEL, index % 2)
        save_graph(_random_graph(rng, rng.randint(0, 60), label), directory / f"{circuit_id}.pt")
        rows.append({"id": circuit_id, "algoritmo": CLASSES[index % 2]})
        conjunto = "train" if index < 28 else ("val" if index < 34 else "test")
        splits.append({"id": circuit_id, "conjunto": conjunto, "solo_evaluacion": index == 3})
    pd.DataFrame({"id": [r["id"] for r in rows]}).to_csv(directory / INDEX_FILE, index=False)
    config.rutas.metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.rutas.metadata, index=False)
    config.rutas.splits.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(splits).to_csv(config.rutas.splits / "aleatorio.csv", index=False)
    weights = {"algoritmo": {"ghz": 1.0, "qft": 1.0}, "familia": dict.fromkeys(FAMILIES, 1.0)}
    (config.rutas.splits / "balanceo_aleatorio.json").write_text(
        json.dumps({"max_bases_por_clase": None, "pesos": weights}), encoding="utf-8"
    )
    vocabulary = {
        "clases": CLASSES,
        "familias": FAMILIES,
        "reservados": [],
        "desconocidos_sin_familia": ["randomcircuit"],
    }
    (config.rutas.graphs / "etiquetas.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    write_version(config, "dataset-prueba")
    return config


def test_measurement_on_cpu_writes_the_json(tiny: DatasetConfig) -> None:
    options = MeasurementOptions(presupuesto_nodos=150, epocas=2, equipo="pruebas")
    path, result = run_measurement(tiny, options, device="cpu")
    assert path == output_path(tiny, "pruebas", "cpu", False) and path.is_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["version_dataset"] == "dataset-prueba"
    assert stored["datos"]["grafos"] == 27  # 28 train minus the evaluation-only variant
    assert stored["equipo"]["torch"] == torch.__version__
    assert [epoch["epoca"] for epoch in stored["epocas"]] == [1, 2]
    for epoch in stored["epocas"]:
        assert epoch["grafos"] == 27 and epoch["nodos"] == stored["datos"]["nodos_por_epoca"]
        assert epoch["tiempo_total_s"] >= epoch["tiempo_computo_s"] > 0
        assert epoch["ram_max_mb"] > 0 and epoch["memoria_gpu_max_mb"] is None
    assert stored["peor_caso"]["ok"]
    assert stored["proyeccion"]["fuente"] == "epoca_2"
    assert stored["mps_fallback"] is None and stored["barrido"] is None
    assert not access_log_path(tiny).exists()  # the test set was never read


def test_measurement_aborts_on_modified_data(tiny: DatasetConfig) -> None:
    from qcclass.versionado import VersionError

    (tiny.rutas.splits / "aleatorio.csv").write_text("id,conjunto,solo_evaluacion\n")
    with pytest.raises(VersionError):
        run_measurement(tiny, MeasurementOptions(epocas=1), device="cpu")


def test_cli_medir_sweep_and_summary(tiny: DatasetConfig, tmp_path: Path) -> None:
    config_path = str(tmp_path / "configs" / "dataset.yaml")
    runner = CliRunner()
    args = ["medir", "--config", config_path, "--dispositivo", "cpu", "--equipo", "pruebas"]
    result = runner.invoke(
        app, [*args, "--epocas", "1", "--presupuesto-nodos", "150", "--max-lotes", "2", "--barrido"]
    )
    assert result.exit_code == 0, result.output
    assert "Reference model:" in result.output and "parameters" in result.output
    stored = json.loads(output_path(tiny, "pruebas", "cpu", False).read_text())
    assert stored["epocas"][0]["lotes"] == 2
    sweep = stored["barrido"]
    assert [b["presupuesto_nodos"] for b in sweep["presupuestos"]] == [
        50_000,
        100_000,
        200_000,
        400_000,
    ]
    assert all(b["ok"] and b["lotes"] == 50 for b in sweep["presupuestos"])
    assert sweep["mayor_presupuesto_ok"] == 400_000
    summary = runner.invoke(app, ["medir", "--config", config_path, "--resumen"])
    assert summary.exit_code == 0, summary.output
    text = (tiny.rutas.resultados / "medicion" / "RESUMEN.md").read_text(encoding="utf-8")
    assert "| pruebas | cpu |" in text and "Barrido de presupuestos" in text
    bad_amp = runner.invoke(app, [*args, "--amp"])
    assert bad_amp.exit_code != 0


def test_cli_verificar(tiny: DatasetConfig, tmp_path: Path) -> None:
    config_path = str(tmp_path / "configs" / "dataset.yaml")
    ok = CliRunner().invoke(app, ["verificar", "--config", config_path])
    assert ok.exit_code == 0 and "matches dataset-prueba" in ok.output
    (tiny.rutas.graphs / "etiquetas.json").write_text("{}", encoding="utf-8")
    bad = CliRunner().invoke(app, ["verificar", "--config", config_path])
    assert bad.exit_code == 1 and "modificado data/graphs/etiquetas.json" in bad.output


def test_summary_needs_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_summary(tmp_path, "RESUMEN.md")


def test_labels_read(tiny: DatasetConfig) -> None:
    labels = Labels.read(tiny.rutas.graphs / "etiquetas.json")
    assert labels.classes == tuple(CLASSES) and labels.unknown == ("randomcircuit",)
