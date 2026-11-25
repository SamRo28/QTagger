"""Tests of the GNN (phase 11): edge variants, model and traces, loss, schedule, sampler and
training with checkpoints (resuming equals training without interruption)."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from qcclass.entrenamiento.bloques import CrossValidationPool, FoldTraining
from qcclass.entrenamiento.gnn import (
    RESULT_FILE,
    TrainingSettings,
    hyperparameter_fingerprint,
    learning_rate_factor,
    run_directory,
    train_block,
)
from qcclass.formatos.comun import NO_LABEL, UNKNOWN, LabelVocabulary
from qcclass.formatos.construir import DAG_DIR
from qcclass.formatos.grafo_dag import INDEX_FILE, NODE_FEATURES, save_graph
from qcclass.modelos.aristas import EDGE_DIM, EdgeTransform
from qcclass.modelos.gnn import CircuitGNN, GNNConfig, MultiTaskLoss

CLASSES, FAMILIES = ("a", "b", "c"), ("f", "g")


def _graph(rng: np.random.Generator, nodes: int, label: int) -> Data:
    gates = rng.integers(0, 4, nodes)
    x = np.zeros((nodes, len(NODE_FEATURES)), dtype=np.float32)
    x[np.arange(nodes), gates] = 1
    angle = rng.uniform(-np.pi, np.pi, nodes) * (gates == 0)
    x[:, 4], x[:, 5] = np.sin(angle) + 0.3 * label, np.cos(angle)
    x[:, 6] = np.linspace(0, 1, nodes)
    edges = np.array([(i, i + 1) for i in range(nodes - 1)], dtype=np.int64).reshape(-1, 2).T
    roles = rng.integers(0, 2, (edges.shape[1], 4)).astype(np.float32)
    return Data(
        x=torch.tensor(x),
        edge_index=torch.tensor(edges),
        edge_attr=torch.tensor(roles),
        num_nodes=nodes,
    )


def _config(**changes) -> GNNConfig:
    values = {
        "node_dim": len(NODE_FEATURES),
        "edge_dim": EDGE_DIM,
        "num_classes": 3,
        "num_families": 2,
        **changes,
    }
    return GNNConfig(**values)


# --------------------------------------------------------------------------------------------
# Edge variants
# --------------------------------------------------------------------------------------------


def test_edge_variants() -> None:
    data = _graph(np.random.default_rng(0), 5, 0)
    data.edge_attr = torch.tensor([[1.0, 0, 0, 1]] * 4)  # control at the source, target at end
    full = EdgeTransform("completo")(data)
    assert full.edge_index.shape == (2, 8) and full.edge_attr.shape == (8, EDGE_DIM)
    assert torch.equal(full.edge_index[:, 4:], data.edge_index.flip(0))
    assert full.edge_attr[0].tolist() == [1, 0, 0, 1, 1, 0]
    assert full.edge_attr[4].tolist() == [0, 1, 1, 0, 0, 1]  # roles of its own ends, backward
    forward = EdgeTransform("solo_adelante")(data)
    assert forward.edge_index.shape == (2, 4) and forward.edge_attr[:, 4].eq(1).all()
    assert EdgeTransform("sin_direccion")(data).edge_attr[:, 4:].eq(0).all()
    no_roles = EdgeTransform("sin_rol")(data).edge_attr
    assert no_roles[:, :4].eq(0).all() and no_roles[:, 4:].sum() == 8
    no_angles = EdgeTransform("sin_angulos")(data)
    assert no_angles.x[:, 4:6].eq(0).all() and torch.equal(no_angles.x[:, :4], data.x[:, :4])
    assert torch.equal(data.x, _graph(np.random.default_rng(0), 5, 0).x)  # input untouched
    with pytest.raises(ValueError, match="variant"):
        EdgeTransform("otra")


# --------------------------------------------------------------------------------------------
# Model and traces
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("layer", "pooling", "virtual", "jk"),
    [
        ("gine", "media_max", False, True),
        ("gatv2", "atencion", True, True),
        ("gine", "atencion", True, False),
        ("gatv2", "media_max", False, False),
    ],
)
def test_model_outputs_and_traces(layer: str, pooling: str, virtual: bool, jk: bool) -> None:
    rng = np.random.default_rng(1)
    transform = EdgeTransform()
    graphs = [transform(_graph(rng, n, 0)) for n in (4, 7, 1)]
    batch = Batch.from_data_list(graphs)
    config = _config(
        tipo_capa=layer,
        agregacion=pooling,
        nodo_virtual=virtual,
        jumping_knowledge=jk,
        capas=3,
        dim=16,
    )
    model = CircuitGNN(config).eval()
    output = model(batch, trazas=True)
    assert output.algoritmo.shape == (3, 3) and output.familia.shape == (3, 2)
    traces = output.trazas
    assert len(traces["nodos"]) == 4 and all(t.shape == (12, 16) for t in traces["nodos"])
    pooled = 32 if pooling == "media_max" else 16
    assert traces["resumen"].shape == (3, pooled)
    assert torch.allclose(traces["prob_algoritmo"].sum(1), torch.ones(3))
    assert (traces["jumping_knowledge"] is not None) == jk
    assert (traces["nodo_virtual"] is not None) == virtual
    if virtual:
        assert len(traces["nodo_virtual"]) == 3 and traces["nodo_virtual"][0].shape == (3, 16)
    if layer == "gatv2":
        assert len(traces["atencion_capas"]) == 3
        assert traces["atencion_capas"][0]["pesos"].shape[1] == config.cabezas_atencion
    else:
        assert traces["atencion_capas"] is None
    if pooling == "atencion":
        weights = traces["atencion_agregacion"]
        per_graph = torch.zeros(3).index_add_(0, batch.batch, weights)
        assert torch.allclose(per_graph, torch.ones(3), atol=1e-5)
    else:
        assert traces["atencion_agregacion"] is None
    assert model(batch).trazas is None
    described = model.describe()
    assert described["parametros"] == sum(p.numel() for p in model.parameters())
    assert described["capas"][0]["nombre"] == "entrada"


def test_model_configuration_is_checked() -> None:
    with pytest.raises(ValueError, match="tipo_capa"):
        _config(tipo_capa="gcn")
    with pytest.raises(ValueError, match="multiple"):
        _config(tipo_capa="gatv2", dim=66, cabezas_atencion=4)
    with pytest.raises(ValueError, match="unknown"):
        GNNConfig.from_mapping(
            {"capas": 3, "suma": True}, node_dim=9, edge_dim=6, num_classes=3, num_families=2
        )


def test_loss_ignores_family_only_and_rejects_unknown() -> None:
    from qcclass.modelos.gnn import GNNOutput

    loss = MultiTaskLoss(torch.ones(3), torch.ones(2), lambda_familia=0.5)
    output = GNNOutput(torch.randn(3, 3), torch.randn(3, 2))
    y_alg, y_fam = torch.tensor([0, NO_LABEL, 2]), torch.tensor([0, 1, 1])
    total, algorithm, family = loss(output, y_alg, y_fam)
    expected = torch.nn.functional.cross_entropy(output.algoritmo[[0, 2]], y_alg[[0, 2]])
    assert algorithm == pytest.approx(float(expected), rel=1e-5)
    assert float(total) == pytest.approx(algorithm + 0.5 * family, rel=1e-5)
    only_family, _, _ = loss(output, torch.full((3,), NO_LABEL), y_fam)
    assert float(only_family) == pytest.approx(0.5 * family, rel=1e-5)
    without_family = MultiTaskLoss(torch.ones(3), torch.ones(2), lambda_familia=0)
    assert without_family(output, y_alg, y_fam)[2] == 0
    with pytest.raises(ValueError, match="-2"):
        loss(output, torch.tensor([0, UNKNOWN, 1]), y_fam)
    with pytest.raises(ValueError, match="-2"):
        loss(output, y_alg, torch.tensor([0, UNKNOWN, 1]))


def test_learning_rate_schedule() -> None:
    assert learning_rate_factor(0, 100, 0.05) == pytest.approx(0, abs=1e-6)
    assert learning_rate_factor(2.5, 100, 0.05) == pytest.approx(0.5)
    assert learning_rate_factor(5, 100, 0.05) == pytest.approx(1)
    assert learning_rate_factor(52.5, 100, 0.05) == pytest.approx(0.5)
    assert learning_rate_factor(100, 100, 0.05) == pytest.approx(0, abs=1e-9)
    assert learning_rate_factor(0, 10, 0) == pytest.approx(1)


def test_run_directory_depends_on_hyperparameters_and_block(tmp_path: Path) -> None:
    a = {"modelo": {"dim": 64}, "lr": 1e-3}
    assert hyperparameter_fingerprint(a) == hyperparameter_fingerprint({"lr": 1e-3, **a})
    assert run_directory(tmp_path, "e", a, 0) != run_directory(tmp_path, "e", a, 1)
    assert run_directory(tmp_path, "e", a, 0) != run_directory(tmp_path, "e", {**a, "lr": 2}, 0)


# --------------------------------------------------------------------------------------------
# Training on a synthetic pool
# --------------------------------------------------------------------------------------------


@pytest.fixture
def synthetic(tmp_path: Path) -> tuple[CrossValidationPool, FoldTraining, Path]:
    rng = np.random.default_rng(2)
    graphs = tmp_path / "graphs"
    rows, index = [], []
    for g in range(30):
        label = g % 3
        for variant in range(2):
            circuit = f"c{g}_{variant}"
            nodes = int(rng.integers(3, 40))
            save_graph(_graph(rng, nodes, label), graphs / DAG_DIR / f"{circuit}.pt")
            index.append({"id": circuit, "n_nodos": nodes})
            family_only = g == 29
            rows.append(
                {
                    "id": circuit,
                    "group_id": f"c{g}",
                    "bloque": g % 5,
                    "clase": CLASSES[label],
                    "familia": FAMILIES[label % 2],
                    "y_algoritmo": NO_LABEL if family_only else label,
                    "y_familia": label % 2,
                    "solo_evaluacion": False,
                    "desconocido": False,
                }
            )
    pd.DataFrame(index).to_csv(graphs / DAG_DIR / INDEX_FILE, index=False)
    pool = CrossValidationPool(pd.DataFrame(rows), LabelVocabulary(CLASSES, FAMILIES, (), ()), 5)
    blocks = pool.rows["bloque"].to_numpy()
    fold = FoldTraining(
        block=0,
        train_index=np.flatnonzero(blocks != 0),
        held_out_index=np.flatnonzero(blocks == 0),
        weights={"algoritmo": {"a": 1.0, "b": 1.5, "c": 0.5}, "familia": {}},
    )
    return pool, fold, graphs


def _train(synthetic, directory: Path, **kwargs):
    pool, fold, graphs = synthetic
    training = TrainingSettings(
        epocas=3,
        tasa_aprendizaje=1e-2,
        regularizacion_pesos=1e-4,
        presupuesto_nodos=120,
        evaluar_cada=2,
    )
    return train_block(
        pool=pool,
        fold=fold,
        graphs_dir=graphs,
        model_settings={"tipo_capa": "gine", "dim": 16, "capas": 2, "dropout": 0.1},
        training=training,
        seed=7,
        device="cpu",
        directory=directory,
        **kwargs,
    )


def test_training_writes_results_and_skips_when_done(synthetic, tmp_path: Path) -> None:
    epochs = []
    result = _train(synthetic, tmp_path / "run", on_epoch=lambda e, m: epochs.append((e, m)))
    assert [e for e, _ in epochs] == [1, 2, 3]
    assert "f1_macro_algoritmo" in epochs[1][1] and "f1_macro_algoritmo" not in epochs[0][1]
    assert 0 <= result.f1_macro_algoritmo <= 1
    stored = json.loads((tmp_path / "run" / RESULT_FILE).read_text())
    assert stored["huella"] and len(stored["historial"]) == 3
    table = pd.read_parquet(tmp_path / "run" / "predicciones.parquet")
    assert len(table) == 12 and set(table["bloque"]) == {0}
    again = _train(synthetic, tmp_path / "run", on_epoch=lambda e, m: epochs.append((e, m)))
    assert len(epochs) == 3 and again.f1_macro_algoritmo == result.f1_macro_algoritmo


def test_resuming_equals_continuous_training(synthetic, tmp_path: Path) -> None:
    continuous = _train(synthetic, tmp_path / "continuo")
    partial = _train(synthetic, tmp_path / "cortado", stop_after=1)
    assert len(partial.history) == 1 and math.isnan(partial.f1_macro_algoritmo)
    resumed = _train(synthetic, tmp_path / "cortado")
    assert resumed.resumed_from == 1
    for a, b in zip(continuous.history, resumed.history, strict=True):
        assert a["perdida"] == pytest.approx(b["perdida"], rel=1e-5)
    first = pd.read_parquet(tmp_path / "continuo" / "predicciones.parquet")
    second = pd.read_parquet(tmp_path / "cortado" / "predicciones.parquet")
    columns = [c for c in first.columns if c.startswith("p_")]
    assert np.allclose(first[columns], second[columns], atol=1e-5)


def test_checkpoints_of_other_hyperparameters_are_refused(synthetic, tmp_path: Path) -> None:
    pool, fold, graphs = synthetic
    _train(synthetic, tmp_path / "run", stop_after=1)
    other = TrainingSettings(
        epocas=3, tasa_aprendizaje=5e-3, regularizacion_pesos=1e-4, presupuesto_nodos=120
    )
    with pytest.raises(ValueError, match="other hyperparameters"):
        train_block(
            pool=pool,
            fold=fold,
            graphs_dir=graphs,
            model_settings={"dim": 16},
            training=other,
            seed=7,
            device="cpu",
            directory=tmp_path / "run",
        )


def test_unknown_circuits_never_reach_training(synthetic, tmp_path: Path) -> None:
    pool, fold, graphs = synthetic
    pool.rows.loc[fold.train_index[0], "y_algoritmo"] = UNKNOWN
    with pytest.raises(ValueError, match="-2"):
        _train(synthetic, tmp_path / "run")


def test_training_batches_respect_the_node_budget(synthetic) -> None:
    from qcclass.entrenamiento.gnn import PoolGraphs, _loader
    from qcclass.modelos.lotes import graph_sizes

    pool, fold, graphs = synthetic
    rows = pool.rows.iloc[fold.train_index]
    sizes = graph_sizes(graphs / DAG_DIR).loc[rows["id"]].to_numpy()
    dataset = PoolGraphs(
        graphs / DAG_DIR,
        list(rows["id"]),
        rows["y_algoritmo"].to_numpy(),
        rows["y_familia"].to_numpy(),
        EdgeTransform(),
    )
    loader, _ = _loader(
        dataset, sizes, 60, shuffle=True, seed=0, workers=0, device="cpu", max_batches=None
    )
    seen = 0
    for batch in loader:
        assert batch.num_nodes <= 60 or batch.num_graphs == 1
        assert batch.edge_attr.shape[1] == EDGE_DIM
        seen += batch.num_graphs
    assert seen == len(rows)


# --------------------------------------------------------------------------------------------
# Search (phase 12)
# --------------------------------------------------------------------------------------------


def test_search_configurations(repo_root: Path, tmp_path: Path) -> None:
    from qcclass.config import ConfigError
    from qcclass.entrenamiento.busqueda_gnn import load_search_config, split_params

    folder = repo_root / "configs" / "experimentos"
    search = load_search_config(folder / "gnn_busqueda.yaml")
    assert search.trials == 50 and search.blocks is None and search.report_every == 10
    model, training = split_params(
        search,
        {
            "tipo_capa": "gatv2",
            "dim": 128,
            "capas": 8,
            "epocas": 150,
            "dropout": 0.2,
            "tasa_aprendizaje": 1e-3,
            "regularizacion_pesos": 1e-4,
        },
    )
    assert model["cabezas_atencion"] == 4 and model["capas"] == 8
    assert training.epocas == 150 and training.evaluar_cada == 10 and training.precision_mixta
    assert load_search_config(folder / "gnn_prueba_reintentos.yaml").blocks == (0, 1)
    broken = (folder / "gnn_busqueda.yaml").read_text().replace("  dropout:", "  goteo:")
    (tmp_path / "roto.yaml").write_text(broken)
    with pytest.raises(ConfigError, match="goteo"):
        load_search_config(tmp_path / "roto.yaml")


def test_intermediate_steps() -> None:
    from qcclass.entrenamiento.busqueda_gnn import block_step, progress_step

    assert [progress_step(e, 50) for e in (10, 50)] == [20, 100]
    assert [progress_step(e, 150) for e in (30, 150)] == [20, 100]
    assert block_step(0) == 1000 and block_step(4) > progress_step(150, 150)


SMALL_SEARCH = """
nombre: s
experimento_mlflow: s
semilla: 3
intentos: 3
bloques: [0, 1]
poda: {intentos_iniciales: 10, informe_cada_epocas: 1}
estimacion: {tras_intentos: 1, limite_horas: 0.000001}
espacio:
  dim: {tipo: categorica, valores: [8, 16]}
  capas: {tipo: fijo, valor: 2}
  tasa_aprendizaje: {tipo: fijo, valor: 0.01}
  regularizacion_pesos: {tipo: fijo, valor: 0.0001}
  epocas: {tipo: fijo, valor: 2}
fijos:
  modelo: {tipo_capa: gine}
  entrenamiento: {presupuesto_nodos: 120, evaluar_cada: 1}
"""


def test_search_objective_dedup_and_estimate(synthetic, tmp_path: Path) -> None:
    import optuna

    from qcclass.entrenamiento.busqueda_gnn import (
        OOF_FILE,
        SearchContext,
        estimate,
        load_search_config,
        make_objective,
    )
    from qcclass.seguimiento.reintentos import RetryingStorage

    pool, fold, graphs = synthetic
    (tmp_path / "s.yaml").write_text(SMALL_SEARCH)
    search = load_search_config(tmp_path / "s.yaml")
    blocks = pool.rows["bloque"].to_numpy()
    folds = [
        FoldTraining(k, np.flatnonzero(blocks != k), np.flatnonzero(blocks == k), fold.weights)
        for k in (0, 1)
    ]
    provenance = type("P", (), {"machine": "ci", "tags": {}, "machine_description": {}})()
    config = type("C", (), {"rutas": type("R", (), {"graphs": graphs})()})()
    storage = RetryingStorage(optuna.storages.InMemoryStorage())
    ctx = SearchContext(
        config=config,
        search=search,
        pool=pool,
        folds=folds,
        storage=storage,
        provenance=provenance,
        device="cpu",
        workers=0,
        output=tmp_path / "gnn",
        log_to_mlflow=False,
    )
    study = optuna.create_study(study_name="s", storage=storage, direction="maximize")
    objective = make_objective(ctx, study)
    study.enqueue_trial({"dim": 8})
    study.enqueue_trial({"dim": 8})  # same hyperparameters: not trained again
    study.optimize(objective, n_trials=2)
    first, second = study.trials
    assert first.state.name == "COMPLETE" and len(first.user_attrs["f1_bloques"]) == 2
    assert second.user_attrs["repite_intento"] == 0 and second.value == first.value
    assert set(first.intermediate_values) == {50, 100, 1000, 1001}
    directory = tmp_path / "gnn" / "s" / first.user_attrs["huella"]
    oof = pd.read_parquet(directory / OOF_FILE)
    assert set(oof["bloque"]) == {0, 1} and len(oof) == 24
    assert (directory / "bloque_1" / "puntos").is_dir()
    figures = estimate(study, search)
    assert figures["intentos_terminados"] == 2 and figures["intentos_restantes"] == 1
    assert figures["supera_limite"]


# --------------------------------------------------------------------------------------------
# Micro-batches with gradient accumulation
# --------------------------------------------------------------------------------------------


def _accumulation_setup(seed: int = 4):
    from qcclass.entrenamiento.microlotes import AccumulatingStep

    rng = np.random.default_rng(seed)
    transform = EdgeTransform()
    graphs = []
    for g in range(12):
        data = transform(_graph(rng, int(rng.integers(5, 30)), g % 3))
        data.y_algoritmo = torch.tensor([NO_LABEL if g == 5 else g % 3])
        data.y_familia = torch.tensor([g % 2])
        graphs.append(data)
    batch = Batch.from_data_list(graphs)

    def stepper(micro: int, *, minimum: int = 1):
        torch.manual_seed(0)
        model = CircuitGNN(_config(dim=16, capas=2, nodo_virtual=True, agregacion="atencion"))
        model.eval()  # BatchNorm with running statistics: the only per-micro-batch part
        loss = MultiTaskLoss(torch.tensor([1.0, 2.0, 0.5]), torch.tensor([1.0, 3.0]), 0.7)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        step = AccumulatingStep(
            model,
            loss,
            optimizer,
            scaler,
            device="cpu",
            amp=False,
            micro_nodes=micro,
            min_micro_nodes=minimum,
        )
        return model, step

    return batch, stepper


def test_accumulated_gradient_equals_the_whole_batch() -> None:
    from qcclass.entrenamiento.microlotes import split_batch

    batch, stepper = _accumulation_setup()
    gradients, losses = {}, {}
    for parts in (1, 2, 4):
        micro = batch.num_nodes if parts == 1 else math.ceil(batch.num_nodes / parts)
        assert len(split_batch(batch, micro)) >= parts
        model, step = stepper(micro)
        result = step(batch)
        assert result.microlotes == len(split_batch(batch, micro))
        gradients[parts] = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )
        losses[parts] = result.total
    # reference: the loss of the whole batch with the (non-accumulating) forward of the loss
    model, _ = stepper(batch.num_nodes)
    loss = MultiTaskLoss(torch.tensor([1.0, 2.0, 0.5]), torch.tensor([1.0, 3.0]), 0.7)
    total, _, _ = loss(model(batch), batch.y_algoritmo, batch.y_familia)
    total.backward()
    reference = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    for parts in (1, 2, 4):
        assert torch.allclose(gradients[parts], reference, rtol=1e-4, atol=1e-6), parts
        assert losses[parts] == pytest.approx(float(total), rel=1e-5)


def test_split_batch_keeps_every_graph_in_order() -> None:
    from qcclass.entrenamiento.microlotes import split_batch

    batch, _ = _accumulation_setup()
    parts = split_batch(batch, 60)
    assert sum(p.num_graphs for p in parts) == batch.num_graphs
    assert all(p.num_nodes <= 60 or p.num_graphs == 1 for p in parts)
    joined = torch.cat([p.y_algoritmo for p in parts])
    assert torch.equal(joined, batch.y_algoritmo)
    assert split_batch(batch, batch.num_nodes)[0] is batch


def test_out_of_memory_halves_the_micro_batch_and_then_fails(monkeypatch) -> None:
    batch, stepper = _accumulation_setup()
    model, step = stepper(batch.num_nodes, minimum=40)
    original = model.forward
    limit = {"nodes": 70}

    def forward(data, **kwargs):  # noqa: ANN001, ANN202
        if data.num_nodes > limit["nodes"]:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(data, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    result = step(batch)
    assert step.micro_nodes <= 70 and result.microlotes > 1
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    limit["nodes"] = 10  # below the minimum micro-batch: the error is raised
    with pytest.raises(torch.OutOfMemoryError):
        step(batch)
    assert step.micro_nodes == 40


def test_memory_model_fit_and_micro_batch_choice(tmp_path: Path) -> None:
    from qcclass.modelos.memoria import MemoryModel, features, initial_micro_nodes, model_path

    truth = np.array([0.0, 1e-5, 2e-5, 3e-5, 1e-6, 2e-6])
    configs = [
        {"tipo_capa": t, "dim": d, "capas": c, "nodo_virtual": v, "agregacion": a}
        for t in ("gine", "gatv2")
        for d in (64, 128)
        for c in (3, 8)
        for v, a in ((False, "media_max"), (True, "atencion"))
    ]
    measured = [{"modelo": m, "mb_por_nodo": float(features(m) @ truth)} for m in configs]
    model = MemoryModel.fit("rtx", measured, useful_mb=3300, margin=1.15)
    big = {
        "tipo_capa": "gatv2",
        "dim": 128,
        "capas": 8,
        "nodo_virtual": True,
        "agregacion": "atencion",
    }
    assert model.per_node_mb(big) == pytest.approx(float(features(big) @ truth), rel=1e-6)
    per_node = model.per_node_mb(big)
    chosen = model.micro_nodes(big, 100_000, 20_000)
    assert chosen in (100_000, 50_000, 25_000, 20_000)
    assert 1.15 * chosen * per_node <= 3300 or chosen == 20_000
    assert model.micro_nodes({"dim": 64, "capas": 3}, 100_000, 20_000) == 100_000
    path = model_path(tmp_path, "rtx")
    path.parent.mkdir(parents=True)
    model.save(path)
    assert MemoryModel.load(path).coefficients == model.coefficients
    assert initial_micro_nodes(tmp_path, "rtx", big, 100_000, 20_000, "cuda") == chosen
    assert initial_micro_nodes(tmp_path, "rtx", big, 100_000, 20_000, "cpu") is None
    assert initial_micro_nodes(tmp_path, "otro", big, 100_000, 20_000, "cuda") is None


def test_fingerprint_covers_configuration_and_code(synthetic, tmp_path: Path, monkeypatch) -> None:
    """Micro-batch, mixed precision, device and training code are part of the identity, and a
    result stored with another fingerprint is never reused."""
    from qcclass.entrenamiento import gnn as training_module
    from qcclass.entrenamiento.gnn import run_identity, training_code_version

    settings = TrainingSettings(epocas=2, tasa_aprendizaje=1e-2, regularizacion_pesos=1e-4)
    base = run_identity({"dim": 16}, settings, 7, device="cuda", micro_nodes=None)
    assert base["modelo"]["capas"] == 3 and base["codigo"] == training_code_version()
    assert base["microlote_inicial"] == 100_000 and base["precision_mixta_efectiva"]
    variants = [
        run_identity({"dim": 16}, settings, 7, device="cuda", micro_nodes=50_000),
        run_identity({"dim": 16}, settings, 7, device="cpu", micro_nodes=None),
        run_identity(
            {"dim": 16},
            TrainingSettings(
                epocas=2, tasa_aprendizaje=1e-2, regularizacion_pesos=1e-4, precision_mixta=False
            ),
            7,
            device="cuda",
            micro_nodes=None,
        ),
        run_identity({"dim": 16, "capas": 3}, settings, 7, device="cuda", micro_nodes=None),
    ]
    prints = [hyperparameter_fingerprint(v) for v in variants]
    assert prints[3] == hyperparameter_fingerprint(base)  # explicit default = same run
    assert len({hyperparameter_fingerprint(base), *prints[:3]}) == 4
    # a finished block is reused only with the same fingerprint
    first = _train(synthetic, tmp_path / "run")
    assert _train(synthetic, tmp_path / "run").f1_macro_algoritmo == first.f1_macro_algoritmo
    monkeypatch.setattr(training_module, "training_code_version", lambda: "otro-codigo")
    with pytest.raises(ValueError, match="never reused"):
        _train(synthetic, tmp_path / "run")
    other = run_identity({"dim": 16}, settings, 7, device="cpu", micro_nodes=None)
    assert other["codigo"] == "otro-codigo"


def test_search_ceiling_uses_the_storage_clock() -> None:
    import datetime as dt

    import optuna

    from qcclass.entrenamiento.busqueda_gnn import elapsed_hours

    study = optuna.create_study()
    assert elapsed_hours(study) == 0.0
    start = dt.datetime(2026, 11, 25, 10, 0)
    for begin, end in ((0, 3), (2, 75)):
        trial = optuna.trial.FrozenTrial(
            number=0,
            state=optuna.trial.TrialState.COMPLETE,
            value=0.5,
            datetime_start=start + dt.timedelta(hours=begin),
            datetime_complete=start + dt.timedelta(hours=end),
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
            trial_id=0,
        )
        study.add_trial(trial)
    assert elapsed_hours(study) == pytest.approx(75)


def test_cpu_threads_leave_free_cores(monkeypatch) -> None:
    import os

    from qcclass.entrenamiento.gnn_experimento import configure_cpu_threads

    before = torch.get_num_threads()
    monkeypatch.setenv("QCCLASS_NUCLEOS_LIBRES", "2")
    try:
        assert configure_cpu_threads("cpu") == max(1, (os.cpu_count() or 1) - 2)
        assert torch.get_num_threads() == max(1, (os.cpu_count() or 1) - 2)
        assert configure_cpu_threads("cuda") is None
    finally:
        torch.set_num_threads(before)


def test_missing_graphs_give_a_clear_error(tmp_path: Path) -> None:
    from qcclass.modelos.lotes import graph_sizes

    with pytest.raises(ValueError, match="graphs are missing"):
        graph_sizes(tmp_path / "dag")
