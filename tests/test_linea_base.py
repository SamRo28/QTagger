"""Tests for the baseline (phase 10): WL features, configuration, preparation, folds."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import scipy.sparse as sp

from qcclass.config import ConfigError, load_splits_config
from qcclass.entrenamiento.bloques import CrossValidationPool, fold_training, sample_weights
from qcclass.formatos.comun import LabelVocabulary
from qcclass.lineabase.caracteristicas import (
    WLSettings,
    _count_matrix,
    initial_label_name,
    initial_labels,
    wl_label_sequence,
)
from qcclass.lineabase.clasificadores import prepare
from qcclass.lineabase.configuracion import load_baseline_config, suggest
from qcclass.lineabase.resumen import _permuted
from qcclass.seguimiento.optuna_cliente import MACHINE_ATTR, fail_stale_trials

SETTINGS = WLSettings(h_max=3, bits_hash=20, divisiones_pi=16, tolerancia_angulo=1e-4)


def _node(gate: str, angle: float | None = None) -> list[float]:
    one_hot = [float(gate == g) for g in ("rz", "sx", "x", "cx")]
    sine, cosine = (math.sin(angle), math.cos(angle)) if angle is not None else (0.0, 1.0)
    return [*one_hot, sine, cosine, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------------------------
# WL
# --------------------------------------------------------------------------------------------


def test_initial_labels_and_names() -> None:
    x = np.array(
        [
            _node("rz", math.pi / 4),
            _node("rz", -3 * math.pi / 16),
            _node("rz", 0.123),
            _node("cx"),
            _node("sx"),
            _node("rz", math.pi),
        ],
        dtype=np.float32,
    )
    names = [initial_label_name(c, SETTINGS) for c in initial_labels(x, SETTINGS)]
    assert names == ["rz(4pi/16)", "rz(-3pi/16)", "rz(otro)", "cx", "sx", "rz(-16pi/16)"]


def _graph(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gates = rng.choice(["rz", "sx", "cx"], n)
    x = np.array([_node(g, rng.choice([0.5, math.pi / 2]) if g == "rz" else None) for g in gates])
    edges = np.array([(i, j) for i in range(n) for j in range(i + 1, min(n, i + 3))]).T
    roles = rng.integers(0, 2, (edges.shape[1], 4)).astype(np.float32)
    return x.astype(np.float32), edges, roles


def test_wl_counts_ignore_node_order_but_not_direction() -> None:
    rng = np.random.default_rng(0)
    x, edges, roles = _graph(rng, 12)
    order = rng.permutation(12)
    inverse = np.argsort(order)
    permuted = wl_label_sequence(x[order], inverse[edges], roles, SETTINGS)
    original = wl_label_sequence(x, edges, roles, SETTINGS)
    for a, b in zip(original, permuted, strict=True):
        assert sorted(a.tolist()) == sorted(b.tolist())
    reversed_edges = wl_label_sequence(x, edges[::-1], roles, SETTINGS)
    assert sorted(reversed_edges[1].tolist()) != sorted(original[1].tolist())
    assert sorted(reversed_edges[0].tolist()) == sorted(original[0].tolist())


def test_count_matrix_normalised_and_empty_graph() -> None:
    matrix = _count_matrix([np.array([3, 3, 7]), np.array([], dtype=np.int64)], 16)
    assert matrix.shape == (2, 16)
    assert matrix[0, 3] == pytest.approx(2 / 3) and matrix[0].sum() == pytest.approx(1.0)
    assert matrix[1].nnz == 0


# --------------------------------------------------------------------------------------------
# Configuration and preparation
# --------------------------------------------------------------------------------------------


def test_baseline_configuration(repo_root: Path, tmp_path: Path) -> None:
    config = load_baseline_config(repo_root / "configs" / "experimentos" / "linea_base.yaml")
    assert list(config.rungs) == ["e7", "f3", "wl"] and config.trials == 50
    assert config.rungs["wl"].wl.bits_hash == 20
    space = config.space("wl", "hist_gradient_boosting")
    assert {"h", "min_circuitos", "columnas_wl", "learning_rate"} <= set(space)
    assert config.space("wl", "random_forest")["max_features"]["valores"] == ["sqrt", "log2"]
    assert "h" not in config.space("f3", "random_forest")
    assert config.study_name("e7", "random_forest", "algoritmo") == (
        "linea_base_v1.e7.random_forest.algoritmo"
    )
    params = suggest(
        optuna.trial.FixedTrial({"C": 2.0, "transformacion": "log1p"}),
        config.space("e7", "regresion_logistica"),
    )
    assert params == {"C": 2.0, "transformacion": "log1p", "max_iter": 2000}
    broken = (
        (repo_root / "configs/experimentos/linea_base.yaml")
        .read_text()
        .replace("tipo: int, min: 1, max: 10}", "tipo: entero, min: 1, max: 10}")
    )
    (tmp_path / "roto.yaml").write_text(broken)
    with pytest.raises(ConfigError, match="tipo"):
        load_baseline_config(tmp_path / "roto.yaml")


def test_wl_columns_chosen_on_the_training_part_only() -> None:
    x = sp.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 5, 1],  # held out: column 3 only appears here
            ],
            dtype=float,
        )
    )
    train, held_out = np.array([0, 1, 2]), np.array([3])
    lr = prepare(x, train, held_out, "regresion_logistica", {"min_circuitos": 2})
    assert lr.columns.tolist() == [0, 1, 4]  # >= 2 training circuits; never column 3
    hgb = prepare(x, train, held_out, "hist_gradient_boosting", {"columnas_wl": 1})
    assert hgb.columns.tolist() == [0] and isinstance(hgb.train, np.ndarray)
    dense = prepare(np.ones((4, 2)), train, held_out, "random_forest", {})
    assert dense.columns is None and dense.train.shape == (3, 2)


def test_permuting_a_column_dense_and_sparse() -> None:
    rng = np.random.default_rng(1)
    dense = rng.random((6, 4)) * (rng.random((6, 4)) > 0.5)
    order = rng.permutation(6)
    expected = dense.copy()
    expected[:, 2] = dense[order, 2]
    assert np.allclose(_permuted(dense, 2, order), expected)
    assert np.allclose(_permuted(sp.csr_matrix(dense), 2, order).toarray(), expected)


# --------------------------------------------------------------------------------------------
# Folds: cap and weights inside each block
# --------------------------------------------------------------------------------------------


def _pool(n_groups: dict[str, int]) -> CrossValidationPool:
    rows = []
    for label, groups in n_groups.items():
        for g in range(groups):
            for variant in range(2):
                rows.append(
                    {
                        "id": f"{label}_{g}_{variant}",
                        "group_id": f"{label}_{g}",
                        "etiqueta": label,
                        "clase": label,
                        "familia": "f",
                        "desconocido": False,
                        "solo_evaluacion": False,
                        "bloque": g % 5,
                    }
                )
    table = pd.DataFrame(rows)
    classes = tuple(sorted(n_groups))
    table["y_algoritmo"] = table["clase"].map(classes.index)
    table["y_familia"] = 0
    vocabulary = LabelVocabulary(classes, ("f",), (), ())
    return CrossValidationPool(table, vocabulary, 5)


def test_fold_training_caps_and_weights(repo_root: Path) -> None:
    splits = load_splits_config(repo_root / "configs" / "splits.yaml")
    splits = replace(splits, balanceo=replace(splits.balanceo, factor_mediana=1))
    pool = _pool({"a": 50, "b": 10, "c": 10})
    fold = fold_training(pool, 0, splits)
    rows = pool.rows
    assert set(rows.loc[fold.held_out_index, "bloque"]) == {0}
    assert len(fold.held_out_index) == 2 * 14  # held-out block kept whole
    train = rows.loc[fold.train_index]
    assert (train["bloque"] != 0).all()
    per_label = train.groupby("etiqueta")["group_id"].nunique()
    assert per_label["a"] == 8 and per_label["b"] == 8  # cap = 1 x median (8)
    weights = fold.weights["algoritmo"]
    assert sum(weights.values()) / 3 == pytest.approx(1.0)
    w = sample_weights(pool, fold.train_index, "algoritmo", fold)
    assert w.shape == (len(fold.train_index),)


# --------------------------------------------------------------------------------------------
# Stale trials (no heartbeat in the gRPC proxy)
# --------------------------------------------------------------------------------------------


def test_stale_trials_of_this_machine_are_failed() -> None:
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(study_name="s", storage=storage)
    mine, other = study.ask(), study.ask()
    mine.set_user_attr(MACHINE_ATTR, "m3pro")
    other.set_user_attr(MACHINE_ATTR, "rtx3050")
    study.tell(study.ask(), 1.0)
    assert fail_stale_trials(study, storage, "m3pro") == [mine.number]
    states = {t.number: t.state.name for t in study.get_trials()}
    assert states == {0: "FAIL", 1: "RUNNING", 2: "COMPLETE"}
