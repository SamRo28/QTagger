"""Tests for the common evaluation module (Phase 9)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    top_k_accuracy_score,
)

from qcclass.config import BootstrapConfig, EvaluationConfig, load_evaluation_config
from qcclass.evaluacion.bootstrap import (
    bootstrap_intervals,
    group_counts,
    holm,
    paired_bootstrap,
)
from qcclass.evaluacion.informe import evaluate, write_report
from qcclass.evaluacion.metricas import confusion, scores_from_confusion, top_k_hits
from qcclass.evaluacion.predicciones import (
    aggregate_out_of_fold,
    attach_metadata,
    head_rows,
    make_table,
)

FAST = EvaluationConfig(semilla=1, bootstrap=BootstrapConfig(remuestreos=400, nivel=0.95), top_k=3)


def _scores(y_true: list[int], y_pred: list[int], n_classes: int) -> dict[str, float]:
    return scores_from_confusion(confusion(np.array(y_true), np.array(y_pred), n_classes))


# --------------------------------------------------------------------------------------------
# Didactic examples of the design document
# --------------------------------------------------------------------------------------------


def test_example_dj_qft() -> None:
    """300 DJ and 30 QFT, half of the QFT called DJ."""
    dj, qft = 0, 1
    y_true = [dj] * 300 + [qft] * 30
    y_pred = [dj] * 300 + [dj] * 15 + [qft] * 15
    scores = _scores(y_true, y_pred, 2)
    assert round(100 * scores["exactitud"], 1) == 95.5
    assert round(100 * scores["exactitud_balanceada"], 1) == 75.0
    assert round(100 * scores["f1_macro"], 1) == 82.1


def test_example_dj_qft_ghz() -> None:
    """300 DJ, 30 QFT and 30 GHZ, 60 DJ called QFT."""
    dj, qft, ghz = 0, 1, 2
    y_true = [dj] * 300 + [qft] * 30 + [ghz] * 30
    y_pred = [dj] * 240 + [qft] * 60 + [qft] * 30 + [ghz] * 30
    scores = _scores(y_true, y_pred, 3)
    assert round(100 * scores["exactitud_balanceada"], 1) == 93.3
    assert round(100 * scores["f1_macro"], 1) == 79.6


# --------------------------------------------------------------------------------------------
# Metrics against scikit-learn
# --------------------------------------------------------------------------------------------


def test_metrics_match_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    k = 6
    y_true = rng.integers(0, k - 1, 500)  # class 5 never true: only a wrong prediction
    y_pred = np.where(rng.random(500) < 0.6, y_true, rng.integers(0, k, 500))
    scores = _scores(y_true, y_pred, k)
    present = sorted(set(y_true))
    assert scores["exactitud"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert scores["f1_macro"] == pytest.approx(
        f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    )
    assert scores["exactitud_balanceada"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert scores["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred))
    probabilities = rng.dirichlet(np.ones(k), 500)
    assert top_k_hits(y_true, probabilities, 3).mean() == pytest.approx(
        top_k_accuracy_score(y_true, probabilities, k=3, labels=range(k))
    )


def test_batch_of_matrices_and_degenerate_mcc() -> None:
    perfect = confusion(np.array([0, 1, 1]), np.array([0, 1, 1]), 2)
    constant = confusion(np.array([0, 1, 1]), np.array([1, 1, 1]), 2)
    batch = scores_from_confusion(np.stack([perfect, constant]))
    assert batch["exactitud"].tolist() == pytest.approx([1.0, 2 / 3])
    assert batch["mcc"].tolist() == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError):
        confusion(np.array([0, 2]), np.array([0, 1]), 2)


# --------------------------------------------------------------------------------------------
# Exclusion rules and out-of-fold aggregation
# --------------------------------------------------------------------------------------------


def _table(block: int | None = None, offset: int = 0, n: int = 6) -> pd.DataFrame:
    ids = [f"c{offset + i}" for i in range(n)]
    y_algorithm = [0, 1, -1, -2, 1, 0][:n]
    y_family = [0, 1, 1, -2, 1, 0][:n]
    probabilities = {
        "algoritmo": np.tile([0.7, 0.2, 0.1], (n, 1)),
        "familia": np.tile([0.3, 0.7], (n, 1)),
    }
    return make_table(
        ids,
        [f"g{offset + i // 2}" for i in range(n)],
        {"algoritmo": y_algorithm, "familia": y_family},
        probabilities,
        evaluation_only=[False, False, False, False, True, False][:n],
        block=block,
    )


def test_exclusion_rules() -> None:
    table = _table()
    algorithm = head_rows(table, "algoritmo")
    family = head_rows(table, "familia")
    assert list(algorithm["id"]) == ["c0", "c1", "c5"]  # no -1, no -2, no solo_evaluacion
    assert list(family["id"]) == ["c0", "c1", "c2", "c5"]  # -1 counts in the family head
    assert (table["pred_algoritmo"] == 0).all() and (table["pred_familia"] == 1).all()


def test_out_of_fold_aggregation() -> None:
    combined = aggregate_out_of_fold([_table(0, 0), _table(1, 100)])
    assert len(combined) == 12 and set(combined["bloque"]) == {0, 1}
    with pytest.raises(ValueError, match="repeated"):
        aggregate_out_of_fold([_table(0, 0), _table(1, 0)])
    with pytest.raises(ValueError, match="bloque"):
        aggregate_out_of_fold([_table(None, 0)])
    moved = _table(1, 100)
    moved.loc[0, "group_id"] = "g0"  # a group present in two blocks
    with pytest.raises(ValueError, match="several blocks"):
        aggregate_out_of_fold([_table(0, 0), moved])


# --------------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------------


def test_bootstrap_resamples_whole_groups() -> None:
    # two groups: one fully right (10 rows), one fully wrong (10 rows)
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 10 + [0] * 10)
    counts = group_counts(y_true, y_pred, np.array(["a"] * 10 + ["b"] * 10), 2)
    result = bootstrap_intervals(counts, resamples=2000, level=0.95, seed=3)
    accuracy = result["exactitud"]
    assert accuracy["valor"] == pytest.approx(0.5)
    # resampling 2 groups gives only 0, 0.5 or 1: the interval spans [0, 1]
    assert accuracy["ic_inf"] == pytest.approx(0.0) and accuracy["ic_sup"] == pytest.approx(1.0)


def test_paired_bootstrap_and_holm() -> None:
    rng = np.random.default_rng(5)
    groups = np.repeat([f"g{i}" for i in range(200)], 3)
    y_true = rng.integers(0, 4, len(groups))
    good = np.where(rng.random(len(groups)) < 0.9, y_true, (y_true + 1) % 4)
    bad = np.where(rng.random(len(groups)) < 0.6, y_true, (y_true + 1) % 4)
    first, second = group_counts(y_true, good, groups, 4), group_counts(y_true, bad, groups, 4)
    result = paired_bootstrap(first, second, resamples=1000, level=0.95, seed=1)
    assert result["diferencia"] > 0.2 and result["ic_inf"] > 0 and result["p_valor"] < 0.01
    same = paired_bootstrap(first, first, resamples=200, level=0.95, seed=1)
    assert same["diferencia"] == 0 and same["p_valor"] == 1.0
    other = group_counts(y_true[:30], good[:30], groups[:30], 4)
    with pytest.raises(ValueError, match="same groups"):
        paired_bootstrap(first, other, resamples=10, level=0.95, seed=1)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.001]) == pytest.approx([0.5, 0.002])


# --------------------------------------------------------------------------------------------
# Whole evaluation, breakdowns and output
# --------------------------------------------------------------------------------------------


def test_evaluation_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 120
    algorithms = np.array(["grover", "dj", "qft"])[rng.integers(0, 3, n)]
    y_algorithm = pd.Series(algorithms).map({"grover": 0, "dj": 1, "qft": 2}).to_numpy()
    y_family = np.where(y_algorithm == 2, 1, 0)
    noisy = np.eye(3)[np.where(rng.random(n) < 0.8, y_algorithm, rng.integers(0, 3, n))]
    table = make_table(
        [f"c{i}" for i in range(n)],
        [f"g{i // 2}" for i in range(n)],
        {"algoritmo": y_algorithm, "familia": y_family},
        {"algoritmo": noisy, "familia": np.eye(2)[y_family]},
    )
    table["bloque"] = (np.arange(n) // 2) % 3
    metadata = pd.DataFrame(
        {
            "id": table["id"],
            "group_id": table["group_id"],
            "variante_comp": "c0",
            "n_qubits": rng.integers(3, 6, n).astype(str),
            "nivel": np.where(np.arange(n) % 2, "alg", "mapped"),
            "familia": np.where(y_family == 1, "qft_fase", "oraculos"),
            "algoritmo": algorithms,
            "construccion_mcx": np.where(algorithms == "grover", "vchain", ""),
            "tipo_oraculo_dj": np.where(algorithms == "dj", "lineal", ""),
        }
    )
    enriched = attach_metadata(table, metadata)
    names = {"algoritmo": ("grover", "dj", "qft"), "familia": ("oraculos", "qft_fase")}
    report = evaluate(enriched, names, FAST, name="prueba")
    algorithm = report["algoritmo"]
    assert set(algorithm["metricas"]) == {
        "exactitud",
        "exactitud_balanceada",
        "f1_macro",
        "mcc",
        "exactitud_top3",
    }
    f1 = algorithm["metricas"]["f1_macro"]
    assert f1["ic_inf"] <= f1["valor"] <= f1["ic_sup"]
    assert f1["valor"] == pytest.approx(
        f1_score(y_algorithm, noisy.argmax(axis=1), average="macro")
    )
    assert report["familia"]["metricas"]["exactitud"]["valor"] == 1.0
    assert set(algorithm["por_bloque"]["bloques"]) == {"0", "1", "2"}
    assert set(algorithm["desgloses"]) == {
        "nivel",
        "tamano",
        "familia",
        "grover_construccion_mcx",
        "dj_tipo_oraculo",
    }
    assert [e["valor"] for e in algorithm["desgloses"]["grover_construccion_mcx"]] == ["vchain"]
    json_path, markdown_path = write_report(report, tmp_path / "eval", "prueba")
    assert json.loads(json_path.read_text())["nombre"] == "prueba"
    text = markdown_path.read_text()
    assert "## Algoritmo" in text and "Desglose: dj_tipo_oraculo" in text


def test_evaluation_config_file(repo_root: Path) -> None:
    config = load_evaluation_config(repo_root / "configs" / "evaluacion.yaml")
    assert config.bootstrap.remuestreos == 10_000 and config.bootstrap.nivel == 0.95
    assert config.top_k == 3


def test_single_head_table() -> None:
    table = make_table(
        ["a", "b", "c"],
        ["g1", "g2", "g3"],
        {"familia": [0, 1, 1]},
        {"familia": np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])},
    )
    assert "pred_algoritmo" not in table.columns
    report = evaluate(table, {"familia": ("a", "b")}, FAST, name="solo_familia")
    assert report["cabezas"] == ["familia"] and "algoritmo" not in report
    assert report["familia"]["metricas"]["exactitud"]["valor"] == pytest.approx(2 / 3)
