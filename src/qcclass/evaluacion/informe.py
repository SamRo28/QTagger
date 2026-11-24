"""Evaluation of a prediction table: metrics, intervals, breakdowns, JSON and Markdown.

:func:`evaluate` computes, for the algorithm and the family heads, the metrics of
:mod:`qcclass.evaluacion.metricas` (plus top-k when probabilities are present), their group
bootstrap intervals, the per-class scores, the confusion matrix, the metrics of every
cross-validation block (when the table has ``bloque``) and the breakdowns (when the table
has the metadata columns of :func:`predicciones.attach_metadata`):

* by level (``nivel``), by size (``n_logico``, the logical size of the group) and by family;
* grover by ``construccion_mcx`` and dj by ``tipo_oraculo_dj``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qcclass.circuitos import write_text_atomic
from qcclass.config import EvaluationConfig
from qcclass.evaluacion.bootstrap import bootstrap_intervals, group_counts
from qcclass.evaluacion.metricas import (
    METRICS,
    PRIMARY_METRIC,
    confusion,
    per_class,
    scores_from_confusion,
    top_k_hits,
)
from qcclass.evaluacion.predicciones import (
    BLOCK_COLUMN,
    HEADS,
    check_table,
    head_arrays,
    head_rows,
)
from qcclass.semillas import derive_seed

BREAKDOWNS = (  # (name, column, restriction on ``algoritmo`` or None)
    ("nivel", "nivel", None),
    ("tamano", "n_logico", None),
    ("familia", "familia", None),
    ("grover_construccion_mcx", "construccion_mcx", "grover"),
    ("dj_tipo_oraculo", "tipo_oraculo_dj", "dj"),
)


def _subset_scores(rows: pd.DataFrame, head: str, n_classes: int) -> dict:
    matrix = confusion(rows[f"y_{head}"], rows[f"pred_{head}"], n_classes)
    return {"n": len(rows), **scores_from_confusion(matrix)}


def breakdowns(rows: pd.DataFrame, head: str, n_classes: int) -> dict[str, list[dict]]:
    """Metrics of every value of each breakdown column present in ``rows``."""
    result = {}
    for name, column, algorithm in BREAKDOWNS:
        if column not in rows.columns:
            continue
        subset = rows if algorithm is None else rows[rows["algoritmo"] == algorithm]
        subset = subset[subset[column].astype(str) != ""]
        if subset.empty:
            continue
        entries = []
        for value, part in subset.groupby(column, sort=True):
            entries.append({"valor": str(value), **_subset_scores(part, head, n_classes)})
        result[name] = entries
    return result


def block_scores(rows: pd.DataFrame, head: str, n_classes: int) -> dict | None:
    """Metrics of every cross-validation block and mean / standard deviation over blocks."""
    if BLOCK_COLUMN not in rows.columns:
        return None
    per_block = {
        str(block): _subset_scores(part, head, n_classes)
        for block, part in rows.groupby(BLOCK_COLUMN, sort=True)
    }
    summary = {}
    for metric in METRICS:
        values = np.array([scores[metric] for scores in per_block.values()])
        summary[metric] = {"media": float(values.mean()), "desviacion": float(values.std(ddof=0))}
    return {"bloques": per_block, "resumen": summary}


def evaluate_head(
    table: pd.DataFrame,
    head: str,
    class_names: tuple[str, ...],
    config: EvaluationConfig,
    seed_key: str,
) -> dict:
    """Every result of one head (see the module docstring)."""
    n_classes = len(class_names)
    y_true, y_pred, probabilities, groups = head_arrays(table, head, n_classes)
    matrix = confusion(y_true, y_pred, n_classes)
    top_name = f"exactitud_top{config.top_k}"
    hits = top_k_hits(y_true, probabilities, config.top_k) if probabilities is not None else None
    counts = group_counts(y_true, y_pred, groups, n_classes, hits)
    intervals = bootstrap_intervals(
        counts,
        resamples=config.bootstrap.remuestreos,
        level=config.bootstrap.nivel,
        seed=derive_seed(config.semilla, "bootstrap", seed_key, head),
        top_k_name=top_name if hits is not None else None,
    )
    rows = head_rows(table, head)
    return {
        "n": int(len(y_true)),
        "grupos": int(len(counts.groups)),
        "metricas": intervals,
        "por_clase": per_class(matrix, class_names),
        "matriz_confusion": {"clases": list(class_names), "valores": matrix.tolist()},
        "por_bloque": block_scores(rows, head, n_classes),
        "desgloses": breakdowns(rows, head, n_classes),
    }


def evaluate(
    table: pd.DataFrame,
    class_names: dict[str, tuple[str, ...]],
    config: EvaluationConfig,
    *,
    name: str,
) -> dict:
    """Evaluation of both heads of a prediction table (``class_names`` per head)."""
    check_table(table)
    return {
        "nombre": name,
        "metrica_principal": f"algoritmo.{PRIMARY_METRIC}",
        "bootstrap": {
            "remuestreos": config.bootstrap.remuestreos,
            "nivel": config.bootstrap.nivel,
            "unidad": "group_id",
        },
        "solo_evaluacion_excluidas": int(table["solo_evaluacion"].astype(bool).sum()),
        **{head: evaluate_head(table, head, class_names[head], config, name) for head in HEADS},
    }


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------


def _pct(value: float) -> str:
    return "—" if value is None or np.isnan(value) else f"{100 * value:.1f}"


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def report_markdown(report: dict) -> str:
    """Markdown rendering of :func:`evaluate` (percentages, 95 % intervals)."""
    level = round(100 * report["bootstrap"]["nivel"])
    lines = [
        f"# Evaluación: {report['nombre']}",
        "",
        f"Métrica principal: F1 macro de algoritmo. Intervalos del {level} % por bootstrap de "
        f"grupos ({report['bootstrap']['remuestreos']} remuestreos). Variantes "
        f"`solo_evaluacion` excluidas: {report['solo_evaluacion_excluidas']}.",
    ]
    for head in HEADS:
        result = report[head]
        lines += ["", f"## {head.capitalize()} (n = {result['n']}, {result['grupos']} grupos)", ""]
        metric_rows = [
            [name, _pct(v["valor"]), f"[{_pct(v['ic_inf'])}, {_pct(v['ic_sup'])}]"]
            for name, v in result["metricas"].items()
        ]
        lines += _markdown_table(["Métrica (%)", "Valor", f"IC {level} %"], metric_rows)
        if result["por_bloque"]:
            summary = result["por_bloque"]["resumen"]
            lines += ["", "Por bloque (media ± desviación):"]
            lines += [
                f"- {metric}: {_pct(v['media'])} ± {_pct(v['desviacion'])}"
                for metric, v in summary.items()
            ]
        lines += ["", "### Por clase", ""]
        lines += _markdown_table(
            ["Clase", "Precisión", "Exhaustividad", "F1", "Soporte"],
            [
                [
                    c["clase"],
                    _pct(c["precision"]),
                    _pct(c["exhaustividad"]),
                    _pct(c["f1"]),
                    str(c["soporte"]),
                ]
                for c in result["por_clase"]
            ],
        )
        for name, entries in result["desgloses"].items():
            lines += ["", f"### Desglose: {name}", ""]
            lines += _markdown_table(
                ["Valor", "n", "Exactitud", "Exactitud balanceada", "F1 macro"],
                [
                    [
                        e["valor"],
                        str(e["n"]),
                        _pct(e["exactitud"]),
                        _pct(e["exactitud_balanceada"]),
                        _pct(e["f1_macro"]),
                    ]
                    for e in entries
                ],
            )
    return "\n".join(lines) + "\n"


def write_report(report: dict, directory: Path, name: str) -> tuple[Path, Path]:
    """Write ``<name>.json`` and ``<name>.md`` in ``directory``."""
    directory = Path(directory)
    json_path, markdown_path = directory / f"{name}.json", directory / f"{name}.md"
    write_text_atomic(json.dumps(report, indent=2, ensure_ascii=False) + "\n", json_path)
    write_text_atomic(report_markdown(report), markdown_path)
    return json_path, markdown_path
