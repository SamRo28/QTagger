"""Summary of the baseline: table of studies, finalists, permutation importance, RESUMEN.md.

Finalists: the best classifier of every rung by the cross-validated macro F1 of the algorithm
target. Their permutation importance is measured block by block: the model fitted on the
training part of a block is evaluated on the held-out block with one column permuted
(``repeticiones`` times); the importance is the mean drop of the macro F1 (same metric as the
evaluation module), averaged over blocks. Dense and sparse inputs are both supported. On WL
only the ``max_columnas_wl`` columns most frequent in each block's training part are
permuted.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from qcclass.circuitos import write_text_atomic
from qcclass.evaluacion.metricas import confusion, scores_from_confusion
from qcclass.lineabase.caracteristicas import bucket_name
from qcclass.lineabase.estudio import RESULT_FILE, STUDIES_DIR, StudyContext, fit_block
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

SUMMARY_FILE = "RESUMEN.md"
SUMMARY_TABLE = "resumen.csv"
TARGET_FOR_FINALISTS = "algoritmo"


def collect_results(output: Path, *, all_revisions: bool = False) -> pd.DataFrame:
    """One row per finished study (``estudios/*/resultado.json``): only the highest revision
    of every rung x classifier x target unless ``all_revisions``."""
    rows = []
    for path in sorted((output / STUDIES_DIR).glob(f"*/{RESULT_FILE}")):
        result = json.loads(path.read_text(encoding="utf-8"))
        oof = result["oof"]
        rows.append(
            {
                "peldano": result["peldano"],
                "clasificador": result["clasificador"],
                "objetivo": result["objetivo"],
                "revision": int(result.get("revision", 1)),
                "cv_f1_macro": result["cv_f1_macro"],
                "cv_f1_macro_desviacion": result["cv_f1_macro_desviacion"],
                **{f"oof_{m}": v["valor"] for m, v in oof.items()},
                **{f"oof_{m}_ic_inf": v["ic_inf"] for m, v in oof.items()},
                **{f"oof_{m}_ic_sup": v["ic_sup"] for m, v in oof.items()},
                "intentos": result["intentos"],
                "segundos_busqueda": result["segundos_busqueda"],
                "parametros": json.dumps(result["parametros"], ensure_ascii=False),
                "estudio": result["estudio"],
            }
        )
    table = pd.DataFrame(rows)
    if all_revisions or table.empty:
        return table
    latest = table.groupby(["peldano", "clasificador", "objetivo"])["revision"].transform("max")
    return table[table["revision"] == latest].reset_index(drop=True)


def finalists(table: pd.DataFrame) -> pd.DataFrame:
    """Best classifier of every rung by cross-validated algorithm macro F1."""
    algorithm = table[table["objetivo"] == TARGET_FOR_FINALISTS]
    best = algorithm.loc[algorithm.groupby("peldano")["cv_f1_macro"].idxmax()]
    return best.sort_values("cv_f1_macro").reset_index(drop=True)


def _macro_f1(y: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    return scores_from_confusion(confusion(y, predicted.astype(int), n_classes))["f1_macro"]


def _permuted(x: Any, column: int, order: np.ndarray) -> Any:
    """Copy of ``x`` with column ``column`` permuted by ``order`` (dense or CSR)."""
    if not sp.issparse(x):
        copy = np.array(x, copy=True)
        copy[:, column] = copy[order, column]
        return copy
    values = x[:, column].toarray().ravel()
    delta = values[order] - values
    rows = np.flatnonzero(delta)
    change = sp.csr_matrix((delta[rows], (rows, np.full(len(rows), column))), shape=x.shape)
    return (x + change).tocsr()


def permutation_importance(
    ctx: StudyContext, study: dict[str, Any], *, max_wl_columns: int, repeats: int
) -> pd.DataFrame:
    """Mean macro F1 drop per feature over the blocks (see the module docstring)."""
    rung, classifier, target = study["peldano"], study["clasificador"], study["objetivo"]
    params = study["parametros"]
    features = ctx.features.matrix(rung, params)
    names = ctx.features.names(rung)
    n_classes = len(ctx.pool.class_names(target))
    y = ctx.pool.rows[f"y_{target}"].to_numpy()
    drops: dict[int, list[float]] = {}
    for fold in ctx.folds:
        seed = derive_seed(ctx.baseline.seed, study["estudio"], "bloque", fold.block)
        fitted = fit_block(ctx, features, fold, target, classifier, params, seed)
        x, truth = fitted.held_out_input, y[fitted.held_out_rows]
        reference = _macro_f1(truth, fitted.model.predict(x), n_classes)
        columns = np.arange(x.shape[1])
        if fitted.columns is not None:  # WL: most frequent kept columns of the training part
            train_rows = fold.train_index[y[fold.train_index] >= 0]
            frequency = features[train_rows][:, fitted.columns].getnnz(axis=0)
            columns = np.argsort(-frequency, kind="stable")[:max_wl_columns]
        rng = np.random.default_rng(
            derive_seed(ctx.baseline.seed, study["estudio"], "perm", fold.block)
        )
        for column in columns:
            scores = [
                _macro_f1(
                    truth,
                    fitted.model.predict(_permuted(x, column, rng.permutation(len(truth)))),
                    n_classes,
                )
                for _ in range(repeats)
            ]
            original = int(fitted.columns[column]) if fitted.columns is not None else int(column)
            drops.setdefault(original, []).append(reference - float(np.mean(scores)))
    wl = ctx.features.wl() if names is None else None
    rows = [
        {
            "caracteristica": names[c] if names is not None else bucket_name(c, wl),
            "columna": c,
            "importancia": float(np.mean(v)),
            "desviacion_bloques": float(np.std(v)),
            "bloques": len(v),
        }
        for c, v in drops.items()
    ]
    return pd.DataFrame(rows).sort_values("importancia", ascending=False).reset_index(drop=True)


def _pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{100 * value:.1f}"


def _markdown(table: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(table.columns) + " |"
    return [
        header,
        "|" + "---|" * len(table.columns),
        *("| " + " | ".join(map(str, r)) + " |" for r in table.itertuples(index=False)),
    ]


def write_summary(
    output: Path,
    table: pd.DataFrame,
    best: pd.DataFrame,
    importances: dict[str, pd.DataFrame],
    header: dict[str, str],
) -> Path:
    """Write ``RESUMEN.md``, ``resumen.csv`` and ``importancia_<rung>.csv``."""
    table.to_csv(output / SUMMARY_TABLE, index=False)
    for rung, frame in importances.items():
        frame.to_csv(output / f"importancia_{rung}.csv", index=False)
    lines = [
        "# Línea base (fase 10)",
        "",
        f"Dataset {header['version_dataset']} ({header['huella_dataset'][:12]}), commit "
        f"`{header['git_commit'][:7]}`, equipo `{header['equipo']}`. Validación cruzada de 5 "
        "bloques sobre train + val de `aleatorio`; objetivo de Optuna = F1 macro medio de los "
        "bloques. Columnas «OOF»: predicciones fuera de bloque del mejor intento, con "
        "intervalos del 95 % por bootstrap de grupos. **Sin test.**",
        "",
    ]
    for target in ("algoritmo", "familia"):
        part = table[table["objetivo"] == target].sort_values(["peldano", "clasificador"])
        if part.empty:
            continue
        lines += [f"## Objetivo: {target}", ""]
        shown = pd.DataFrame(
            {
                "Peldaño": part["peldano"],
                "Clasificador": part["clasificador"],
                "F1 macro CV (%)": [
                    f"{_pct(m)} ± {_pct(s)}"
                    for m, s in zip(
                        part["cv_f1_macro"], part["cv_f1_macro_desviacion"], strict=True
                    )
                ],
                "F1 macro OOF [IC 95 %]": [
                    f"{_pct(v)} [{_pct(lo)}, {_pct(hi)}]"
                    for v, lo, hi in zip(
                        part["oof_f1_macro"],
                        part["oof_f1_macro_ic_inf"],
                        part["oof_f1_macro_ic_sup"],
                        strict=True,
                    )
                ],
                "Exactitud bal.": part["oof_exactitud_balanceada"].map(_pct),
                "Exactitud": part["oof_exactitud"].map(_pct),
                "Top-3": part.get(
                    "oof_exactitud_top3", pd.Series([None] * len(part), index=part.index)
                ).map(_pct),
                "MCC": part["oof_mcc"].map(_pct),
                "Rev.": part["revision"],
                "Intentos": part["intentos"].map(
                    lambda d: ", ".join(f"{k.lower()} {v}" for k, v in sorted(d.items()))
                ),
                "Búsqueda (min)": (part["segundos_busqueda"] / 60).round(1),
            }
        )
        lines += [*_markdown(shown), ""]
    lines += ["## Finalistas (mejor clasificador de cada peldaño, F1 macro de algoritmo)", ""]
    for row in best.itertuples(index=False):
        lines.append(
            f"- **{row.peldano}**: `{row.clasificador}`, F1 macro CV {_pct(row.cv_f1_macro)} %, "
            f"OOF {_pct(row.oof_f1_macro)} % [{_pct(row.oof_f1_macro_ic_inf)}, "
            f"{_pct(row.oof_f1_macro_ic_sup)}]. Parámetros: `{row.parametros}`"
        )
    for rung, frame in importances.items():
        lines += ["", f"## Importancia por permutación: {rung}", ""]
        top = frame.head(15).assign(
            importancia=lambda f: (100 * f["importancia"]).round(2),
            desviacion_bloques=lambda f: (100 * f["desviacion_bloques"]).round(2),
        )[["caracteristica", "importancia", "desviacion_bloques"]]
        top.columns = ["Característica", "Caída de F1 macro (puntos)", "Desv. entre bloques"]
        lines += _markdown(top)
    every = collect_results(output, all_revisions=True)
    replaced = every[~every["estudio"].isin(table["estudio"])]
    if len(replaced):
        lines += ["", "## Estudios sustituidos por una revisión (rango ampliado)", ""]
        for row in replaced.sort_values("estudio").itertuples(index=False):
            new = table[
                (table["peldano"] == row.peldano)
                & (table["clasificador"] == row.clasificador)
                & (table["objetivo"] == row.objetivo)
            ].iloc[0]
            lines.append(
                f"- `{row.estudio}`: F1 macro CV {_pct(row.cv_f1_macro)} % → "
                f"`{new.estudio}` {_pct(new.cv_f1_macro)} %"
            )
    total = every["segundos_busqueda"].sum() / 3600
    lines += ["", f"Tiempo total de búsqueda: {total:.1f} h.", ""]
    path = output / SUMMARY_FILE
    write_text_atomic("\n".join(lines), path)
    return path
