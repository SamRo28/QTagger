"""Diagnostics of the baseline (``qcclass diagnosticos-linea-base``), cross-validation only.

1. **Shuffled labels.** The WL finalist (algorithm target, best configuration) is refitted on
   every block with the algorithm labels permuted among the *groups* of the block's training
   part (all variants of a group keep one label, so the group structure is preserved); the
   held-out block keeps its true labels. A macro F1 near chance (1 / classes) means nothing
   leaks through the pipeline; above ``umbral_f1_macro`` the run stops
   (:class:`ShuffledLabelAlarm`).
2. **Structural twins.** The WL fingerprint of a circuit is the multiset of its WL node
   labels at iteration ``h`` (full 64-bit labels, not hash buckets; same initial labels as the
   WL rung). A held-out circuit has a twin when some circuit of the training part of its
   block has the same fingerprint. Reported: share of circuits with a twin per class, share
   of twins of another class, and the macro / per-class F1 of the three finalists on the
   circuits with and without a twin, with group bootstrap intervals.
3. **Confusion** of the WL finalist (most confused class pairs) and **F1 by level** of the
   three finalists.
4. **Range review.** Best value of every numeric hyperparameter of every study in the outer
   ``extremo`` fraction of its range (log scale when the range is), or at an end of an
   ordered numeric list; natural bounds are marked. The revisions of
   ``linea_base.yaml`` (``revisiones``) repeat those studies with the range widened.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from qcclass.circuitos import write_text_atomic
from qcclass.config import ConfigError, load_experiment_config
from qcclass.entrenamiento.bloques import head_index
from qcclass.evaluacion.bootstrap import CHUNK, _resample_weights, bootstrap_intervals, group_counts
from qcclass.evaluacion.metricas import confusion, scores_from_confusion
from qcclass.evaluacion.predicciones import attach_metadata, head_arrays, head_rows
from qcclass.formatos.construir import DAG_DIR
from qcclass.formatos.grafo_dag import load_graph
from qcclass.lineabase.caracteristicas import wl_label_sequence
from qcclass.lineabase.clasificadores import build, fit, prepare
from qcclass.lineabase.configuracion import BaselineConfig
from qcclass.lineabase.estudio import RESULT_FILE, STUDIES_DIR, StudyContext
from qcclass.lineabase.resumen import collect_results, finalists
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

TARGET = "algoritmo"
REPORT_NAME = "diagnosticos"
LEVELS = ("alg", "indep", "nativegates", "mapped")
REQUIRED = ("linea_base", "semilla", "barajado", "gemelos", "confusion", "rangos")


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Parsed ``diagnosticos_linea_base.yaml``."""

    baseline_path: Path
    seed: int
    shuffle_seeds: tuple[int, ...]
    shuffle_threshold: float
    twin_h: tuple[int, ...]
    confused_pairs: int
    range_fraction: float
    natural_bounds: dict[str, float]
    min_support: int
    raw: dict[str, Any]


def load_diagnostics_config(path: Path | str) -> DiagnosticsConfig:
    """Load ``diagnosticos_linea_base.yaml``.

    Raises:
        ConfigError: if a key is missing or invalid.
    """
    data = load_experiment_config(path, REQUIRED)
    try:
        config = DiagnosticsConfig(
            baseline_path=Path(data["linea_base"]),
            seed=int(data["semilla"]),
            shuffle_seeds=tuple(int(s) for s in data["barajado"]["semillas"]),
            shuffle_threshold=float(data["barajado"]["umbral_f1_macro"]),
            twin_h=tuple(int(h) for h in data["gemelos"]["h"]),
            confused_pairs=int(data["confusion"]["pares"]),
            range_fraction=float(data["rangos"]["extremo"]),
            natural_bounds={
                k: float(v) for k, v in (data["rangos"].get("limites_naturales") or {}).items()
            },
            min_support=int(data.get("soporte_minimo", 20)),
            raw=data,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc
    if not config.shuffle_seeds or not config.twin_h or min(config.twin_h) < 0:
        raise ConfigError(f"{Path(path).name}: barajado.semillas and gemelos.h must be set")
    if not 0 < config.range_fraction < 0.5:
        raise ConfigError(f"{Path(path).name}: rangos.extremo must be in (0, 0.5)")
    return config


class ShuffledLabelAlarm(RuntimeError):
    """The shuffled-label test scored above its threshold."""


# --------------------------------------------------------------------------------------------
# Bootstrap of per-class F1 (the macro metrics come from evaluacion.bootstrap)
# --------------------------------------------------------------------------------------------


def _f1_per_class(matrices: np.ndarray) -> np.ndarray:
    """Per-class F1 of a batch of confusion matrices (``B x K x K``; NaN when undefined)."""
    tp = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = matrices.sum(axis=2) + matrices.sum(axis=1)  # 2TP + FP + FN
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * tp / denominator, np.nan)


def per_class_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    *,
    resamples: int,
    level: float,
    seed: int,
) -> list[dict[str, float]]:
    """F1 of every class with a group bootstrap interval (``support`` = true rows)."""
    counts = group_counts(y_true, y_pred, groups, n_classes)
    k = n_classes
    point = _f1_per_class(counts.confusion.sum(axis=0).reshape(1, k, k))[0]
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, resamples, CHUNK):
        weights = _resample_weights(len(counts.groups), min(CHUNK, resamples - start), rng)
        samples.append(_f1_per_class((weights @ counts.confusion).reshape(-1, k, k)))
    values = np.concatenate(samples)
    tail = (1 - level) / 2 * 100
    support = np.bincount(np.asarray(y_true, int), minlength=k)
    result = []
    for c in range(k):
        column = values[:, c]
        low, high = (
            np.nanpercentile(column, [tail, 100 - tail])
            if np.isfinite(column).any()
            else (np.nan, np.nan)
        )
        result.append(
            {
                "soporte": int(support[c]),
                "f1": float(point[c]),
                "ic_inf": float(low),
                "ic_sup": float(high),
            }
        )
    return result


def _macro(
    ctx: StudyContext,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    key: str,
    classes: list[int] | None = None,
) -> dict[str, float]:
    """Macro F1 with a group bootstrap interval; with ``classes``, only rows of those true
    classes (a class with a handful of rows otherwise swings the macro average as it enters
    and leaves the resamples)."""
    n_classes = len(ctx.pool.class_names(TARGET))
    if classes is not None:
        keep = np.isin(y_true, classes)
        y_true, y_pred, groups = y_true[keep], y_pred[keep], groups[keep]
    if len(y_true) == 0:
        return {"n": 0, "valor": np.nan, "ic_inf": np.nan, "ic_sup": np.nan}
    counts = group_counts(y_true, y_pred, groups, n_classes)
    interval = bootstrap_intervals(
        counts,
        resamples=ctx.evaluation.bootstrap.remuestreos,
        level=ctx.evaluation.bootstrap.nivel,
        seed=derive_seed(ctx.evaluation.semilla, "diagnosticos", key),
    )["f1_macro"]
    return {"n": int(len(y_true)), "grupos": int(len(counts.groups)), **interval}


def _common_classes(y_true: np.ndarray, masks: list[np.ndarray], minimum: int) -> list[int]:
    """Classes with at least ``minimum`` rows in every subset."""
    size = int(y_true.max()) + 1
    counts = [np.bincount(y_true[m], minlength=size) for m in masks]
    return [c for c in range(size) if all(n[c] >= minimum for n in counts)]


# --------------------------------------------------------------------------------------------
# 1. Shuffled labels
# --------------------------------------------------------------------------------------------


def shuffled_labels(ctx: StudyContext, study: dict[str, Any], seeds: tuple[int, ...]) -> dict:
    """Block and pooled macro F1 of ``study`` refitted with shuffled training labels."""
    rung, classifier, params = study["peldano"], study["clasificador"], study["parametros"]
    features = ctx.features.matrix(rung, params)
    rows = ctx.pool.rows
    y = rows[f"y_{TARGET}"].to_numpy()
    groups = rows["group_id"].to_numpy()
    names = np.array(ctx.pool.class_names(TARGET))
    n_classes = len(names)
    runs = []
    for seed in seeds:
        block_f1, pooled = [], np.zeros((n_classes, n_classes))
        for fold in ctx.folds:
            train = head_index(ctx.pool, fold.train_index, TARGET)
            held_out = head_index(ctx.pool, fold.held_out_index, TARGET)
            unique, inverse = np.unique(groups[train], return_inverse=True)
            group_label = np.full(len(unique), -1)
            group_label[inverse] = y[train]
            if (group_label[inverse] != y[train]).any():
                raise ValueError("a training group has more than one algorithm label")
            rng = np.random.default_rng(
                derive_seed(ctx.baseline.seed, "barajado", seed, fold.block)
            )
            shuffled = rng.permutation(group_label)[inverse]
            prepared = prepare(features, train, held_out, classifier, params)
            model = build(
                classifier,
                params,
                derive_seed(ctx.baseline.seed, study["estudio"], "bloque", fold.block),
                sparse=sp.issparse(prepared.train),
            )
            table = fold.weights.get(TARGET)
            weights = (
                np.array([table.get(name, 1.0) for name in names[shuffled]]) if table else None
            )
            fit(model, prepared.train, shuffled, weights)
            matrix = confusion(y[held_out], model.predict(prepared.held_out).astype(int), n_classes)
            pooled += matrix
            block_f1.append(float(scores_from_confusion(matrix)["f1_macro"]))
            logger.info(
                "shuffled labels, seed %d, block %d: F1 %.4f", seed, fold.block, block_f1[-1]
            )
        runs.append(
            {
                "semilla": seed,
                "f1_bloques": block_f1,
                "f1_macro_media_bloques": float(np.mean(block_f1)),
                "f1_macro_oof": float(scores_from_confusion(pooled)["f1_macro"]),
            }
        )
    return {
        "estudio": study["estudio"],
        "parametros": params,
        "azar": 1 / n_classes,
        "ejecuciones": runs,
        "f1_macro_media": float(np.mean([r["f1_macro_media_bloques"] for r in runs])),
        "f1_macro_max": float(max(r["f1_macro_media_bloques"] for r in runs)),
    }


# --------------------------------------------------------------------------------------------
# 2. Structural twins
# --------------------------------------------------------------------------------------------


def _fingerprint(labels: np.ndarray) -> str:
    return hashlib.blake2b(np.sort(labels).tobytes(), digest_size=16).hexdigest()


def wl_fingerprints(ctx: StudyContext, hs: tuple[int, ...]) -> pd.DataFrame:
    """WL fingerprint of every pool circuit at every ``h`` (columns ``h<h>``; cached)."""
    rung = next(r for r in ctx.baseline.rungs.values() if r.source == "wl")
    settings = replace(rung.wl, h_max=max(hs))
    ids = list(ctx.pool.rows["id"])
    key = json.dumps(
        {"ajustes": settings.__dict__, "h": list(hs), "ids": ids}, sort_keys=True
    ).encode()
    path = ctx.features.cache / f"huellas_{hashlib.sha256(key).hexdigest()[:16]}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    rows = []
    for circuit_id in tqdm(ids, desc="WL fingerprints"):
        graph = load_graph(ctx.config.rutas.graphs / DAG_DIR / f"{circuit_id}.pt")
        sequence = wl_label_sequence(
            graph.x.numpy(), graph.edge_index.numpy(), graph.edge_attr.numpy(), settings
        )
        rows.append({"id": circuit_id, **{f"h{h}": _fingerprint(sequence[h]) for h in hs}})
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def twin_flags(ctx: StudyContext, fingerprints: pd.DataFrame, h: int) -> pd.DataFrame:
    """Per held-out circuit: has a twin in its block's training part, whether a twin belongs
    to another class, and which other classes (``otras_clases``, sorted target indices)."""
    rows = ctx.pool.rows
    prints = fingerprints.set_index("id").loc[rows["id"], f"h{h}"].to_numpy()
    y = rows[f"y_{TARGET}"].to_numpy()
    parts = []
    for fold in ctx.folds:
        train = head_index(ctx.pool, fold.train_index, TARGET)
        held_out = head_index(ctx.pool, fold.held_out_index, TARGET)
        labels = pd.DataFrame({"huella": prints[train], "y": y[train]}).groupby("huella")["y"]
        seen = labels.agg(lambda s: frozenset(s.tolist())).to_dict()
        twins = [seen.get(p) for p in prints[held_out]]
        parts.append(
            pd.DataFrame(
                {
                    "id": rows["id"].to_numpy()[held_out],
                    "gemelo": [t is not None for t in twins],
                    "gemelo_otra_clase": [
                        t is not None and bool(t - {label})
                        for t, label in zip(twins, y[held_out], strict=True)
                    ],
                    "otras_clases": [
                        tuple(sorted(t - {label})) if t is not None else ()
                        for t, label in zip(twins, y[held_out], strict=True)
                    ],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _cross_twins(labelled: pd.DataFrame, names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Share of the held-out circuits of every class with a twin of each other class."""
    sizes = labelled[f"y_{TARGET}"].value_counts()
    pairs = (
        labelled[[f"y_{TARGET}", "otras_clases"]].explode("otras_clases").dropna().value_counts()
    )
    return sorted(
        (
            {
                "clase": names[int(a)],
                "clase_gemelo": names[int(b)],
                "n_clase": int(sizes[a]),
                "circuitos": int(n),
                "porcentaje": float(n / sizes[a]),
            }
            for (a, b), n in pairs.items()
        ),
        key=lambda r: (-r["circuitos"], r["clase"], r["clase_gemelo"]),
    )


def _predictions(ctx: StudyContext, study: str) -> pd.DataFrame:
    path = ctx.output / STUDIES_DIR / study / "predicciones.parquet"
    return head_rows(pd.read_parquet(path), TARGET)


def twins(
    ctx: StudyContext, best: pd.DataFrame, hs: tuple[int, ...], min_support: int
) -> dict[str, Any]:
    """Twin shares per class and finalist scores with / without a twin (see the module)."""
    fingerprints = wl_fingerprints(ctx, hs)
    names = ctx.pool.class_names(TARGET)
    n_classes = len(names)
    boot = ctx.evaluation.bootstrap
    result: dict[str, Any] = {}
    for h in hs:
        flags = twin_flags(ctx, fingerprints, h)
        labelled = flags.merge(ctx.pool.rows[["id", f"y_{TARGET}"]], on="id")
        by_class = labelled.groupby(f"y_{TARGET}").agg(
            n=("gemelo", "size"),
            con_gemelo=("gemelo", "mean"),
            gemelo_otra_clase=("gemelo_otra_clase", "mean"),
        )
        entry: dict[str, Any] = {
            "con_gemelo": float(flags["gemelo"].mean()),
            "gemelo_otra_clase": float(flags["gemelo_otra_clase"].mean()),
            "por_clase": [
                {
                    "clase": names[int(c)],
                    "n": int(r.n),
                    "con_gemelo": float(r.con_gemelo),
                    "gemelo_otra_clase": float(r.gemelo_otra_clase),
                }
                for c, r in by_class.iterrows()
            ],
            "cruzados": _cross_twins(labelled, names),
            "finalistas": {},
        }
        for row in best.itertuples(index=False):
            table = _predictions(ctx, row.estudio).merge(flags, on="id", how="left")
            if table["gemelo"].isna().any():
                raise ValueError(f"{row.estudio}: predictions without twin flags")
            y_true, y_pred, _, groups = head_arrays(table, TARGET, n_classes)
            mask = table["gemelo"].to_numpy(bool)
            common = _common_classes(y_true, [mask, ~mask], min_support)
            entry["clases_comunes"] = [names[c] for c in common]
            entry["soporte_minimo"] = min_support
            parts = {}
            for label, part in (("con_gemelo", mask), ("sin_gemelo", ~mask)):
                key = f"{row.estudio}.h{h}.{label}"
                parts[label] = {
                    "macro": _macro(ctx, y_true[part], y_pred[part], groups[part], key),
                    "macro_comun": _macro(
                        ctx, y_true[part], y_pred[part], groups[part], f"{key}.comun", common
                    ),
                    "por_clase": per_class_intervals(
                        y_true[part],
                        y_pred[part],
                        groups[part],
                        n_classes,
                        resamples=boot.remuestreos,
                        level=boot.nivel,
                        seed=derive_seed(ctx.evaluation.semilla, "diagnosticos", key, "clases"),
                    ),
                }
            entry["finalistas"][row.peldano] = {"estudio": row.estudio, **parts}
        result[f"h{h}"] = entry
    return result


# --------------------------------------------------------------------------------------------
# 3. Confusion and levels
# --------------------------------------------------------------------------------------------


def confused_pairs(ctx: StudyContext, study: str, k: int) -> dict[str, Any]:
    """Confusion matrix of ``study`` and its ``k`` most confused class pairs (both ways)."""
    names = ctx.pool.class_names(TARGET)
    n_classes = len(names)
    y_true, y_pred, _, _ = head_arrays(_predictions(ctx, study), TARGET, n_classes)
    matrix = confusion(y_true, y_pred, n_classes).astype(int)
    support = matrix.sum(axis=1)
    pairs = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            if total := int(matrix[i, j] + matrix[j, i]):
                pairs.append(
                    {
                        "a": names[i],
                        "b": names[j],
                        "total": total,
                        "a_como_b": int(matrix[i, j]),
                        "b_como_a": int(matrix[j, i]),
                        "soporte_a": int(support[i]),
                        "soporte_b": int(support[j]),
                    }
                )
    pairs.sort(key=lambda p: (-p["total"], p["a"], p["b"]))
    return {
        "estudio": study,
        "clases": list(names),
        "matriz": matrix.tolist(),
        "pares": pairs[:k],
        "errores": int(matrix.sum() - np.trace(matrix)),
    }


def by_level(ctx: StudyContext, best: pd.DataFrame, min_support: int) -> dict[str, Any]:
    """Macro F1 of every finalist per level, with group bootstrap intervals: over every class
    and over the classes with ``min_support`` rows at every level (``comun``)."""
    n_classes = len(ctx.pool.class_names(TARGET))
    result = {}
    for row in best.itertuples(index=False):
        table = attach_metadata(_predictions(ctx, row.estudio), ctx.metadata)
        y_true, y_pred, _, groups = head_arrays(table, TARGET, n_classes)
        levels = table["nivel"].astype(str).to_numpy()
        common = _common_classes(y_true, [levels == lv for lv in LEVELS], min_support)
        result[row.peldano] = {
            "estudio": row.estudio,
            "soporte_minimo": min_support,
            "clases_comunes": [ctx.pool.class_names(TARGET)[c] for c in common],
            "comun": {
                level: _macro(
                    ctx,
                    y_true[levels == level],
                    y_pred[levels == level],
                    groups[levels == level],
                    f"{row.estudio}.nivel.{level}.comun",
                    common,
                )
                for level in LEVELS
            },
            **{
                level: _macro(
                    ctx,
                    y_true[levels == level],
                    y_pred[levels == level],
                    groups[levels == level],
                    f"{row.estudio}.nivel.{level}",
                )
                for level in LEVELS
            },
        }
    return result


# --------------------------------------------------------------------------------------------
# 4. Range review
# --------------------------------------------------------------------------------------------


def _position(value: float, spec: dict) -> float:
    low, high = float(spec["min"]), float(spec["max"])
    if spec.get("log"):
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return (value - low) / (high - low)


def edge_values(
    baseline: BaselineConfig, study: dict[str, Any], fraction: float, natural: dict[str, float]
) -> list[dict[str, Any]]:
    """Hyperparameters of a finished study whose best value is at an edge of its range."""
    rung, classifier, target = study["peldano"], study["clasificador"], study["objetivo"]
    revised = int(study.get("revision", 1)) > 1
    space = baseline.space(rung, classifier, target if revised else None)
    flagged = []
    for name, value in study["parametros"].items():
        spec = space.get(name, {})
        side = None
        if spec.get("tipo") in ("float", "int") and float(spec["max"]) > float(spec["min"]):
            position = _position(float(value), spec)
            side = (
                "inferior"
                if position <= fraction
                else "superior"
                if position >= 1 - fraction
                else None
            )
            where = f"{position:.2f}"
        elif spec.get("tipo") == "categorica":
            values = spec["valores"]
            numeric = all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
            if numeric and len(values) > 2 and value in (min(values), max(values)):
                side = "inferior" if value == min(values) else "superior"
                where = f"{values}"
        if side is None:
            continue
        bound = float(spec["min"]) if side == "inferior" and "min" in spec else None
        flagged.append(
            {
                "parametro": name,
                "valor": value,
                "lado": side,
                "posicion": where,
                "limite_natural": name in natural and bound == natural[name],
            }
        )
    return flagged


def range_review(ctx: StudyContext, fraction: float, natural: dict[str, float]) -> dict[str, Any]:
    """Edge values of every study (all revisions), revisions run and finalist changes."""
    every = collect_results(ctx.output, all_revisions=True)
    studies = []
    for name in sorted(every["estudio"]):
        study = json.loads(
            (ctx.output / STUDIES_DIR / name / RESULT_FILE).read_text(encoding="utf-8")
        )
        studies.append(
            {
                "estudio": name,
                "revision": int(study.get("revision", 1)),
                "cv_f1_macro": study["cv_f1_macro"],
                "extremos": edge_values(ctx.baseline, study, fraction, natural),
            }
        )
    pending = []
    for key in ctx.baseline.revisions:
        rung, classifier, target = key.split(".")
        if ctx.baseline.study_name(rung, classifier, target) not in set(every["estudio"]):
            pending.append(key)
    original = every[every["revision"] == 1]
    current = collect_results(ctx.output)
    before = finalists(original) if len(original) else original
    after = finalists(current) if len(current) else current
    return {
        "estudios": studies,
        "revisiones_pendientes": sorted(pending),
        "finalistas_originales": before[["peldano", "estudio", "cv_f1_macro"]].to_dict("records"),
        "finalistas_actuales": after[["peldano", "estudio", "cv_f1_macro"]].to_dict("records"),
        "cambio_finalistas": sorted(set(before["estudio"]) ^ set(after["estudio"])),
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:.{digits}f}"


def _ci(entry: dict) -> str:
    return f"{_pct(entry['valor'])} [{_pct(entry['ic_inf'])}, {_pct(entry['ic_sup'])}]"


def _ci_n(entry: dict) -> str:
    return f"{_ci(entry)} (n {entry['n']})"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
    ]


def _edge_text(edge: dict) -> str:
    value = edge["valor"]
    shown = f"{value:.4g}" if isinstance(value, float) else str(value)
    natural = " — límite natural" if edge["limite_natural"] else ""
    return f"{edge['parametro']} = {shown} ({edge['lado']}, {edge['posicion']}){natural}"


def report_markdown(result: dict[str, Any], header: dict[str, str]) -> str:
    """``diagnosticos.md``."""
    lines = [
        "# Diagnósticos de la línea base (fase 10)",
        "",
        f"Dataset {header['version_dataset']} ({header['huella_dataset'][:12]}), commit "
        f"`{header['git_commit'][:7]}`, equipo `{header['equipo']}`. Solo validación cruzada "
        "de 5 bloques sobre train + val de `aleatorio`; objetivo algoritmo. Intervalos del 95 % "
        "por bootstrap de grupos. **Sin test.**",
        "",
    ]
    shuffle = result.get("barajado")
    if shuffle:
        lines += [
            "## 1. Etiquetas barajadas",
            "",
            f"`{shuffle['estudio']}` con su mejor configuración "
            f"(`{json.dumps(shuffle['parametros'], ensure_ascii=False)}`), etiquetas de "
            "algoritmo permutadas entre los grupos de la parte de entrenamiento de cada bloque "
            f"(el bloque retenido conserva las verdaderas). Azar: {_pct(shuffle['azar'])} %.",
            "",
            *_table(
                ["Semilla", "F1 macro por bloque (%)", "Media de bloques (%)", "OOF (%)"],
                [
                    [
                        r["semilla"],
                        ", ".join(_pct(v) for v in r["f1_bloques"]),
                        _pct(r["f1_macro_media_bloques"]),
                        _pct(r["f1_macro_oof"]),
                    ]
                    for r in shuffle["ejecuciones"]
                ],
            ),
            "",
        ]
    twin = result.get("gemelos")
    if twin:
        lines += [
            "## 2. Gemelos estructurales",
            "",
            "Un circuito retenido tiene gemelo si algún circuito de la parte de entrenamiento "
            "de su bloque tiene la misma huella WL (multiconjunto de etiquetas WL de 64 bits en "
            "la iteración h, mismas etiquetas iniciales que el peldaño WL).",
            "",
        ]
        for h, entry in twin.items():
            lines += [
                f"### {h}: {_pct(entry['con_gemelo'])} % con gemelo; "
                f"{_pct(entry['gemelo_otra_clase'])} % con algún gemelo de otra clase",
                "",
                *_table(
                    [
                        "Peldaño",
                        "Con gemelo: n",
                        "F1 macro con gemelo",
                        "Sin gemelo: n",
                        "F1 macro sin gemelo",
                        "Con gemelo, clases comunes",
                        "Sin gemelo, clases comunes",
                    ],
                    [
                        [
                            p,
                            f["con_gemelo"]["macro"]["n"],
                            _ci(f["con_gemelo"]["macro"]),
                            f["sin_gemelo"]["macro"]["n"],
                            _ci(f["sin_gemelo"]["macro"]),
                            _ci_n(f["con_gemelo"]["macro_comun"]),
                            _ci_n(f["sin_gemelo"]["macro_comun"]),
                        ]
                        for p, f in entry["finalistas"].items()
                    ],
                ),
                "",
                f"Clases comunes (al menos {entry['soporte_minimo']} circuitos con gemelo y "
                "otros tantos sin gemelo): " + ", ".join(entry["clases_comunes"]) + ". La F1 "
                "macro de todas las clases oscila cuando una clase con muy pocos circuitos entra "
                "o sale del remuestreo; por eso su intervalo puede quedar casi por encima del "
                "valor.",
                "",
            ]
            finals = entry["finalistas"]
            rows = []
            for c, cls in enumerate(entry["por_clase"]):
                row = [
                    cls["clase"],
                    cls["n"],
                    _pct(cls["con_gemelo"]),
                    _pct(cls["gemelo_otra_clase"]),
                ]
                for f in finals.values():
                    for part in ("con_gemelo", "sin_gemelo"):
                        s = f[part]["por_clase"][c]
                        row.append(
                            "—"
                            if s["soporte"] == 0
                            else f"{_pct(s['f1'])} [{_pct(s['ic_inf'], 0)}, {_pct(s['ic_sup'], 0)}]"
                        )
                rows.append(row)
            lines += [
                f"F1 por clase ({h}; «con» / «sin» gemelo, %):",
                "",
                *_table(
                    [
                        "Clase",
                        "n",
                        "% con gemelo",
                        "% gemelo otra clase",
                        *[f"{p} {part}" for p in finals for part in ("con", "sin")],
                    ],
                    rows,
                ),
                "",
                f"Gemelos de otra clase ({h}): circuitos de una clase con algún gemelo WL de "
                "otra clase en la parte de entrenamiento de su bloque.",
                "",
                *_table(
                    ["Clase", "Clase del gemelo", "Circuitos", "% de la clase"],
                    [
                        [
                            c["clase"],
                            c["clase_gemelo"],
                            f"{c['circuitos']} de {c['n_clase']}",
                            _pct(c["porcentaje"], 2),
                        ]
                        for c in entry.get("cruzados", [])
                    ],
                ),
                "",
            ]
    conf = result.get("confusion")
    if conf:
        lines += [
            f"## 3. Confusión de `{conf['estudio']}` ({conf['errores']} errores fuera de bloque)",
            "",
            *_table(
                ["Clase A", "Clase B", "Total", "A → B", "B → A", "Soporte A", "Soporte B"],
                [
                    [
                        p["a"],
                        p["b"],
                        p["total"],
                        p["a_como_b"],
                        p["b_como_a"],
                        p["soporte_a"],
                        p["soporte_b"],
                    ]
                    for p in conf["pares"]
                ],
            ),
            "",
            "Matriz completa (filas: clase real; columnas: predicha, en el mismo orden):",
            "",
            *_table(
                ["", *[str(i) for i in range(len(conf["clases"]))]],
                [
                    [f"{i} {name}", *row]
                    for i, (name, row) in enumerate(
                        zip(conf["clases"], conf["matriz"], strict=True)
                    )
                ],
            ),
            "",
        ]
    levels = result.get("niveles")
    if levels:
        lines += [
            "### F1 macro por nivel de los finalistas (%)",
            "",
            *_table(
                ["Peldaño", "Estudio", *LEVELS],
                [
                    [p, f"`{e['estudio']}`", *[f"{_ci(e[lv])} (n {e[lv]['n']})" for lv in LEVELS]]
                    for p, e in levels.items()
                ],
            ),
            "",
            "Solo clases con al menos "
            f"{next(iter(levels.values()))['soporte_minimo']} circuitos en los cuatro niveles "
            f"({', '.join(next(iter(levels.values()))['clases_comunes'])}):",
            "",
            *_table(
                ["Peldaño", *LEVELS],
                [[p, *[_ci_n(e["comun"][lv]) for lv in LEVELS]] for p, e in levels.items()],
            ),
            "",
        ]
    ranges = result.get("rangos")
    if ranges:
        lines += ["## 4. Rangos de hiperparámetros", ""]
        rows = [
            [
                f"`{s['estudio']}`",
                _pct(s["cv_f1_macro"]),
                "; ".join(_edge_text(e) for e in s["extremos"]) or "—",
            ]
            for s in ranges["estudios"]
        ]
        lines += [*_table(["Estudio", "F1 macro CV (%)", "Valores en el extremo"], rows), ""]
        if ranges["revisiones_pendientes"]:
            lines += [f"Revisiones pendientes: {', '.join(ranges['revisiones_pendientes'])}.", ""]
        lines += [
            "Finalistas originales: "
            + ", ".join(
                f"{f['peldano']} `{f['estudio']}` ({_pct(f['cv_f1_macro'])} %)"
                for f in ranges["finalistas_originales"]
            ),
            "",
            "Finalistas actuales: "
            + ", ".join(
                f"{f['peldano']} `{f['estudio']}` ({_pct(f['cv_f1_macro'])} %)"
                for f in ranges["finalistas_actuales"]
            ),
            "",
            "**Cambia algún finalista: "
            + (
                "sí** (" + ", ".join(ranges["cambio_finalistas"]) + ")"
                if ranges["cambio_finalistas"]
                else "no**"
            ),
            "",
        ]
    return "\n".join(lines)


def run_diagnostics(
    ctx: StudyContext, config: DiagnosticsConfig, *, parts: tuple[str, ...]
) -> tuple[dict[str, Any], Path]:
    """Run the selected parts (``barajado``, ``gemelos``, ``confusion``, ``rangos``), write
    ``diagnosticos.json`` / ``.md`` (merged with earlier parts) and return them.

    Raises:
        ShuffledLabelAlarm: if the shuffled-label F1 is above the threshold (after writing).
    """
    json_path = ctx.output / f"{REPORT_NAME}.json"
    result = json.loads(json_path.read_text("utf-8")) if json_path.is_file() else {}
    best = finalists(collect_results(ctx.output))
    wl_study = best[best["peldano"] == "wl"].iloc[0]["estudio"]
    if "barajado" in parts:
        study = json.loads((ctx.output / STUDIES_DIR / wl_study / RESULT_FILE).read_text("utf-8"))
        result["barajado"] = shuffled_labels(ctx, study, config.shuffle_seeds)
    if "gemelos" in parts:
        result["gemelos"] = twins(ctx, best, config.twin_h, config.min_support)
    if "confusion" in parts:
        result["confusion"] = confused_pairs(ctx, wl_study, config.confused_pairs)
        result["niveles"] = by_level(ctx, best, config.min_support)
    if "rangos" in parts:
        result["rangos"] = range_review(ctx, config.range_fraction, config.natural_bounds)
    result["procedencia"] = {
        k: ctx.provenance.tags[k]
        for k in ("version_dataset", "huella_dataset", "git_commit", "equipo")
    }
    write_text_atomic(json.dumps(result, indent=2, ensure_ascii=False) + "\n", json_path)
    path = ctx.output / f"{REPORT_NAME}.md"
    write_text_atomic(report_markdown(result, result["procedencia"]), path)
    shuffle = result.get("barajado")
    if "barajado" in parts and shuffle["f1_macro_max"] > config.shuffle_threshold:
        raise ShuffledLabelAlarm(
            f"shuffled-label macro F1 {shuffle['f1_macro_max']:.3f} > "
            f"{config.shuffle_threshold:.2f}: stop and review"
        )
    return result, path


def log_diagnostics(ctx: StudyContext, config: DiagnosticsConfig, result: dict, path: Path) -> None:
    """MLflow run ``<nombre>.diagnosticos`` with the headline numbers and both files."""
    from qcclass.seguimiento.mlflow_cliente import tracked_run

    with tracked_run(
        experiment=ctx.baseline.mlflow_experiment,
        run_name=f"{ctx.baseline.name}.diagnosticos",
        configuration={"diagnosticos": config.raw},
        provenance={**ctx.provenance.tags, "estudio": "diagnosticos"},
        machine_description=ctx.provenance.machine_description,
    ) as run:
        if shuffle := result.get("barajado"):
            run.log_metric("barajado_f1_macro_media", shuffle["f1_macro_media"])
            run.log_metric("barajado_f1_macro_max", shuffle["f1_macro_max"])
        for h, entry in (result.get("gemelos") or {}).items():
            run.log_metric(f"gemelos_{h}_con_gemelo", entry["con_gemelo"])
            run.log_metric(f"gemelos_{h}_otra_clase", entry["gemelo_otra_clase"])
            for rung, f in entry["finalistas"].items():
                for part in ("con_gemelo", "sin_gemelo"):
                    value = f[part]["macro"]["valor"]
                    if np.isfinite(value):
                        run.log_metric(f"gemelos_{h}_{rung}_{part}_f1_macro", value)
        for rung, levels in (result.get("niveles") or {}).items():
            for level in LEVELS:
                if np.isfinite(levels[level]["valor"]):
                    run.log_metric(f"nivel_{rung}_{level}_f1_macro", levels[level]["valor"])
        run.log_artifact(str(path))
        run.log_artifact(str(path.with_suffix(".json")))
