"""Run the whole baseline (``qcclass linea-base``): every study, then the summary.

Order: targets as configured (algorithm first), rungs from cheapest to most expensive (E7,
F3, WL), classifiers as configured. Finished studies are skipped, so an interrupted run is
resumed by launching the same command again. Results go to ``<resultados>/baseline/``:
``estudios/<study>/`` (predictions, report, trials, result), ``RESUMEN.md``, ``resumen.csv``
and ``importancia_<rung>.csv``; the WL cache to ``cache/``.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig, EvaluationConfig, SplitsConfig
from qcclass.entrenamiento.bloques import fold_training, load_pool
from qcclass.lineabase.configuracion import BaselineConfig
from qcclass.lineabase.estudio import FeatureStore, StudyContext, run_study
from qcclass.lineabase.resumen import (
    SUMMARY_FILE,
    SUMMARY_TABLE,
    collect_results,
    finalists,
    permutation_importance,
    write_summary,
)
from qcclass.metadatos import METADATA_COLUMNS, read_table
from qcclass.seguimiento.mlflow_cliente import tracked_run
from qcclass.seguimiento.procedencia import start_provenance

logger = logging.getLogger(__name__)

OUTPUT_DIR = "baseline"
CACHE_DIR = "cache"


@dataclass(frozen=True)
class BaselineRun:
    """Outcome of ``qcclass linea-base``."""

    results: list[dict]
    table: pd.DataFrame
    finalists: pd.DataFrame
    summary: Path | None


def output_dir(config: DatasetConfig) -> Path:
    """``<resultados>/baseline``."""
    return config.rutas.resultados / OUTPUT_DIR


def _selected(values: Sequence[str], chosen: Sequence[str] | None) -> list[str]:
    if chosen is None:
        return list(values)
    if unknown := [c for c in chosen if c not in values]:
        raise ValueError(f"unknown {unknown} (valid: {list(values)})")
    return [v for v in values if v in chosen]


def _importances(ctx: StudyContext, best: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    """Permutation importance of every finalist (reused while the finalist study is the same)."""
    importances = {}
    for row in best.itertuples(index=False):
        marker = output / f"importancia_{row.peldano}.json"
        table_path = output / f"importancia_{row.peldano}.csv"
        if marker.is_file() and table_path.is_file():
            if json.loads(marker.read_text(encoding="utf-8")).get("estudio") == row.estudio:
                importances[row.peldano] = pd.read_csv(table_path)
                continue
        study = json.loads(
            (output / "estudios" / row.estudio / "resultado.json").read_text("utf-8")
        )
        logger.info("permutation importance of %s", row.estudio)
        importances[row.peldano] = permutation_importance(
            ctx,
            study,
            max_wl_columns=ctx.baseline.importance_wl_columns,
            repeats=ctx.baseline.importance_repeats,
        )
        write_text_atomic(json.dumps({"estudio": row.estudio}), marker)
    return importances


def build_context(
    config: DatasetConfig,
    splits: SplitsConfig,
    baseline: BaselineConfig,
    evaluation: EvaluationConfig,
    *,
    storage: optuna.storages.BaseStorage | None,
    experiment: str | None = None,
) -> StudyContext:
    """Pool, blocks, features and provenance shared by the studies (and the diagnostics)."""
    provenance = start_provenance(config, experiment or baseline.name, device="cpu")
    pool = load_pool(config, splits, experiment or baseline.name)
    output = output_dir(config)
    return StudyContext(
        config=config,
        splits=splits,
        baseline=baseline,
        evaluation=evaluation,
        pool=pool,
        folds=[fold_training(pool, block, splits) for block in range(pool.n_blocks)],
        features=FeatureStore(config, pool, baseline, output / CACHE_DIR),
        storage=storage,
        provenance=provenance,
        metadata=read_table(config.rutas.metadata, METADATA_COLUMNS),
        output=output,
    )


def run_baseline(
    config: DatasetConfig,
    splits: SplitsConfig,
    baseline: BaselineConfig,
    evaluation: EvaluationConfig,
    *,
    storage: optuna.storages.BaseStorage,
    rungs: Sequence[str] | None = None,
    classifiers: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    redo: bool = False,
    summary: bool = True,
) -> BaselineRun:
    """Run the selected studies and write the summary (see the module docstring)."""
    ctx = build_context(config, splits, baseline, evaluation, storage=storage)
    provenance, output = ctx.provenance, ctx.output
    results = []
    for target in _selected(baseline.targets, targets):
        for rung in _selected(list(baseline.rungs), rungs):
            for classifier in _selected(list(baseline.spaces), classifiers):
                results.append(run_study(ctx, rung, classifier, target, redo=redo))
                logger.info(
                    "%s: CV macro F1 %.4f", results[-1]["estudio"], results[-1]["cv_f1_macro"]
                )
    table = collect_results(output)
    best = finalists(table) if len(table) else table
    path = None
    if summary and len(best):
        importances = _importances(ctx, best, output)
        path = write_summary(output, table, best, importances, provenance.tags)
        with tracked_run(
            experiment=baseline.mlflow_experiment,
            run_name=f"{baseline.name}.resumen",
            configuration={"linea_base": baseline.raw, "finalistas": list(best["estudio"])},
            provenance={**provenance.tags, "estudio": "resumen"},
            machine_description=provenance.machine_description,
        ) as run:
            run.log_artifact(str(path))
            run.log_artifact(str(output / SUMMARY_TABLE))
            for rung in importances:
                run.log_artifact(str(output / f"importancia_{rung}.csv"))
    return BaselineRun(results, table, best, path)


__all__ = ["SUMMARY_FILE", "BaselineRun", "build_context", "output_dir", "run_baseline"]
