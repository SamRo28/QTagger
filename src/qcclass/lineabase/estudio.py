"""One baseline study: Optuna search with 5-block cross-validation, out-of-fold predictions of
the best trial, evaluation and MLflow run.

The objective of a trial is the mean macro F1 of the target over the held-out blocks (only
rows whose target is a real class). After each block the running mean is reported, so the
median pruner can stop poor trials once ``intentos_iniciales`` trials are complete. Studies
live in the shared Optuna storage and can be resumed: an interrupted run leaves its trial
``RUNNING``, which is marked as failed the next time (see
:func:`qcclass.seguimiento.optuna_cliente.fail_stale_trials`), and only the missing complete
or pruned trials are run.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig, EvaluationConfig, SplitsConfig
from qcclass.entrenamiento.bloques import (
    CrossValidationPool,
    FoldTraining,
    head_index,
    sample_weights,
)
from qcclass.evaluacion.informe import evaluate, write_report
from qcclass.evaluacion.metricas import confusion, scores_from_confusion
from qcclass.evaluacion.predicciones import aggregate_out_of_fold, attach_metadata, make_table
from qcclass.lineabase.caracteristicas import WLFeatures, f3_table, wl_features
from qcclass.lineabase.clasificadores import build, fit, full_probabilities, prepare
from qcclass.lineabase.configuracion import BaselineConfig, fixed_params, suggest
from qcclass.seguimiento.mlflow_cliente import tracked_run
from qcclass.seguimiento.optuna_cliente import MACHINE_ATTR, fail_stale_trials
from qcclass.seguimiento.procedencia import Provenance
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
RESULT_FILE = "resultado.json"
STUDIES_DIR = "estudios"


class FeatureStore:
    """Feature matrices of the pool rows for every rung (computed once, WL cached)."""

    def __init__(
        self,
        config: DatasetConfig,
        pool: CrossValidationPool,
        baseline: BaselineConfig,
        cache: Path,
    ) -> None:
        self.config, self.pool, self.baseline, self.cache = config, pool, baseline, cache
        self._f3: pd.DataFrame | None = None
        self._wl: WLFeatures | None = None

    def wl(self) -> WLFeatures:
        """WL features of the pool (iterations 0..h_max)."""
        if self._wl is None:
            settings = next(r.wl for r in self.baseline.rungs.values() if r.source == "wl")
            self._wl = wl_features(self.config, list(self.pool.rows["id"]), settings, self.cache)
        return self._wl

    def matrix(self, rung: str, params: dict[str, Any]) -> np.ndarray | sp.csr_matrix:
        """Features of ``rung`` (WL with ``h`` iterations from ``params``)."""
        spec = self.baseline.rungs[rung]
        if spec.source == "wl":
            return self.wl().matrix(int(params["h"]))
        if self._f3 is None:
            self._f3 = f3_table(self.config, list(self.pool.rows["id"]), None)
        columns = list(spec.columns) if spec.columns else list(self._f3.columns)
        return self._f3[columns].to_numpy(dtype=float)

    def names(self, rung: str) -> list[str] | None:
        """Column names of an F3 rung (``None`` for WL)."""
        spec = self.baseline.rungs[rung]
        if spec.source == "wl":
            return None
        if self._f3 is None:
            self._f3 = f3_table(self.config, list(self.pool.rows["id"]), None)
        return list(spec.columns) if spec.columns else list(self._f3.columns)


@dataclass
class StudyContext:
    """Everything a study needs."""

    config: DatasetConfig
    splits: SplitsConfig
    baseline: BaselineConfig
    evaluation: EvaluationConfig
    pool: CrossValidationPool
    folds: list[FoldTraining]
    features: FeatureStore
    storage: optuna.storages.BaseStorage
    provenance: Provenance
    metadata: pd.DataFrame
    output: Path
    log_to_mlflow: bool = True
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockFit:
    """A model fitted on the training part of one block, with its held-out rows."""

    model: Any
    held_out_rows: np.ndarray  # positions in the pool
    held_out_input: Any
    columns: np.ndarray | None


def fit_block(
    ctx: StudyContext,
    features: np.ndarray | sp.csr_matrix,
    fold: FoldTraining,
    target: str,
    classifier: str,
    params: dict[str, Any],
    seed: int,
) -> BlockFit:
    """Fit on the (capped, weighted) training part of a block for ``target``."""
    train = head_index(ctx.pool, fold.train_index, target)
    held_out = head_index(ctx.pool, fold.held_out_index, target)
    prepared = prepare(features, train, held_out, classifier, params)
    y = ctx.pool.rows[f"y_{target}"].to_numpy()
    model = build(classifier, params, seed, sparse=sp.issparse(prepared.train))
    fit(model, prepared.train, y[train], sample_weights(ctx.pool, train, target, fold))
    return BlockFit(model, held_out, prepared.held_out, prepared.columns)


def _objective(ctx: StudyContext, rung: str, classifier: str, target: str, name: str) -> Any:
    space = ctx.baseline.space(rung, classifier)
    n_classes = len(ctx.pool.class_names(target))
    y = ctx.pool.rows[f"y_{target}"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr(MACHINE_ATTR, ctx.provenance.machine)
        start = time.perf_counter()
        params = suggest(trial, space)
        features = ctx.features.matrix(rung, params)
        scores = []
        for fold in ctx.folds:
            seed = derive_seed(ctx.baseline.seed, name, "bloque", fold.block)
            fitted = fit_block(ctx, features, fold, target, classifier, params, seed)
            predicted = fitted.model.predict(fitted.held_out_input).astype(int)
            matrix = confusion(y[fitted.held_out_rows], predicted, n_classes)
            scores.append(scores_from_confusion(matrix)["f1_macro"])
            trial.report(float(np.mean(scores)), step=fold.block)
            if trial.should_prune():
                trial.set_user_attr("f1_bloques", scores)
                raise optuna.TrialPruned()
        trial.set_user_attr("f1_bloques", scores)
        trial.set_user_attr("segundos", round(time.perf_counter() - start, 2))
        return float(np.mean(scores))

    return objective


def out_of_fold(
    ctx: StudyContext, rung: str, classifier: str, target: str, name: str, params: dict[str, Any]
) -> pd.DataFrame:
    """Held-out predictions of every block with ``params`` (one table, column ``bloque``)."""
    features = ctx.features.matrix(rung, params)
    n_classes = len(ctx.pool.class_names(target))
    rows = ctx.pool.rows
    tables = []
    for fold in ctx.folds:
        seed = derive_seed(ctx.baseline.seed, name, "bloque", fold.block)
        fitted = fit_block(ctx, features, fold, target, classifier, params, seed)
        probabilities = full_probabilities(fitted.model, fitted.held_out_input, n_classes)
        held = rows.iloc[fitted.held_out_rows]
        tables.append(
            make_table(
                held["id"],
                held["group_id"],
                {target: held[f"y_{target}"]},
                {target: probabilities},
                evaluation_only=held["solo_evaluacion"],
                block=fold.block,
            )
        )
    return aggregate_out_of_fold(tables)


def _run_search(
    ctx: StudyContext, rung: str, classifier: str, target: str, name: str
) -> optuna.Study:
    baseline = ctx.baseline
    study = optuna.create_study(
        study_name=name,
        storage=ctx.storage,
        sampler=optuna.samplers.TPESampler(seed=derive_seed(baseline.seed, name, "tpe")),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=baseline.pruning_startup),
        direction="maximize",
        load_if_exists=True,
    )
    fail_stale_trials(study, ctx.storage, ctx.provenance.machine)
    finished = len(study.get_trials(deepcopy=False, states=FINISHED))
    remaining = baseline.trials - finished
    if remaining > 0:
        logger.info("%s: %d trials done, running %d more", name, finished, remaining)
        study.optimize(
            _objective(ctx, rung, classifier, target, name),
            n_trials=remaining,
            callbacks=[optuna.study.MaxTrialsCallback(baseline.trials, states=FINISHED)],
            gc_after_trial=True,
        )
    return study


def _trial_seconds(study: optuna.Study) -> float:
    return float(
        sum(
            (t.datetime_complete - t.datetime_start).total_seconds()
            for t in study.get_trials(deepcopy=False, states=FINISHED)
            if t.datetime_complete and t.datetime_start
        )
    )


def run_study(
    ctx: StudyContext, rung: str, classifier: str, target: str, *, redo: bool = False
) -> dict[str, Any]:
    """Search, out-of-fold predictions of the best trial, evaluation, files and MLflow run.

    A study whose ``resultado.json`` exists is skipped unless ``redo``.
    """
    name = ctx.baseline.study_name(rung, classifier, target)
    folder = ctx.output / STUDIES_DIR / name
    result_file = folder / RESULT_FILE
    if result_file.is_file() and not redo:
        logger.info("%s: already finished", name)
        return json.loads(result_file.read_text(encoding="utf-8"))
    started = time.perf_counter()
    study = _run_search(ctx, rung, classifier, target, name)
    best = study.best_trial
    space = ctx.baseline.space(rung, classifier)
    params = fixed_params(space, best.params)
    predictions = out_of_fold(ctx, rung, classifier, target, name, params)
    enriched = attach_metadata(predictions, ctx.metadata)
    report = evaluate(enriched, {target: ctx.pool.class_names(target)}, ctx.evaluation, name=name)
    folder.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(folder / "predicciones.parquet", index=False)
    write_report(report, folder, "informe")
    trials = study.trials_dataframe()
    trials.to_csv(folder / "intentos.csv", index=False)
    states = trials["state"].value_counts().to_dict() if len(trials) else {}
    block_scores = best.user_attrs.get("f1_bloques", [])
    result = {
        "estudio": name,
        "peldano": rung,
        "clasificador": classifier,
        "objetivo": target,
        "cv_f1_macro": float(best.value),
        "cv_f1_macro_desviacion": float(np.std(block_scores)) if block_scores else None,
        "f1_bloques": block_scores,
        "mejor_intento": best.number,
        "parametros": params,
        "intentos": {str(k): int(v) for k, v in states.items()},
        "segundos_busqueda": round(_trial_seconds(study), 1),
        "segundos_esta_ejecucion": round(time.perf_counter() - started, 1),
        "oof": report[target]["metricas"],
        "n_oof": report[target]["n"],
        **{
            k: v
            for k, v in ctx.provenance.tags.items()
            if k in ("version_dataset", "huella_dataset", "git_commit", "equipo")
        },
    }
    write_text_atomic(json.dumps(result, indent=2, ensure_ascii=False) + "\n", result_file)
    if ctx.log_to_mlflow:
        _log_study(ctx, result, space, folder)
    return result


def _log_study(ctx: StudyContext, result: dict, space: dict, folder: Path) -> None:
    import mlflow

    configuration = {
        "estudio": result["estudio"],
        "peldano": result["peldano"],
        "clasificador": result["clasificador"],
        "objetivo": result["objetivo"],
        "espacio": space,
        "mejores_parametros": result["parametros"],
        "linea_base": ctx.baseline.raw,
    }
    with tracked_run(
        experiment=ctx.baseline.mlflow_experiment,
        run_name=result["estudio"],
        configuration=configuration,
        provenance={**ctx.provenance.tags, "estudio": result["estudio"]},
        machine_description=ctx.provenance.machine_description,
    ):
        mlflow.log_metric("cv_f1_macro", result["cv_f1_macro"])
        if result["cv_f1_macro_desviacion"] is not None:
            mlflow.log_metric("cv_f1_macro_desviacion", result["cv_f1_macro_desviacion"])
        for metric, values in result["oof"].items():
            mlflow.log_metric(f"oof_{metric}", values["valor"])
            mlflow.log_metric(f"oof_{metric}_ic_inf", values["ic_inf"])
            mlflow.log_metric(f"oof_{metric}_ic_sup", values["ic_sup"])
        mlflow.log_metric("segundos_busqueda", result["segundos_busqueda"])
        mlflow.log_artifacts(str(folder))
