"""Hyperparameter search of the GNN (phase 12, ``qcclass buscar-gnn``).

One Optuna study (``nombre``) in the shared storage, TPE with a seed, median pruning after
``intentos_iniciales`` complete trials. A trial trains its configuration on every block of
the cross-validation (``entrenamiento.gnn.train_block``, 1 seed per block) and returns the
mean algorithm macro F1 of the held-out blocks; the family F1 is kept as a user attribute.

Intermediate values (for the pruner):

* block 0, every ``informe_cada_epocas`` epochs (and at the last one): the held-out F1 at
  step ``round(100 * epoch / epocas)``, the *percentage* of the training done, so that
  trials with 50, 100 or 150 epochs are compared at the same point of their schedules;
* after every block ``k``: the running mean of the block F1 at step ``1000 + k``.

Robustness: every trial has the directory ``<resultados>/gnn/<nombre>/<fingerprint>/``
(fingerprint of its hyperparameters and seed), with one ``bloque_<k>`` per block holding its
checkpoints, so a trial interrupted and requeued (``fail_stale_trials(requeue=True)`` when the
search is launched again) resumes from its last epoch. Optuna reports and MLflow logging go
through local queues (``BufferedReports``, ``ResilientRun``): a server cut never stops a
training, and the trial only fails if a server is still unreachable when it ends. A trial
whose hyperparameters equal those of a complete trial is not trained: it returns the known
value (discrete spaces repeat configurations).

Each trial keeps the last checkpoints of its blocks and ``predicciones_oof.parquet`` (the
held-out predictions of every block). After every trial ``estimacion.json`` is rewritten
with the time per trial and the projected remaining time; after ``tras_intentos`` trials, a
projection above ``limite_horas`` is logged as a warning.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch

from qcclass.circuitos import write_text_atomic
from qcclass.config import ConfigError, DatasetConfig, SplitsConfig, load_experiment_config
from qcclass.entrenamiento.bloques import fold_training, load_pool
from qcclass.entrenamiento.gnn import (
    PREDICTIONS_FILE,
    TrainingSettings,
    hyperparameter_fingerprint,
    resolve_training_device,
    run_directory,
    train_block,
)
from qcclass.entrenamiento.gnn_experimento import OUTPUT_DIR, loading_workers
from qcclass.evaluacion.predicciones import aggregate_out_of_fold
from qcclass.lineabase.configuracion import check_space, suggest
from qcclass.modelos.gnn import GNNConfig
from qcclass.seguimiento.mlflow_cliente import flatten, tracking_uri, wait_for_mlflow
from qcclass.seguimiento.optuna_cliente import MACHINE_ATTR, BufferedReports, fail_stale_trials
from qcclass.seguimiento.procedencia import start_provenance
from qcclass.seguimiento.registro import ResilientRun
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
BLOCK_STEP_OFFSET = 1000
OOF_FILE = "predicciones_oof.parquet"
ESTIMATE_FILE = "estimacion.json"
FINGERPRINT_ATTR = "huella"
REQUIRED = (
    "nombre",
    "experimento_mlflow",
    "semilla",
    "intentos",
    "poda",
    "estimacion",
    "espacio",
    "fijos",
)
MODEL_KEYS = {f.name for f in fields(GNNConfig)} - {
    "node_dim",
    "edge_dim",
    "num_classes",
    "num_families",
}
TRAINING_KEYS = {f.name for f in fields(TrainingSettings)}


@dataclass(frozen=True)
class SearchConfig:
    """Parsed ``gnn_busqueda.yaml``."""

    name: str
    mlflow_experiment: str
    seed: int
    trials: int
    blocks: tuple[int, ...] | None
    pruning_startup: int
    report_every: int
    estimate_after: int
    limit_hours: float
    space: dict[str, dict]
    fixed_model: dict[str, Any]
    fixed_training: dict[str, Any]
    raw: dict[str, Any]


def load_search_config(path: Path | str) -> SearchConfig:
    """Load and validate a GNN search configuration.

    Raises:
        ConfigError: if a key is missing or a hyperparameter is unknown or misplaced.
    """
    data = load_experiment_config(path, REQUIRED)
    name = Path(path).name
    space = check_space(data["espacio"], "espacio")
    fixed = data["fijos"] or {}
    fixed_model, fixed_training = (
        dict(fixed.get("modelo") or {}),
        dict(fixed.get("entrenamiento") or {}),
    )
    for key in space:
        if key not in MODEL_KEYS | TRAINING_KEYS:
            raise ConfigError(f"{name}: espacio[{key}] is not a model or training setting")
    if bad := sorted(set(fixed_model) - MODEL_KEYS):
        raise ConfigError(f"{name}: fijos.modelo has unknown keys {bad}")
    if bad := sorted(set(fixed_training) - TRAINING_KEYS):
        raise ConfigError(f"{name}: fijos.entrenamiento has unknown keys {bad}")
    if overlap := sorted(set(space) & (set(fixed_model) | set(fixed_training))):
        raise ConfigError(f"{name}: {overlap} both searched and fixed")
    blocks = data.get("bloques")
    config = SearchConfig(
        name=str(data["nombre"]),
        mlflow_experiment=str(data["experimento_mlflow"]),
        seed=int(data["semilla"]),
        trials=int(data["intentos"]),
        blocks=tuple(int(b) for b in blocks) if blocks else None,
        pruning_startup=int(data["poda"]["intentos_iniciales"]),
        report_every=int(data["poda"]["informe_cada_epocas"]),
        estimate_after=int(data["estimacion"]["tras_intentos"]),
        limit_hours=float(data["estimacion"]["limite_horas"]),
        space=space,
        fixed_model=fixed_model,
        fixed_training=fixed_training,
        raw=data,
    )
    try:  # every point of the space must build valid settings
        split_params(config, {k: _example(s) for k, s in space.items()})
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}: {exc}") from exc
    return config


def _example(spec: dict) -> Any:
    kind = spec["tipo"]
    if kind == "fijo":
        return spec["valor"]
    if kind == "categorica":
        return spec["valores"][0]
    return int(spec["min"]) if kind == "int" else float(spec["min"])


def split_params(
    search: SearchConfig, params: dict[str, Any]
) -> tuple[dict[str, Any], TrainingSettings]:
    """Model settings and training settings of a point of the space (plus the fixed ones)."""
    model = {**search.fixed_model, **{k: v for k, v in params.items() if k in MODEL_KEYS}}
    training = {**search.fixed_training, **{k: v for k, v in params.items() if k in TRAINING_KEYS}}
    GNNConfig.from_mapping(model, node_dim=1, edge_dim=1, num_classes=2, num_families=2)
    return model, TrainingSettings.from_mapping(training)


def progress_step(epoch: int, epochs: int) -> int:
    """Step of a block-0 intermediate value: percentage of the training done."""
    return round(100 * epoch / epochs)


def block_step(block: int) -> int:
    """Step of the running mean after ``block``."""
    return BLOCK_STEP_OFFSET + block


@dataclass
class SearchContext:
    """Everything the objective needs."""

    config: DatasetConfig
    search: SearchConfig
    pool: Any
    folds: list[Any]
    storage: Any
    provenance: Any
    device: str
    workers: int
    output: Path
    max_batches: int | None = None
    log_to_mlflow: bool = True


def _trial_run(ctx: SearchContext, trial: optuna.Trial, directory: Path, params: dict) -> Any:
    if not ctx.log_to_mlflow:
        return None
    run = ResilientRun(
        experiment=ctx.search.mlflow_experiment,
        run_name=f"{ctx.search.name}.intento{trial.number}",
        tags={**ctx.provenance.tags, "estudio": ctx.search.name, FINGERPRINT_ATTR: directory.name},
        spool=directory / "mlflow",
    )
    run.set_tags({"intento": trial.number})
    if run.run_id is None:
        run.log_params(
            flatten({**params, "dispositivo": ctx.device, "procesos_carga": ctx.workers})
        )
        run.log_dict(ctx.search.raw, "configuracion.json")
        run.log_dict(ctx.provenance.machine_description, "equipo.json")
    run.flush(block=False)
    return run


def _finish(run: Any, status: str) -> None:
    if run is not None:
        run.finish(status)


def make_objective(ctx: SearchContext, study: optuna.Study) -> Callable[[optuna.Trial], float]:
    """Objective of the search (see the module docstring)."""
    search = ctx.search
    blocks = [fold.block for fold in ctx.folds]

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr(MACHINE_ATTR, ctx.provenance.machine)
        point = suggest(trial, search.space)
        model, training = split_params(search, point)
        identity = {"modelo": model, "entrenamiento": training.as_dict(), "semilla": search.seed}
        fingerprint = hyperparameter_fingerprint(identity)
        trial.set_user_attr(FINGERPRINT_ATTR, fingerprint)
        for done in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if done.user_attrs.get(FINGERPRINT_ATTR) == fingerprint:
                logger.info("trial %d repeats trial %d: not trained", trial.number, done.number)
                trial.set_user_attr("repite_intento", done.number)
                for key in ("f1_bloques", "f1_familia_bloques", "f1_familia_media"):
                    if key in done.user_attrs:
                        trial.set_user_attr(key, done.user_attrs[key])
                return float(done.value)
        directory = run_directory(ctx.output, search.name, identity, 0).parent
        run = _trial_run(ctx, trial, directory, identity)
        reports = BufferedReports(trial, ctx.storage)
        scores, family, seconds = [], [], []
        try:
            for fold in ctx.folds:

                def on_epoch(
                    epoch: int, metrics: dict[str, float], block: int = fold.block
                ) -> None:
                    if block != blocks[0] or "f1_macro_algoritmo" not in metrics:
                        return
                    if epoch % search.report_every and epoch != training.epocas:
                        return
                    reports.report(
                        metrics["f1_macro_algoritmo"], progress_step(epoch, training.epocas)
                    )
                    if reports.should_prune():
                        raise optuna.TrialPruned(f"block {block}, epoch {epoch}")

                result = train_block(
                    pool=ctx.pool,
                    fold=fold,
                    graphs_dir=ctx.config.rutas.graphs,
                    model_settings=model,
                    training=training,
                    seed=search.seed,
                    device=ctx.device,
                    directory=directory / f"bloque_{fold.block}",
                    workers=ctx.workers,
                    run=run,
                    on_epoch=on_epoch,
                    max_batches=ctx.max_batches,
                    metric_prefix=f"bloque{fold.block}.",
                )
                scores.append(result.f1_macro_algoritmo)
                family.append(result.f1_macro_familia)
                seconds.append(result.seconds)
                trial.set_user_attr("f1_bloques", scores)
                reports.report(float(np.mean(scores)), block_step(fold.block))
                if run is not None:
                    run.log_metrics(
                        {"cv_f1_macro_media_parcial": float(np.mean(scores))}, step=fold.block
                    )
                    run.flush(block=False)
                if reports.should_prune():
                    raise optuna.TrialPruned(f"after block {fold.block}")
        except optuna.TrialPruned:
            reports.flush()
            if run is not None:
                run.set_tags({"estado": "podado"})
            _finish(run, "FINISHED")
            raise
        except BaseException:
            try:
                _finish(run, "FAILED")
            except Exception:  # noqa: BLE001 - the original error matters more
                logger.exception("could not close the MLflow run of trial %d", trial.number)
            raise
        tables = [pd.read_parquet(directory / f"bloque_{b}" / PREDICTIONS_FILE) for b in blocks]
        aggregate_out_of_fold(tables).to_parquet(directory / OOF_FILE, index=False)
        value = float(np.mean(scores))
        trial.set_user_attr("f1_familia_bloques", family)
        trial.set_user_attr("f1_familia_media", float(np.mean(family)))
        trial.set_user_attr("segundos_bloques", seconds)
        reports.flush()
        if run is not None:
            run.log_metrics(
                {
                    "cv_f1_macro_algoritmo": value,
                    "cv_f1_macro_algoritmo_desviacion": float(np.std(scores)),
                    "cv_f1_macro_familia": float(np.mean(family)),
                    "segundos_intento": float(sum(seconds)),
                }
            )
            run.set_tags({"estado": "completo"})
            run.log_artifact(directory / OOF_FILE)
        _finish(run, "FINISHED")
        return value

    return objective


def estimate(study: optuna.Study, search: SearchConfig) -> dict[str, Any]:
    """Real time per finished trial and projection of the remaining time."""
    finished = [
        t
        for t in study.get_trials(deepcopy=False, states=FINISHED)
        if t.datetime_start and t.datetime_complete and "repite_intento" not in t.user_attrs
    ]
    durations = [(t.datetime_complete - t.datetime_start).total_seconds() for t in finished]
    by_state = {
        state.name.lower(): [
            (t.datetime_complete - t.datetime_start).total_seconds() / 3600
            for t in finished
            if t.state == state
        ]
        for state in FINISHED
    }
    done = len(study.get_trials(deepcopy=False, states=FINISHED))
    mean = float(np.mean(durations)) / 3600 if durations else None
    remaining = max(search.trials - done, 0)
    projected = None if mean is None else mean * remaining
    spent = float(sum(durations)) / 3600
    total = None if projected is None else spent + projected
    return {
        "intentos_terminados": done,
        "intentos_restantes": remaining,
        "horas_consumidas": round(spent, 2),
        "horas_por_intento_media": None if mean is None else round(mean, 3),
        "horas_por_estado": {k: [round(v, 3) for v in vs] for k, vs in by_state.items()},
        "horas_restantes_estimadas": None if projected is None else round(projected, 1),
        "horas_totales_estimadas": None if total is None else round(total, 1),
        "limite_horas": search.limit_hours,
        "supera_limite": bool(
            done >= search.estimate_after and total is not None and total > search.limit_hours
        ),
    }


def run_search(
    config: DatasetConfig,
    splits: SplitsConfig,
    search: SearchConfig,
    *,
    storage: Any,
    device: str = "auto",
    workers: int | None = None,
    max_batches: int | None = None,
    log_to_mlflow: bool = True,
) -> optuna.Study:
    """Run (or resume) the search until it has ``intentos`` finished trials."""
    device = resolve_training_device(device)
    provenance = start_provenance(config, search.name, device=device)
    if log_to_mlflow:
        wait_for_mlflow(tracking_uri())
    pool = load_pool(config, splits, search.name)
    blocks = search.blocks if search.blocks is not None else tuple(range(pool.n_blocks))
    if bad := [b for b in blocks if not 0 <= b < pool.n_blocks]:
        raise ValueError(f"unknown blocks {bad}")
    output = config.rutas.resultados / OUTPUT_DIR
    ctx = SearchContext(
        config=config,
        search=search,
        pool=pool,
        folds=[fold_training(pool, b, splits) for b in blocks],
        storage=storage,
        provenance=provenance,
        device=device,
        workers=loading_workers() if workers is None else workers,
        output=output,
        max_batches=max_batches,
        log_to_mlflow=log_to_mlflow,
    )
    study = optuna.create_study(
        study_name=search.name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=derive_seed(search.seed, search.name, "tpe")),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=search.pruning_startup),
        direction="maximize",
        load_if_exists=True,
    )
    fail_stale_trials(study, storage, provenance.machine, requeue=True)
    estimate_path = output / search.name / ESTIMATE_FILE
    warned = False

    def after_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal warned
        if device == "cuda":
            torch.cuda.empty_cache()
        figures = estimate(study, search)
        estimate_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(json.dumps(figures, indent=2) + "\n", estimate_path)
        logger.info(
            "trial %d %s; %d/%d finished, %.1f h spent, about %s h left",
            trial.number,
            trial.state.name,
            figures["intentos_terminados"],
            search.trials,
            figures["horas_consumidas"],
            figures["horas_restantes_estimadas"],
        )
        if figures["supera_limite"] and not warned:
            warned = True
            logger.warning(
                "AVISO: the search would take about %.0f h in total (limit %.0f h): decide "
                "whether to add a second worker",
                figures["horas_totales_estimadas"],
                search.limit_hours,
            )

    finished = len(study.get_trials(deepcopy=False, states=FINISHED))
    remaining = search.trials - finished
    started = time.perf_counter()
    if remaining > 0:
        logger.info(
            "%s: %d trials finished, running up to %d more", search.name, finished, remaining
        )
        study.optimize(
            make_objective(ctx, study),
            n_trials=remaining,
            callbacks=[
                optuna.study.MaxTrialsCallback(search.trials, states=FINISHED),
                after_trial,
            ],
            catch=(torch.OutOfMemoryError,),
            gc_after_trial=True,
        )
    logger.info(
        "search %s: %.1f h in this run", search.name, (time.perf_counter() - started) / 3600
    )
    return study


__all__ = [
    "BLOCK_STEP_OFFSET",
    "ESTIMATE_FILE",
    "OOF_FILE",
    "SearchConfig",
    "block_step",
    "estimate",
    "load_search_config",
    "make_objective",
    "progress_step",
    "run_search",
    "split_params",
]
