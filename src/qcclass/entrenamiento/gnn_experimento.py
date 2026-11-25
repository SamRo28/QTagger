"""GNN experiments of the phase 11: one block (``qcclass entrenar-gnn``) and the
mixed-precision check (``qcclass comprobar-precision-mixta``).

Results go to ``<resultados>/gnn/<nombre>/<fingerprint>/bloque_<k>/`` (see
:func:`~qcclass.entrenamiento.gnn.run_directory`): checkpoints, ``resultado.json``,
``predicciones.parquet``, ``arquitectura.json`` and the local MLflow queue (``mlflow/``). The
MLflow run carries the provenance, the configuration and the metrics of every epoch; the
run continues after an interruption (same directory, same run).
"""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qcclass.circuitos import write_text_atomic
from qcclass.config import ConfigError, DatasetConfig, SplitsConfig, load_experiment_config
from qcclass.entrenamiento.bloques import fold_training, load_pool
from qcclass.entrenamiento.gnn import (
    ARCHITECTURE_FILE,
    PREDICTIONS_FILE,
    RESULT_FILE,
    BlockResult,
    TrainingSettings,
    resolve_training_device,
    run_directory,
    train_block,
)
from qcclass.modelos.gnn import GNNConfig
from qcclass.seguimiento.entorno import int_setting
from qcclass.seguimiento.mlflow_cliente import flatten, wait_for_mlflow
from qcclass.seguimiento.procedencia import start_provenance
from qcclass.seguimiento.registro import ResilientRun

logger = logging.getLogger(__name__)

OUTPUT_DIR = "gnn"
WORKERS_VAR = "QCCLASS_PROCESOS_CARGA"
DEFAULT_WORKERS = 4
REQUIRED = ("nombre", "experimento_mlflow", "semilla", "modelo", "entrenamiento")
AMP_REPORT = "precision_mixta.json"


@dataclass(frozen=True)
class GNNExperiment:
    """Parsed GNN experiment configuration."""

    name: str
    mlflow_experiment: str
    seed: int
    model: dict[str, Any]
    training: TrainingSettings
    amp_check_block: int
    amp_tolerance: float
    raw: dict[str, Any]

    def params(self, training: TrainingSettings | None = None) -> dict[str, Any]:
        """Hyperparameters that identify a run (fingerprint of its directory)."""
        return {
            "modelo": self.model,
            "entrenamiento": (training or self.training).as_dict(),
            "semilla": self.seed,
        }


def load_gnn_experiment(path: Path | str) -> GNNExperiment:
    """Load a GNN experiment configuration.

    Raises:
        ConfigError: if a key is missing or a value is invalid.
    """
    data = load_experiment_config(path, REQUIRED)
    try:
        training = TrainingSettings.from_mapping(dict(data["entrenamiento"]))
        model = dict(data["modelo"])
        GNNConfig.from_mapping(model, node_dim=1, edge_dim=1, num_classes=2, num_families=2)
        check = data.get("comprobacion_precision_mixta") or {}
        return GNNExperiment(
            name=str(data["nombre"]),
            mlflow_experiment=str(data["experimento_mlflow"]),
            seed=int(data["semilla"]),
            model=model,
            training=training,
            amp_check_block=int(check.get("bloque", 0)),
            amp_tolerance=float(check.get("tolerancia_f1", 0.01)),
            raw=data,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


def loading_workers() -> int:
    """Data-loading processes of this machine (``QCCLASS_PROCESOS_CARGA``, default 4)."""
    return int_setting(WORKERS_VAR, DEFAULT_WORKERS)


def train_one_block(
    config: DatasetConfig,
    splits: SplitsConfig,
    experiment: GNNExperiment,
    *,
    block: int,
    device: str = "auto",
    training: TrainingSettings | None = None,
    workers: int | None = None,
    max_batches: int | None = None,
) -> BlockResult:
    """Train ``experiment`` on one block with an MLflow run (see the module docstring)."""
    training = training or experiment.training
    device = resolve_training_device(device)
    provenance = start_provenance(config, experiment.name, device=device)
    pool = load_pool(config, splits, experiment.name)
    if not 0 <= block < pool.n_blocks:
        raise ValueError(f"block must be in 0..{pool.n_blocks - 1}")
    fold = fold_training(pool, block, splits)
    params = experiment.params(training)
    directory = run_directory(config.rutas.resultados / OUTPUT_DIR, experiment.name, params, block)
    directory.mkdir(parents=True, exist_ok=True)
    amp = training.precision_mixta and device == "cuda"
    workers = loading_workers() if workers is None else workers
    from qcclass.seguimiento.mlflow_cliente import tracking_uri

    wait_for_mlflow(tracking_uri())  # never start without the server (it may fail later)
    run = ResilientRun(
        experiment=experiment.mlflow_experiment,
        run_name=f"{experiment.name}.bloque{block}{'' if amp else '.fp32'}",
        tags={**provenance.tags, "bloque": str(block), "huella": directory.parent.name},
        spool=directory / "mlflow",
    )
    if run.run_id is None:
        run.log_params(
            flatten(
                {
                    **params,
                    "bloque": block,
                    "dispositivo": device,
                    "precision_mixta": amp,
                    "procesos_carga": workers,
                }
            )
        )
        run.log_dict(experiment.raw, "configuracion.json")
        run.log_dict(provenance.machine_description, "equipo.json")
    run.flush(block=True)
    try:
        result = train_block(
            pool=pool,
            fold=fold,
            graphs_dir=config.rutas.graphs,
            model_settings=experiment.model,
            training=training,
            seed=experiment.seed,
            device=device,
            directory=directory,
            workers=workers,
            run=run,
            max_batches=max_batches,
        )
    except BaseException:
        run.finish("FAILED")
        raise
    run.log_metrics(
        {
            "final_f1_macro_algoritmo": result.f1_macro_algoritmo,
            "final_f1_macro_familia": result.f1_macro_familia,
            "segundos_totales": result.seconds,
        }
    )
    for name in (RESULT_FILE, ARCHITECTURE_FILE, PREDICTIONS_FILE):
        run.log_artifact(directory / name)
    run.finish("FINISHED")
    return result


def check_mixed_precision(
    config: DatasetConfig,
    splits: SplitsConfig,
    experiment: GNNExperiment,
    *,
    device: str = "auto",
    workers: int | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Train the reference block with and without mixed precision and compare (plan,
    phase 11 task 5); writes ``<resultados>/gnn/precision_mixta.json``.

    Raises:
        ValueError: if the device is not CUDA (mixed precision only exists there).
    """
    if resolve_training_device(device) != "cuda":
        raise ValueError("the mixed-precision check needs CUDA")
    block = experiment.amp_check_block
    runs = {}
    for amp in (True, False):
        training = replace(experiment.training, precision_mixta=amp)
        result = train_one_block(
            config,
            splits,
            experiment,
            block=block,
            device="cuda",
            training=training,
            workers=workers,
            max_batches=max_batches,
        )
        epochs = [h["segundos_epoca"] for h in result.history]
        runs["con" if amp else "sin"] = {
            "f1_macro_algoritmo": result.f1_macro_algoritmo,
            "f1_macro_familia": result.f1_macro_familia,
            "segundos_epoca_mediana": sorted(epochs)[len(epochs) // 2],
            "memoria_gpu_mb_max": max(h.get("memoria_gpu_mb", 0) for h in result.history),
            "directorio": str(result.directory),
        }
    difference = runs["con"]["f1_macro_algoritmo"] - runs["sin"]["f1_macro_algoritmo"]
    report = {
        "bloque": block,
        "tolerancia_f1": experiment.amp_tolerance,
        "diferencia_f1_macro_algoritmo": difference,
        "mantener_precision_mixta": abs(difference) <= experiment.amp_tolerance,
        **runs,
    }
    path = config.rutas.resultados / OUTPUT_DIR / AMP_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(json.dumps(report, indent=2, ensure_ascii=False) + "\n", path)
    logger.info("mixed precision: F1 difference %.4f (%s)", difference, path)
    return report


__all__ = [
    "AMP_REPORT",
    "OUTPUT_DIR",
    "GNNExperiment",
    "check_mixed_precision",
    "load_gnn_experiment",
    "loading_workers",
    "train_one_block",
]
