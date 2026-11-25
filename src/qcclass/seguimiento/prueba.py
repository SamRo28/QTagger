"""Phase 9 check of the shared servers (``qcclass prueba-servidores``).

From each client machine: one MLflow run with the full provenance, and ``intentos`` trivial
Optuna trials (minimise ``(x - 1)^2``) added to one shared study. Every trial records the
machine that ran it, so the study shows that several machines write to the same storage.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import optuna

from qcclass.config import DatasetConfig
from qcclass.seguimiento.mlflow_cliente import tracked_run
from qcclass.seguimiento.optuna_cliente import connect
from qcclass.seguimiento.procedencia import start_provenance
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

MACHINE_ATTR = "equipo"


@dataclass(frozen=True)
class ServerTestResult:
    """Outcome of the server test."""

    experiment: str
    run_id: str
    study: str
    machine: str
    added: int
    trials_by_machine: dict[str, int]


def _objective(trial: optuna.Trial, machine: str) -> float:
    trial.set_user_attr(MACHINE_ATTR, machine)
    x = trial.suggest_float("x", -5.0, 5.0)
    return (x - 1.0) ** 2


def run_server_test(config: DatasetConfig, settings: dict[str, Any]) -> ServerTestResult:
    """Run the check described in the module docstring."""
    experiment = settings["experimento_mlflow"]
    provenance = start_provenance(config, experiment, device="cpu")
    storage = connect()  # abort before creating the MLflow run if Optuna is down
    run_name = f"prueba-{provenance.machine}"
    with tracked_run(
        experiment=experiment,
        run_name=run_name,
        configuration=settings,
        provenance=provenance.tags,
        machine_description=provenance.machine_description,
    ) as run:
        sampler = optuna.samplers.TPESampler(
            seed=derive_seed(settings["semilla"], "prueba_servidores", provenance.machine)
        )
        study = optuna.create_study(
            study_name=settings["estudio_optuna"],
            storage=storage,
            sampler=sampler,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: _objective(trial, provenance.machine), n_trials=settings["intentos"]
        )
        run.log_metric("mejor_valor_estudio", study.best_value)
        complete = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        by_machine = Counter(t.user_attrs.get(MACHINE_ATTR, "?") for t in complete)
        run.log_dict(dict(by_machine), "intentos_por_equipo.json")
        run_id = run.run_id
    return ServerTestResult(
        experiment=experiment,
        run_id=run_id,
        study=settings["estudio_optuna"],
        machine=provenance.machine,
        added=settings["intentos"],
        trials_by_machine=dict(by_machine),
    )
