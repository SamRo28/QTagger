"""Provenance of a run: dataset version, git state, machine and hardware.

:func:`start_provenance` is the first call of every experiment command: it loads ``.env``,
checks the dataset fingerprints (:func:`qcclass.versionado.begin_experiment`, abort if they
differ) and returns what every MLflow run must record.
"""

from dataclasses import dataclass, field

from qcclass.config import DatasetConfig
from qcclass.medicion.equipo import machine_info, machine_name
from qcclass.seguimiento.entorno import load_env, machine
from qcclass.seguimiento.mlflow_cliente import git_state
from qcclass.versionado import begin_experiment, library_versions


@dataclass(frozen=True)
class Provenance:
    """Tags and machine description of a run."""

    tags: dict[str, str]
    machine: str
    machine_description: dict = field(default_factory=dict)


def start_provenance(config: DatasetConfig, experiment: str, device: str) -> Provenance:
    """Load ``.env``, check the dataset and describe the run (see the module docstring)."""
    load_env(config.rutas.raiz)
    stamp = begin_experiment(config, experiment)
    name = machine(machine_name())
    tags = {
        **stamp,
        **git_state(config.rutas.raiz),
        "equipo": name,
        "dispositivo": device,
        "experimento": experiment,
        **{f"version.{lib}": v for lib, v in library_versions().items()},  # installed now
    }
    return Provenance(tags=tags, machine=name, machine_description=machine_info(device, name))
