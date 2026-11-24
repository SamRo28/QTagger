"""Shared Optuna storage: client from ``QCCLASS_OPTUNA_STORAGE`` and connection check.

Accepted values:

* ``grpc://<host>:<port>``: Optuna's gRPC storage proxy (``qcclass servidores`` on the
  server machine runs it in front of an SQLite database);
* ``postgresql://...`` or ``mysql...://...``: a relational database server.

File databases (``sqlite:``) and in-memory storage are refused: studies must be shared.
"""

import logging
import os
import warnings
from urllib.parse import urlparse

# gRPC's C core logs harmless INFO lines when the process forks (git, workers); read at import
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import optuna  # noqa: E402

from qcclass.seguimiento.entorno import OPTUNA_STORAGE_VAR, ServerConnectionError, required

logger = logging.getLogger(__name__)

CHECK_TIMEOUT = 5  # seconds
RDB_SCHEMES = ("postgresql", "mysql")


def storage_uri() -> str:
    """``QCCLASS_OPTUNA_STORAGE`` (required)."""
    return required(OPTUNA_STORAGE_VAR)


def make_storage(uri: str) -> optuna.storages.BaseStorage:
    """Storage client for a shared storage URI (see the module docstring).

    Raises:
        ServerConnectionError: if the URI is not a shared storage.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "grpc":
        if not parsed.hostname or not parsed.port:
            raise ServerConnectionError(f"{OPTUNA_STORAGE_VAR}={uri!r}: use grpc://host:port")
        with warnings.catch_warnings():  # experimental since 4.2: chosen knowingly
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            return optuna.storages.GrpcStorageProxy(host=parsed.hostname, port=parsed.port)
    if parsed.scheme.split("+")[0] in RDB_SCHEMES:
        return optuna.storages.RDBStorage(uri)
    raise ServerConnectionError(
        f"{OPTUNA_STORAGE_VAR}={uri!r} is not a shared storage (grpc://, postgresql://, "
        "mysql://); local storage is not allowed"
    )


def connect(uri: str | None = None, timeout: float = CHECK_TIMEOUT) -> optuna.storages.BaseStorage:
    """Storage client after checking that the server answers.

    Raises:
        ServerConnectionError: if it cannot be reached.
    """
    uri = uri or storage_uri()
    try:
        storage = make_storage(uri)
        if isinstance(storage, optuna.storages.GrpcStorageProxy):
            storage.wait_server_ready(timeout=timeout)
        storage.get_all_studies()
    except ServerConnectionError:
        raise
    except Exception as exc:  # gRPC, SQLAlchemy and network errors alike
        raise ServerConnectionError(f"cannot reach the Optuna storage at {uri}: {exc}") from exc
    return storage


def study_summary(storage: optuna.storages.BaseStorage) -> dict[str, int]:
    """Number of trials of every study in the storage."""
    return {s.study_name: s.n_trials for s in optuna.get_all_study_summaries(storage)}


MACHINE_ATTR = "equipo"  # user attribute set by every trial when it starts


def fail_stale_trials(
    study: optuna.Study, storage: optuna.storages.BaseStorage, machine: str
) -> list[int]:
    """Mark as failed the ``RUNNING`` trials of ``machine`` left by an interrupted process.

    The gRPC proxy has no heartbeat, so an interrupted trial would stay ``RUNNING`` forever.
    Run this before optimising: it assumes one process per study and machine at a time.
    Returns the numbers of the trials marked as failed.
    """
    study_id = storage.get_study_id_from_name(study.study_name)
    stale = [
        trial.number
        for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        if trial.user_attrs.get(MACHINE_ATTR) == machine
    ]
    for number in stale:
        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, number)
        storage.set_trial_state_values(trial_id, optuna.trial.TrialState.FAIL)
    if stale:
        logger.warning(
            "study %s: stale trials of %s marked as failed: %s", study.study_name, machine, stale
        )
    return stale
