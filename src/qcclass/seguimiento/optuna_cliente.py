"""Shared Optuna storage: client from ``QCCLASS_OPTUNA_STORAGE`` and connection check.

Accepted values:

* ``grpc://<host>:<port>``: Optuna's gRPC storage proxy (``qcclass servidores`` on the
  server machine runs it in front of an SQLite database);
* ``postgresql://...`` or ``mysql...://...``: a relational database server.

File databases (``sqlite:``) and in-memory storage are refused: studies must be shared.

:func:`connect` returns a :class:`~qcclass.seguimiento.reintentos.RetryingStorage`: every call
is retried with exponential backoff when the server cannot be reached. While a model trains,
:class:`BufferedReports` keeps the intermediate values locally if the server is down (and
then never prunes), so a cut never stops the training. The gRPC proxy has no heartbeat:
:func:`fail_stale_trials` marks the trials left ``RUNNING`` by an interrupted process of this
machine as failed and, with ``requeue``, enqueues their hyperparameters again, so the trial
resumes from its checkpoints (identified by the hyperparameters and the block).
"""

import logging
import os
import warnings
from urllib.parse import urlparse

# gRPC's C core logs harmless INFO lines when the process forks (git, workers); read at import
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import optuna  # noqa: E402

from qcclass.seguimiento.entorno import OPTUNA_STORAGE_VAR, ServerConnectionError, required
from qcclass.seguimiento.reintentos import RetryingStorage, is_transient, retry

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


def connect(
    uri: str | None = None, timeout: float = CHECK_TIMEOUT, *, retries: bool = True
) -> RetryingStorage:
    """Storage client (with retries) after checking that the server answers (with retries
    unless ``retries=False``: status checks such as ``qcclass ping`` answer at once).

    Raises:
        ServerConnectionError: if it is still unreachable (after the retries).
    """
    uri = uri or storage_uri()
    storage = make_storage(uri)

    def check() -> None:
        try:
            if isinstance(storage, optuna.storages.GrpcStorageProxy):
                storage.wait_server_ready(timeout=timeout)
            storage.get_all_studies()
        except Exception as exc:  # gRPC, SQLAlchemy and network errors alike
            raise ServerConnectionError(f"cannot reach the Optuna storage at {uri}: {exc}") from exc

    if retries:
        retry(check, what="Optuna")
    else:
        check()
    return RetryingStorage(storage)


def study_summary(storage: optuna.storages.BaseStorage) -> dict[str, int]:
    """Number of trials of every study in the storage."""
    return {s.study_name: s.n_trials for s in optuna.get_all_study_summaries(storage)}


MACHINE_ATTR = "equipo"  # user attribute set by every trial when it starts


REQUEUED_ATTR = "reencolado_de"  # trial number this trial repeats
REQUEUES_ATTR = "reencolados"  # how many times these hyperparameters were requeued


def fail_stale_trials(
    study: optuna.Study,
    storage: optuna.storages.BaseStorage,
    machine: str,
    *,
    requeue: bool = False,
    max_requeues: int = 3,
) -> list[int]:
    """Mark as failed the ``RUNNING`` trials of ``machine`` left by an interrupted process.

    The gRPC proxy has no heartbeat, so an interrupted trial would stay ``RUNNING`` forever.
    Run this before optimising: it assumes one process per study and machine at a time.
    With ``requeue``, the hyperparameters of each such trial are enqueued again (at most
    ``max_requeues`` times in a row), so the next ``optimize`` repeats it first and the
    training resumes from its checkpoints. Returns the numbers of the trials marked failed.
    """
    study_id = storage.get_study_id_from_name(study.study_name)
    stale = [
        trial.number
        for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        if trial.user_attrs.get(MACHINE_ATTR) == machine
    ]
    trials = {t.number: t for t in study.get_trials(deepcopy=False)}
    for number in stale:
        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, number)
        storage.set_trial_state_values(trial_id, optuna.trial.TrialState.FAIL)
        if not requeue:
            continue
        times = int(trials[number].user_attrs.get(REQUEUES_ATTR, 0)) + 1
        if times > max_requeues:
            logger.error("trial %d already requeued %d times: not again", number, times - 1)
            continue
        study.enqueue_trial(
            trials[number].params,
            user_attrs={REQUEUED_ATTR: number, REQUEUES_ATTR: times},
        )
        logger.warning("study %s: trial %d requeued (%d)", study.study_name, number, times)
    if stale:
        logger.warning(
            "study %s: stale trials of %s marked as failed: %s", study.study_name, machine, stale
        )
    return stale


class BufferedReports:
    """Intermediate values of a trial that never block the training (module docstring).

    ``report`` tries to send the queued values once (``single_attempt``); if the server does
    not answer they stay queued. ``should_prune`` answers ``False`` while values are queued
    or the server cannot be read. ``flush`` sends what is left, with retries (end of trial).
    """

    def __init__(self, trial: optuna.Trial, storage: RetryingStorage) -> None:
        self.trial = trial
        self.storage = storage
        self.queued: list[tuple[float, int]] = []

    def _send(self) -> None:
        while self.queued:
            value, step = self.queued[0]
            self.trial.report(value, step)
            self.queued.pop(0)

    def report(self, value: float, step: int) -> bool:
        """Queue ``value`` at ``step`` and try to send the queue; ``True`` if all was sent."""
        self.queued.append((float(value), int(step)))
        try:
            with self.storage.single_attempt():
                self._send()
        except Exception as error:
            if not is_transient(error):
                raise
            logger.warning("Optuna unreachable: %d intermediate values queued", len(self.queued))
            return False
        return True

    def should_prune(self) -> bool:
        """Pruning decision, or ``False`` if it cannot be taken now."""
        if self.queued:
            return False
        try:
            with self.storage.single_attempt():
                return self.trial.should_prune()
        except Exception as error:
            if not is_transient(error):
                raise
            return False

    def flush(self) -> None:
        """Send the queued values with retries."""
        if self.queued:
            retry(self._send, what="Optuna report", backoff=self.storage.backoff)
