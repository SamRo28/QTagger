"""Retries with exponential backoff for the tracking servers (MLflow and Optuna).

A call that fails with a *transient* error (the server cannot be reached, a timeout, a 5xx
answer; see :func:`is_transient`) is repeated after 1, 2, 4... seconds, each wait capped at
``espera_maxima``, until it succeeds or ``total`` seconds have passed (about 10 minutes by
default); then the last error is raised. Any other error is raised at once.

The waits come from the environment / ``.env`` (see ``docs/EQUIPOS.md``):
``QCCLASS_REINTENTO_ESPERA_INICIAL`` (1 s), ``QCCLASS_REINTENTO_ESPERA_MAXIMA`` (60 s) and
``QCCLASS_REINTENTO_TOTAL`` (600 s).

:class:`RetryingStorage` wraps an Optuna storage so that every call is retried. Inside
``storage.single_attempt()`` calls are tried once (used while a model trains, so that a cut
of the server never stops the training; see ``seguimiento.registro``).
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import optuna

from qcclass.seguimiento.entorno import ServerConnectionError

logger = logging.getLogger(__name__)

INITIAL_VAR = "QCCLASS_REINTENTO_ESPERA_INICIAL"
MAXIMUM_VAR = "QCCLASS_REINTENTO_ESPERA_MAXIMA"
TOTAL_VAR = "QCCLASS_REINTENTO_TOTAL"
TRANSIENT_GRPC = ("UNAVAILABLE", "DEADLINE_EXCEEDED")
TRANSIENT_HTTP = (408, 429, 500, 502, 503, 504)
TRANSIENT_TEXT = (
    "max retries exceeded",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to establish a new connection",
    "timed out",
    "name or service not known",
    "temporarily unavailable",
)


@dataclass(frozen=True)
class Backoff:
    """Exponential backoff: waits ``inicial * 2**k`` capped at ``maxima``, within ``total``."""

    inicial: float = 1.0
    maxima: float = 60.0
    total: float = 600.0

    @classmethod
    def from_env(cls) -> "Backoff":
        """Read the waits from the environment (defaults above)."""

        def number(name: str, default: float) -> float:
            value = os.environ.get(name, "").strip()
            if not value:
                return default
            try:
                return float(value)
            except ValueError as exc:
                raise ServerConnectionError(f"{name} must be a number, got {value!r}") from exc

        return cls(
            inicial=number(INITIAL_VAR, cls.inicial),
            maxima=number(MAXIMUM_VAR, cls.maxima),
            total=number(TOTAL_VAR, cls.total),
        )

    def delays(self) -> Iterator[float]:
        """Successive waits whose sum stays within ``total``."""
        spent, delay = 0.0, self.inicial
        while spent + delay <= self.total:
            yield delay
            spent += delay
            delay = min(2 * delay, self.maxima)
        if spent < self.total:
            yield self.total - spent


def _chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient(error: BaseException) -> bool:
    """Whether ``error`` (or an error it was raised from) means "server unreachable or busy"."""
    for item in _chain(error):
        if isinstance(item, ServerConnectionError | ConnectionError | TimeoutError):
            return True
        code = getattr(item, "code", None)
        if callable(code):  # grpc.RpcError
            try:
                if getattr(code(), "name", "") in TRANSIENT_GRPC:
                    return True
            except Exception:  # noqa: BLE001 - a code() that fails is not informative
                pass
        if type(item).__name__ in ("OperationalError", "DisconnectionError"):  # SQLAlchemy
            return True
        status = getattr(getattr(item, "response", None), "status_code", None)
        status = status or getattr(item, "status_code", None)
        if isinstance(status, int) and status in TRANSIENT_HTTP:
            return True
        if type(item).__module__.startswith(("requests", "urllib3")) and isinstance(item, OSError):
            return True
        text = str(item).lower()
        if any(fragment in text for fragment in TRANSIENT_TEXT):
            return True
    return False


def retry[T](
    call: Callable[[], T],
    *,
    what: str,
    backoff: Backoff | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``call``, retrying transient errors with exponential backoff (module docstring)."""
    backoff = backoff or Backoff.from_env()
    delays = backoff.delays()
    while True:
        try:
            return call()
        except Exception as error:
            if not is_transient(error):
                raise
            delay = next(delays, None)
            if delay is None:
                logger.error("%s: server still unreachable after %.0f s", what, backoff.total)
                raise
            logger.warning("%s: %s; retrying in %.0f s", what, _short(error), delay)
            sleep(delay)


def _short(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text[:200]}"


# --------------------------------------------------------------------------------------------
# Optuna storage
# --------------------------------------------------------------------------------------------

_STORAGE_METHODS = (
    "check_trial_is_updatable",
    "create_new_study",
    "create_new_trial",
    "delete_study",
    "get_all_studies",
    "get_all_trials",
    "get_best_trial",
    "get_n_trials",
    "get_study_directions",
    "get_study_id_from_name",
    "get_study_name_from_id",
    "get_study_system_attrs",
    "get_study_user_attrs",
    "get_trial",
    "get_trial_id_from_study_id_trial_number",
    "get_trial_number_from_id",
    "get_trial_param",
    "get_trial_params",
    "get_trial_system_attrs",
    "get_trial_user_attrs",
    "remove_session",
    "set_study_system_attr",
    "set_study_user_attr",
    "set_trial_intermediate_value",
    "set_trial_param",
    "set_trial_state_values",
    "set_trial_system_attr",
    "set_trial_user_attr",
)


def _delegate(name: str) -> Callable[..., Any]:
    def method(self: "RetryingStorage", *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.inner, name)
        if self._attempts.single:
            return target(*args, **kwargs)
        return retry(lambda: target(*args, **kwargs), what=f"Optuna {name}", backoff=self.backoff)

    method.__name__ = name
    method.__doc__ = f"``{name}`` of the wrapped storage, with retries."
    return method


class _Attempts(threading.local):
    single = False


def _init(
    self: "RetryingStorage", inner: optuna.storages.BaseStorage, backoff: Backoff | None = None
) -> None:
    self.inner = inner
    self.backoff = backoff or Backoff.from_env()
    self._attempts = _Attempts()


@contextmanager
def _single_attempt(self: "RetryingStorage") -> Iterator[None]:
    previous = self._attempts.single
    self._attempts.single = True
    try:
        yield
    finally:
        self._attempts.single = previous


def _getattr(self: "RetryingStorage", name: str) -> Any:
    if name in ("inner", "backoff", "_attempts"):  # not set yet (unpickling)
        raise AttributeError(name)
    return getattr(self.inner, name)


# Built with ``type`` so that ABCMeta sees every abstract method implemented.
RetryingStorage = type(
    "RetryingStorage",
    (optuna.storages.BaseStorage,),
    {
        "__doc__": "Optuna storage whose calls are retried (see the module docstring).",
        "__module__": __name__,
        "__init__": _init,
        "__getattr__": _getattr,
        "single_attempt": _single_attempt,
        **{name: _delegate(name) for name in _STORAGE_METHODS},
    },
)
