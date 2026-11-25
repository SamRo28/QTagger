"""MLflow client: connection check and runs that always carry their provenance.

Only remote tracking servers (``http://`` or ``https://``) are accepted, and the server is
checked (``/health``, with the retries of ``seguimiento.reintentos``) before any run starts:
if it cannot be reached the program aborts with
:class:`~qcclass.seguimiento.entorno.ServerConnectionError` instead of logging locally. Runs
are :class:`~qcclass.seguimiento.registro.ResilientRun` objects: what they log is queued and
sent with retries, so a short cut of the server does not lose anything.

Every run records the dataset version and fingerprint, the git commit (and whether the
working tree had uncommitted changes), the machine, its hardware description and the full
configuration (as parameters and as ``configuracion.json``).
"""

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qcclass.seguimiento.entorno import MLFLOW_URI_VAR, ServerConnectionError, required

logger = logging.getLogger(__name__)

CHECK_TIMEOUT = 5  # seconds
REQUEST_TIMEOUT = "60"  # seconds per MLflow HTTP request
REQUEST_RETRIES = "1"  # MLflow's own retries; ours (seguimiento.reintentos) come on top
PARAM_LIMIT = 6000  # MLflow limit on parameter values


def tracking_uri() -> str:
    """``QCCLASS_MLFLOW_URI``, which must be an http(s) server.

    Raises:
        ServerConnectionError: if it is missing or not a server URI.
    """
    uri = required(MLFLOW_URI_VAR).rstrip("/")
    if not uri.startswith(("http://", "https://")):
        raise ServerConnectionError(
            f"{MLFLOW_URI_VAR}={uri!r} is not a tracking server (http:// or https://); "
            "local tracking is not allowed"
        )
    return uri


def check_mlflow(uri: str, timeout: float = CHECK_TIMEOUT) -> str:
    """Check that the server answers ``/health``; return its version.

    Raises:
        ServerConnectionError: if the server cannot be reached.
    """
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=timeout) as response:
            if response.status != 200:
                raise ServerConnectionError(f"MLflow at {uri} answered {response.status}")
        with urllib.request.urlopen(f"{uri}/version", timeout=timeout) as response:
            return response.read().decode("utf-8").strip()
    except (urllib.error.URLError, OSError) as exc:
        raise ServerConnectionError(f"cannot reach MLflow at {uri}: {exc}") from exc


def configure_client_environment() -> None:
    """Timeouts and retries of MLflow's HTTP client (set before MLflow reads them)."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", REQUEST_TIMEOUT)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", REQUEST_RETRIES)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def wait_for_mlflow(uri: str) -> str:
    """:func:`check_mlflow` with retries; return the server version.

    Raises:
        ServerConnectionError: if the server is still unreachable after the retries.
    """
    from qcclass.seguimiento.reintentos import retry

    return retry(lambda: check_mlflow(uri), what="MLflow")


def configure_mlflow() -> str:
    """Check the server (with retries) and point MLflow at it; return the URI."""
    uri = tracking_uri()
    wait_for_mlflow(uri)
    configure_client_environment()
    import mlflow

    mlflow.set_tracking_uri(uri)
    return uri


def git_state(root: Path) -> dict[str, str]:
    """Current commit and whether versioned files have uncommitted changes (untracked files
    such as results or logs do not change the code and are ignored)."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_sucio": str(bool(git("status", "--porcelain", "--untracked-files=no"))),
    }


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested mapping as ``a.b.c`` keys with string values (MLflow parameters)."""
    flat = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            text = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, list | tuple)
                else str(value)
            )
            flat[name] = text[:PARAM_LIMIT]
    return flat


@contextmanager
def tracked_run(
    *,
    experiment: str,
    run_name: str,
    configuration: Mapping[str, Any],
    provenance: Mapping[str, str],
    machine_description: Mapping[str, Any],
    spool: Path | None = None,
) -> Iterator[Any]:
    """MLflow run with provenance tags, the configuration and the machine description.

    ``provenance`` holds the dataset stamp, the git state and the machine name (tags);
    ``configuration`` is logged flattened as parameters and whole as ``configuracion.json``.
    Yields a :class:`~qcclass.seguimiento.registro.ResilientRun` (``log_metric``,
    ``log_metrics``, ``log_params``, ``log_artifact``, ``log_artifacts``, ``log_dict``); the
    run is created before the body runs and everything is sent when it ends (status
    ``FAILED`` if the body raises). ``spool`` keeps the local queue (default: temporary).

    Raises:
        ServerConnectionError: if the server is not configured or still unreachable after
            the retries (at the start or at the end).
    """
    from qcclass.seguimiento.registro import ResilientRun

    uri = configure_mlflow()
    run = ResilientRun(experiment=experiment, run_name=run_name, tags=dict(provenance), spool=spool)
    run.log_params(flatten(configuration))
    run.log_dict(dict(configuration), "configuracion.json")
    run.log_dict(dict(machine_description), "equipo.json")
    run.flush(block=True)
    logger.debug("MLflow run %s at %s", run.run_id, uri)
    try:
        yield run
    except BaseException:
        run.finish("FAILED")
        raise
    run.finish("FINISHED")
