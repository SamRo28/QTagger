"""MLflow client: connection check and runs that always carry their provenance.

Only remote tracking servers (``http://`` or ``https://``) are accepted, and the server is
checked (``/health``) before any run starts: if it cannot be reached the program aborts with
:class:`~qcclass.seguimiento.entorno.ServerConnectionError` instead of logging locally.

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
REQUEST_RETRIES = "3"
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


def configure_mlflow() -> str:
    """Check the server and point MLflow at it; return the URI."""
    uri = tracking_uri()
    check_mlflow(uri)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", REQUEST_TIMEOUT)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", REQUEST_RETRIES)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
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
) -> Iterator[Any]:
    """MLflow run with provenance tags, the configuration and the machine description.

    ``provenance`` holds the dataset stamp, the git state and the machine name (tags);
    ``configuration`` is logged flattened as parameters and whole as ``configuracion.json``.

    Raises:
        ServerConnectionError: if the server is not configured or cannot be reached.
    """
    configure_mlflow()
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=dict(provenance)) as run:
        mlflow.log_params(flatten(configuration))
        mlflow.log_dict(dict(configuration), "configuracion.json")
        mlflow.log_dict(dict(machine_description), "equipo.json")
        logger.info("MLflow run %s (%s) in experiment %s", run.info.run_id, run_name, experiment)
        yield run
