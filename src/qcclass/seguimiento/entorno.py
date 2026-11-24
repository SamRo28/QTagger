"""Per-machine settings read from the environment and from the unversioned ``.env`` file.

The ``.env`` file lives in the repository root (``rutas.raiz``) and is never committed. Values
already present in the environment take precedence. See ``docs/EQUIPOS.md``.

Client variables (every machine that runs experiments):

* ``QCCLASS_EQUIPO``: machine name recorded with every run (``m3pro``, ``rtx3050``...).
* ``QCCLASS_MLFLOW_URI``: tracking server, e.g. ``http://192.168.1.133:5050``.
* ``QCCLASS_OPTUNA_STORAGE``: shared Optuna storage, e.g. ``grpc://192.168.1.133:13000``.

Server variables (only the machine that runs ``qcclass servidores``):

* ``QCCLASS_SERVIDORES_DIR``: directory for the databases, artifacts, logs and pid files.
* ``QCCLASS_SERVIDORES_HOST`` (default ``0.0.0.0``), ``QCCLASS_MLFLOW_PUERTO`` (default 5050)
  and ``QCCLASS_OPTUNA_PUERTO`` (default 13000).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_FILE = ".env"
MACHINE_VAR = "QCCLASS_EQUIPO"
MLFLOW_URI_VAR = "QCCLASS_MLFLOW_URI"
OPTUNA_STORAGE_VAR = "QCCLASS_OPTUNA_STORAGE"
SERVERS_DIR_VAR = "QCCLASS_SERVIDORES_DIR"
SERVERS_HOST_VAR = "QCCLASS_SERVIDORES_HOST"
MLFLOW_PORT_VAR = "QCCLASS_MLFLOW_PUERTO"
OPTUNA_PORT_VAR = "QCCLASS_OPTUNA_PUERTO"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_MLFLOW_PORT = 5050  # 5000 is taken by the AirPlay receiver on macOS
DEFAULT_OPTUNA_PORT = 13000
HELP = "see docs/EQUIPOS.md"


class ServerConnectionError(RuntimeError):
    """A tracking server is not configured or cannot be reached."""


def load_env(root: Path) -> Path | None:
    """Load ``<root>/.env`` without overriding the environment; return its path if found."""
    path = Path(root) / ENV_FILE
    if not path.is_file():
        logger.debug("no %s in %s", ENV_FILE, root)
        return None
    load_dotenv(path, override=False)
    return path


def required(name: str) -> str:
    """Value of a required variable.

    Raises:
        ServerConnectionError: if it is not set (nothing is ever logged locally instead).
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ServerConnectionError(f"{name} is not set in the environment or in .env ({HELP})")
    return value


def machine(default: str) -> str:
    """Machine name: ``QCCLASS_EQUIPO`` or ``default`` (the host name)."""
    return os.environ.get(MACHINE_VAR, "").strip() or default


def int_setting(name: str, default: int) -> int:
    """Integer variable with a default.

    Raises:
        ServerConnectionError: if it is set but not an integer.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ServerConnectionError(f"{name} must be an integer, got {value!r}") from exc
