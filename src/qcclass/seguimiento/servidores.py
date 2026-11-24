"""Start, stop and inspect the tracking servers on the server machine (``m1pro``).

Two detached processes (they survive closing the terminal):

* MLflow tracking server: SQLite backend store and artifacts stored on this machine and
  served through the server itself (``--serve-artifacts``), so clients need no access to its
  disk;
* Optuna gRPC storage proxy (:mod:`qcclass.seguimiento.proxy_optuna`) in front of an SQLite
  database.

On macOS a ``caffeinate -i -s -w <pid>`` process keeps the machine awake (idle and, on AC
power, system sleep) for as long as the MLflow server lives. Everything is written under
``QCCLASS_SERVIDORES_DIR``: ``mlflow/``, ``optuna/``, ``logs/`` and ``pids/``.
"""

import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from qcclass.seguimiento.entorno import (
    DEFAULT_HOST,
    DEFAULT_MLFLOW_PORT,
    DEFAULT_OPTUNA_PORT,
    MLFLOW_PORT_VAR,
    OPTUNA_PORT_VAR,
    SERVERS_DIR_VAR,
    SERVERS_HOST_VAR,
    ServerConnectionError,
    int_setting,
    required,
)
from qcclass.seguimiento.mlflow_cliente import check_mlflow
from qcclass.seguimiento.optuna_cliente import connect

logger = logging.getLogger(__name__)

SERVICES = ("mlflow", "optuna", "caffeinate")
START_TIMEOUT = 90  # seconds until both servers must answer
STOP_TIMEOUT = 20  # seconds before SIGKILL


@dataclass(frozen=True)
class ServerSettings:
    """Where and how the servers run."""

    directory: Path
    host: str
    mlflow_port: int
    optuna_port: int

    @classmethod
    def from_env(cls) -> "ServerSettings":
        """Settings from ``QCCLASS_SERVIDORES_*`` (see :mod:`qcclass.seguimiento.entorno`)."""
        return cls(
            directory=Path(required(SERVERS_DIR_VAR)).expanduser(),
            host=os.environ.get(SERVERS_HOST_VAR, "").strip() or DEFAULT_HOST,
            mlflow_port=int_setting(MLFLOW_PORT_VAR, DEFAULT_MLFLOW_PORT),
            optuna_port=int_setting(OPTUNA_PORT_VAR, DEFAULT_OPTUNA_PORT),
        )

    def pid_file(self, service: str) -> Path:
        """``pids/<service>.pid``."""
        return self.directory / "pids" / f"{service}.pid"

    def log_file(self, service: str) -> Path:
        """``logs/<service>.log``."""
        return self.directory / "logs" / f"{service}.log"

    @property
    def local_mlflow_uri(self) -> str:
        """MLflow URI as seen from this machine."""
        return f"http://127.0.0.1:{self.mlflow_port}"

    @property
    def local_optuna_uri(self) -> str:
        """Optuna storage URI as seen from this machine."""
        return f"grpc://127.0.0.1:{self.optuna_port}"


def commands(settings: ServerSettings) -> dict[str, list[str]]:
    """Command line of each server."""
    directory = settings.directory
    mlflow = Path(sys.executable).parent / "mlflow"
    return {
        "mlflow": [
            str(mlflow),
            "server",
            "--host",
            settings.host,
            "--port",
            str(settings.mlflow_port),
            "--backend-store-uri",
            f"sqlite:///{directory / 'mlflow' / 'mlflow.db'}",
            "--artifacts-destination",
            str(directory / "mlflow" / "artefactos"),
            "--serve-artifacts",
        ],
        "optuna": [
            sys.executable,
            "-m",
            "qcclass.seguimiento.proxy_optuna",
            "--almacenamiento",
            f"sqlite:///{directory / 'optuna' / 'optuna.db'}",
            "--host",
            settings.host,
            "--puerto",
            str(settings.optuna_port),
        ],
    }


def running_pid(settings: ServerSettings, service: str) -> int | None:
    """Pid of a running service (stale pid files are removed)."""
    path = settings.pid_file(service)
    if not path.is_file():
        return None
    pid = int(path.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        path.unlink()
        return None
    except PermissionError:
        pass  # exists, owned by someone else
    return pid


def _spawn(settings: ServerSettings, service: str, command: list[str]) -> int:
    log = settings.log_file(service)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group: survives the terminal
        )
    pid_file = settings.pid_file(service)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def _wait_ready(settings: ServerSettings, timeout: float) -> list[str]:
    """Problems left after waiting up to ``timeout`` seconds for both servers."""
    deadline = time.monotonic() + timeout
    while True:
        problems = []
        try:
            check_mlflow(settings.local_mlflow_uri, timeout=2)
        except ServerConnectionError as exc:
            problems.append(str(exc))
        try:
            connect(settings.local_optuna_uri, timeout=2)
        except ServerConnectionError as exc:
            problems.append(str(exc))
        if not problems or time.monotonic() > deadline:
            return problems
        time.sleep(2)


def start(settings: ServerSettings) -> dict[str, int]:
    """Start the servers that are not running and wait until both answer.

    Raises:
        ServerConnectionError: if they do not answer within :data:`START_TIMEOUT` seconds.
    """
    for folder in ("mlflow/artefactos", "optuna", "logs", "pids"):
        (settings.directory / folder).mkdir(parents=True, exist_ok=True)
    pids = {}
    for service, command in commands(settings).items():
        pids[service] = running_pid(settings, service) or _spawn(settings, service, command)
    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        awake = ["caffeinate", "-i", "-s", "-w", str(pids["mlflow"])]
        pids["caffeinate"] = running_pid(settings, "caffeinate") or _spawn(
            settings, "caffeinate", awake
        )
    if problems := _wait_ready(settings, START_TIMEOUT):
        raise ServerConnectionError(
            "servers started but not answering: "
            + "; ".join(problems)
            + f" (logs in {settings.directory / 'logs'})"
        )
    return pids


def stop(settings: ServerSettings) -> dict[str, bool]:
    """Stop every running service (SIGTERM to its process group, SIGKILL after a while)."""
    stopped = {}
    for service in SERVICES:
        pid = running_pid(settings, service)
        if pid is None:
            stopped[service] = False
            continue
        _terminate(pid)
        settings.pid_file(service).unlink(missing_ok=True)
        stopped[service] = True
    return stopped


def _gone(pid: int) -> bool:
    """True once the process has exited (reaping it if it is a child of this process)."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return True
    except ChildProcessError:
        pass  # not our child: launchd/init reaps it
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _terminate(pid: int) -> None:
    """SIGTERM the process group of ``pid``; SIGKILL if it is still alive after a while."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):  # gone, or only a zombie leader left
            _gone(pid)
            return
        deadline = time.monotonic() + STOP_TIMEOUT
        while time.monotonic() < deadline:
            if _gone(pid):
                return
            time.sleep(0.5)
        logger.warning("process %d did not stop after signal %s", pid, sig.name)


def status(settings: ServerSettings) -> dict[str, str]:
    """State of every service (pid and whether it answers)."""
    result = {}
    for service in SERVICES:
        pid = running_pid(settings, service)
        result[service] = f"running (pid {pid})" if pid else "stopped"
    for service, check in (
        ("mlflow", lambda: f"MLflow {check_mlflow(settings.local_mlflow_uri)}"),
        ("optuna", lambda: f"{len(connect(settings.local_optuna_uri).get_all_studies())} studies"),
    ):
        if running_pid(settings, service):
            try:
                result[service] += f", answering: {check()}"
            except ServerConnectionError as exc:
                result[service] += f", NOT answering: {exc}"
    return result
