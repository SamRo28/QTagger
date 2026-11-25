"""MLflow run that survives server cuts: everything is queued locally and sent when possible.

:class:`ResilientRun` keeps the operations of a run (tags, parameters, metrics, artifacts) in
a queue that is also written to ``<spool>/pendientes.json``, together with the run id in
``<spool>/run.json`` once the run exists. Two ways of sending them:

* ``flush(block=False)``: one quick attempt (a ``/health`` check, then each operation once);
  if the server does not answer, the operations stay queued and the caller goes on. Used
  after every epoch, so a cut of the server never stops a training.
* ``flush(block=True)``: retries with exponential backoff (``seguimiento.reintentos``) and
  raises :class:`~qcclass.seguimiento.entorno.ServerConnectionError` if the server is still
  unreachable at the end. Used when a run (or an Optuna trial) finishes.

A run created with the same ``spool`` directory continues the same MLflow run and sends
what was left queued (resumed trainings). Only remote servers are accepted
(``mlflow_cliente.tracking_uri``); nothing is ever logged to a local MLflow store.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from qcclass.circuitos import write_text_atomic
from qcclass.seguimiento.entorno import ServerConnectionError
from qcclass.seguimiento.reintentos import Backoff, is_transient, retry

logger = logging.getLogger(__name__)

PENDING_FILE = "pendientes.json"
RUN_FILE = "run.json"
ARTIFACTS_DIR = "artefactos"
QUICK_CHECK_TIMEOUT = 2.0  # seconds, health check of a non-blocking flush


class ResilientRun:
    """An MLflow run with a local queue of operations (see the module docstring)."""

    def __init__(
        self,
        *,
        experiment: str,
        run_name: str,
        tags: dict[str, str],
        spool: Path | None = None,
        backoff: Backoff | None = None,
    ) -> None:
        from qcclass.seguimiento.mlflow_cliente import configure_client_environment, tracking_uri

        configure_client_environment()
        self.uri = tracking_uri()
        self.experiment = experiment
        self.run_name = run_name
        self.backoff = backoff or Backoff.from_env()
        self._temporary = None if spool is not None else tempfile.TemporaryDirectory()
        self.spool = Path(spool) if spool is not None else Path(self._temporary.name)
        self.spool.mkdir(parents=True, exist_ok=True)
        stored = self._read(RUN_FILE, {})
        self.run_id: str | None = stored.get("run_id")
        self._pending: list[dict[str, Any]] = self._read(PENDING_FILE, [])
        if self.run_id is None and not self._pending:
            self._queue({"op": "tags", "values": {"mlflow.runName": run_name, **tags}})

    # ---- queue ---------------------------------------------------------------------------

    def _read(self, name: str, default: Any) -> Any:
        path = self.spool / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default

    def _save(self) -> None:
        write_text_atomic(json.dumps(self._pending, ensure_ascii=False), self.spool / PENDING_FILE)

    def _queue(self, operation: dict[str, Any]) -> None:
        self._pending.append(operation)
        self._save()

    @property
    def pending(self) -> int:
        """Number of queued operations."""
        return len(self._pending)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Queue tags."""
        self._queue({"op": "tags", "values": {k: str(v) for k, v in tags.items()}})

    def log_params(self, params: dict[str, Any]) -> None:
        """Queue parameters (values as strings)."""
        self._queue({"op": "params", "values": {k: str(v) for k, v in params.items()}})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Queue metrics with the current time."""
        stamp = int(time.time() * 1000)
        values = [[k, float(v), stamp, step or 0] for k, v in metrics.items()]
        self._queue({"op": "metrics", "values": values})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Queue one metric."""
        self.log_metrics({key: value}, step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        """Queue a file (it must still exist when the queue is sent)."""
        self._queue({"op": "artifact", "path": str(path), "artifact_path": artifact_path})

    def log_artifacts(self, directory: Path | str, artifact_path: str | None = None) -> None:
        """Queue every file of a directory (sent as it is when the queue is sent)."""
        self._queue({"op": "artifacts", "path": str(directory), "artifact_path": artifact_path})

    def log_dict(self, content: Any, name: str) -> None:
        """Write ``content`` as JSON into the spool and queue it as an artifact ``name``."""
        path = self.spool / ARTIFACTS_DIR / f"{len(self._pending):05d}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(json.dumps(content, indent=2, ensure_ascii=False, default=str), path)
        parent = str(Path(name).parent) if Path(name).parent != Path(".") else None
        self.log_artifact(path, parent)

    # ---- sending -------------------------------------------------------------------------

    def _client(self) -> Any:
        from mlflow import MlflowClient

        return MlflowClient(tracking_uri=self.uri)

    def _ensure_run(self, client: Any) -> None:
        if self.run_id is not None:
            return
        experiment = client.get_experiment_by_name(self.experiment)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else client.create_experiment(self.experiment)
        )
        first = self._pending[0] if self._pending and self._pending[0]["op"] == "tags" else None
        tags = first["values"] if first else {"mlflow.runName": self.run_name}
        run = client.create_run(experiment_id, tags=tags, run_name=self.run_name)
        self.run_id = run.info.run_id
        write_text_atomic(json.dumps({"run_id": self.run_id}), self.spool / RUN_FILE)
        if first:
            self._pending.pop(0)
            self._save()
        logger.info(
            "MLflow run %s (%s) in experiment %s", self.run_id, self.run_name, self.experiment
        )

    def _send(self, client: Any, operation: dict[str, Any]) -> None:
        from mlflow.entities import Metric, Param, RunTag

        kind = operation["op"]
        if kind == "tags":
            client.log_batch(
                self.run_id, tags=[RunTag(k, v) for k, v in operation["values"].items()]
            )
        elif kind == "params":
            params = [Param(k, v) for k, v in operation["values"].items()]
            for start in range(0, len(params), 100):  # MLflow: 100 parameters per batch
                client.log_batch(self.run_id, params=params[start : start + 100])
        elif kind == "metrics":
            metrics = [Metric(k, v, t, s) for k, v, t, s in operation["values"]]
            for start in range(0, len(metrics), 1000):
                client.log_batch(self.run_id, metrics=metrics[start : start + 1000])
        elif kind == "artifact":
            if Path(operation["path"]).is_file():
                client.log_artifact(self.run_id, operation["path"], operation["artifact_path"])
            else:
                logger.warning("artifact %s no longer exists: skipped", operation["path"])
        elif kind == "artifacts":
            if Path(operation["path"]).is_dir():
                client.log_artifacts(self.run_id, operation["path"], operation["artifact_path"])
        elif kind == "end":
            client.set_terminated(self.run_id, operation["status"])
        else:
            raise ValueError(f"unknown queued operation {kind!r}")

    def _send_all(self) -> None:
        client = self._client()
        self._ensure_run(client)
        while self._pending:
            self._send(client, self._pending[0])
            self._pending.pop(0)
            self._save()

    def flush(self, *, block: bool) -> bool:
        """Send the queue (see the module docstring); ``True`` if nothing is left.

        Raises:
            ServerConnectionError: with ``block=True``, if the server is still unreachable
                after the retries.
        """
        from qcclass.seguimiento.mlflow_cliente import check_mlflow

        if not self._pending and self.run_id is not None:
            return True
        if not block:
            try:
                check_mlflow(self.uri, timeout=QUICK_CHECK_TIMEOUT)
                self._send_all()
            except Exception as error:
                if not is_transient(error):
                    raise
                logger.warning(
                    "MLflow unreachable: %d operations kept in %s", len(self._pending), self.spool
                )
                return False
            return True
        try:
            retry(self._send_all, what="MLflow", backoff=self.backoff)
        except Exception as error:
            if is_transient(error):
                raise ServerConnectionError(
                    f"MLflow at {self.uri} still unreachable; {len(self._pending)} operations "
                    f"kept in {self.spool}"
                ) from error
            raise
        return True

    def finish(self, status: str = "FINISHED") -> None:
        """Queue the end of the run and send everything (blocking)."""
        self._queue({"op": "end", "status": status})
        self.flush(block=True)
        if self._temporary is not None:
            self._temporary.cleanup()

    def __repr__(self) -> str:
        return f"ResilientRun({self.run_name!r}, run_id={self.run_id}, pending={self.pending})"
