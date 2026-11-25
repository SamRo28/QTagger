"""Tests of the server retries: backoff, transient errors, Optuna storage wrapper, requeued
trials, buffered reports and the MLflow run with a local queue."""

import json
from pathlib import Path

import optuna
import pytest

from qcclass.seguimiento.entorno import ServerConnectionError
from qcclass.seguimiento.optuna_cliente import (
    MACHINE_ATTR,
    REQUEUED_ATTR,
    BufferedReports,
    fail_stale_trials,
)
from qcclass.seguimiento.reintentos import Backoff, RetryingStorage, is_transient, retry

QUICK = Backoff(inicial=0.01, maxima=0.02, total=0.1)


class FakeRpcError(Exception):
    """Mimics ``grpc.RpcError`` with a status code."""

    def __init__(self, name: str) -> None:
        super().__init__(f"status {name}")
        self.name = name

    def code(self):  # noqa: ANN201
        return type("Code", (), {"name": self.name})()


def test_backoff_doubles_and_stays_within_the_total() -> None:
    delays = list(Backoff(inicial=1, maxima=60, total=600).delays())
    assert delays[:7] == [1, 2, 4, 8, 16, 32, 60]
    assert sum(delays) == pytest.approx(600)
    assert list(Backoff(inicial=1, maxima=4, total=10).delays()) == [1, 2, 4, 3]


def test_backoff_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QCCLASS_REINTENTO_TOTAL", "30")
    assert Backoff.from_env().total == 30
    monkeypatch.setenv("QCCLASS_REINTENTO_TOTAL", "mucho")
    with pytest.raises(ServerConnectionError):
        Backoff.from_env()


def test_transient_errors() -> None:
    assert is_transient(FakeRpcError("UNAVAILABLE"))
    assert not is_transient(FakeRpcError("NOT_FOUND"))
    assert is_transient(ConnectionRefusedError())
    assert is_transient(ServerConnectionError("down"))
    wrapped = RuntimeError("API request failed")
    wrapped.__cause__ = ConnectionError("Max retries exceeded")
    assert is_transient(wrapped)
    assert not is_transient(ValueError("bad value"))
    assert not is_transient(KeyError("trial"))


def test_retry_until_success_and_give_up() -> None:
    calls, waits = [], []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("refused")
        return "ok"

    assert retry(flaky, what="t", backoff=QUICK, sleep=waits.append) == "ok"
    assert waits == [0.01, 0.02]

    def down() -> None:
        raise ConnectionError("refused")

    waits.clear()
    with pytest.raises(ConnectionError):
        retry(down, what="t", backoff=QUICK, sleep=waits.append)
    assert sum(waits) == pytest.approx(0.1)

    def broken() -> None:
        raise ValueError("bug")

    waits.clear()
    with pytest.raises(ValueError):
        retry(broken, what="t", backoff=QUICK, sleep=waits.append)
    assert waits == []


class FlakyStorage(optuna.storages.InMemoryStorage):
    """In-memory storage whose writes fail while ``down`` is set."""

    down = False

    def set_trial_intermediate_value(self, *args, **kwargs):  # noqa: ANN201
        if self.down:
            raise FakeRpcError("UNAVAILABLE")
        return super().set_trial_intermediate_value(*args, **kwargs)

    def get_all_trials(self, *args, **kwargs):  # noqa: ANN201
        if self.down:
            raise FakeRpcError("UNAVAILABLE")
        return super().get_all_trials(*args, **kwargs)


def test_retrying_storage_is_a_storage_and_retries() -> None:
    inner = FlakyStorage()
    storage = RetryingStorage(inner, QUICK)
    assert isinstance(storage, optuna.storages.BaseStorage)
    study = optuna.create_study(study_name="s", storage=storage)
    trial = study.ask()
    inner.down = True
    with pytest.raises(FakeRpcError), storage.single_attempt():
        trial.report(1.0, 0)
    with pytest.raises(FakeRpcError):  # retried until the budget is spent
        trial.report(1.0, 0)
    inner.down = False
    trial.report(1.0, 0)
    study.tell(trial, 1.0)
    assert study.trials[0].intermediate_values == {0: 1.0}


def test_buffered_reports_never_block_and_send_later() -> None:
    inner = FlakyStorage()
    storage = RetryingStorage(inner, QUICK)
    study = optuna.create_study(study_name="s", storage=storage)
    trial = study.ask()
    reports = BufferedReports(trial, storage)
    inner.down = True
    assert reports.report(0.5, 1) is False and reports.report(0.6, 2) is False
    assert reports.should_prune() is False and len(reports.queued) == 2
    inner.down = False
    assert reports.report(0.7, 3) is True and not reports.queued
    study.tell(trial, 0.7)
    assert study.trials[0].intermediate_values == {1: 0.5, 2: 0.6, 3: 0.7}


def test_stale_trials_are_requeued_with_their_hyperparameters() -> None:
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(study_name="s", storage=storage)
    trial = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    trial.set_user_attr(MACHINE_ATTR, "rtx3050")
    params = dict(trial.params)
    assert fail_stale_trials(study, storage, "rtx3050", requeue=True) == [0]
    again = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    assert again.params == params and again.user_attrs[REQUEUED_ATTR] == 0
    again.set_user_attr(MACHINE_ATTR, "rtx3050")
    for _ in range(3):  # requeued at most 3 times in a row
        fail_stale_trials(study, storage, "rtx3050", requeue=True, max_requeues=3)
        nxt = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
        nxt.set_user_attr(MACHINE_ATTR, "rtx3050")
    waiting = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.WAITING]
    assert not waiting
    states = [t.state.name for t in study.get_trials()]
    assert states.count("FAIL") == 4


# --------------------------------------------------------------------------------------------
# MLflow run with a local queue
# --------------------------------------------------------------------------------------------


class FakeClient:
    """Records MLflow client calls; raises while ``down``."""

    def __init__(self, log: list, state: dict) -> None:
        self.log, self.state = log, state

    def _check(self) -> None:
        if self.state["down"]:
            raise ConnectionError("Max retries exceeded")

    def get_experiment_by_name(self, name):  # noqa: ANN001, ANN201
        self._check()
        return None

    def create_experiment(self, name):  # noqa: ANN001, ANN201
        self._check()
        return "1"

    def create_run(self, experiment_id, tags=None, run_name=None):  # noqa: ANN001, ANN201
        self._check()
        self.log.append(("create", run_name, dict(tags)))
        return type("R", (), {"info": type("I", (), {"run_id": "abc"})()})()

    def log_batch(self, run_id, metrics=(), params=(), tags=()):  # noqa: ANN001, ANN201
        self._check()
        self.log.append(("batch", len(metrics), len(params), len(tags)))

    def log_artifact(self, run_id, path, artifact_path=None):  # noqa: ANN001, ANN201
        self._check()
        self.log.append(("artifact", Path(path).name))

    def log_artifacts(self, run_id, path, artifact_path=None):  # noqa: ANN001, ANN201
        self._check()

    def set_terminated(self, run_id, status):  # noqa: ANN001, ANN201
        self._check()
        self.log.append(("end", status))


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    from qcclass.seguimiento import mlflow_cliente, registro

    log, state = [], {"down": False}

    def check(uri, timeout=5):  # noqa: ANN001, ANN202
        if state["down"]:
            raise ServerConnectionError("down")
        return "3.18.0"

    monkeypatch.setenv("QCCLASS_MLFLOW_URI", "http://127.0.0.1:1")
    monkeypatch.setattr(mlflow_cliente, "check_mlflow", check)
    monkeypatch.setattr(registro.ResilientRun, "_client", lambda self: FakeClient(log, state))
    return log, state


def test_resilient_run_queues_while_down_and_resumes(fake_mlflow, tmp_path: Path) -> None:
    from qcclass.seguimiento.registro import PENDING_FILE, ResilientRun

    log, state = fake_mlflow
    run = ResilientRun(
        experiment="e", run_name="r", tags={"equipo": "x"}, spool=tmp_path, backoff=QUICK
    )
    run.log_params({"a": 1})
    state["down"] = True
    run.log_metrics({"perdida": 1.0}, step=1)
    assert run.flush(block=False) is False and run.run_id is None
    assert len(json.loads((tmp_path / PENDING_FILE).read_text())) == 3
    # the process stops; a new one with the same spool continues
    again = ResilientRun(
        experiment="e", run_name="r", tags={"equipo": "x"}, spool=tmp_path, backoff=QUICK
    )
    assert again.pending == 3
    state["down"] = False
    again.log_dict({"k": 1}, "configuracion.json")
    assert again.flush(block=False) is True and again.run_id == "abc"
    assert log[0] == ("create", "r", {"mlflow.runName": "r", "equipo": "x"})
    assert ("artifact", "configuracion.json") in log
    state["down"] = True
    again.log_metrics({"perdida": 0.5}, step=2)
    with pytest.raises(ServerConnectionError, match="still unreachable"):
        again.finish()
    state["down"] = False
    ResilientRun(experiment="e", run_name="r", tags={}, spool=tmp_path, backoff=QUICK).flush(
        block=True
    )
    assert log[-1] == ("end", "FINISHED")
