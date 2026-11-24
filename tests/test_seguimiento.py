"""Tests for the training utilities and the experiment-tracking clients (Phase 9)."""

import random
import socket
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from qcclass.cli import app
from qcclass.config import DatasetConfig
from qcclass.entrenamiento.puntos_guardado import CheckpointManager
from qcclass.entrenamiento.reproducibilidad import set_global_seeds
from qcclass.seguimiento import servidores
from qcclass.seguimiento.entorno import ServerConnectionError, load_env, machine, required
from qcclass.seguimiento.mlflow_cliente import check_mlflow, flatten, tracking_uri
from qcclass.seguimiento.optuna_cliente import connect, make_storage
from qcclass.versionado import write_version

ConfigFactory = Callable[..., DatasetConfig]
CLIENT_VARS = ("QCCLASS_MLFLOW_URI", "QCCLASS_OPTUNA_STORAGE", "QCCLASS_EQUIPO")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never pick up the developer's own .env or variables."""
    for name in (*CLIENT_VARS, "QCCLASS_SERVIDORES_DIR", "QCCLASS_MLFLOW_PUERTO"):
        monkeypatch.delenv(name, raising=False)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --------------------------------------------------------------------------------------------
# Settings and refusals
# --------------------------------------------------------------------------------------------


def test_env_file_does_not_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("QCCLASS_EQUIPO=desde_env\nQCCLASS_MLFLOW_URI=http://x:1\n")
    monkeypatch.setenv("QCCLASS_EQUIPO", "desde_entorno")
    assert load_env(tmp_path) == tmp_path / ".env"
    assert machine("host") == "desde_entorno"
    assert required("QCCLASS_MLFLOW_URI") == "http://x:1"
    monkeypatch.delenv("QCCLASS_MLFLOW_URI")
    assert load_env(tmp_path / "nada") is None


def test_missing_or_local_settings_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ServerConnectionError, match="EQUIPOS.md"):
        tracking_uri()
    monkeypatch.setenv("QCCLASS_MLFLOW_URI", "file:///tmp/mlruns")
    with pytest.raises(ServerConnectionError, match="local tracking"):
        tracking_uri()
    for uri in ("sqlite:///x.db", "inmemory", "grpc://sinpuerto"):
        with pytest.raises(ServerConnectionError):
            make_storage(uri)


def test_unreachable_servers_abort_quickly() -> None:
    port = _free_port()
    with pytest.raises(ServerConnectionError, match="cannot reach MLflow"):
        check_mlflow(f"http://127.0.0.1:{port}", timeout=1)
    with pytest.raises(ServerConnectionError, match="cannot reach the Optuna storage"):
        connect(f"grpc://127.0.0.1:{port}", timeout=1)


def test_ping_reports_failures(make_config: ConfigFactory, tmp_path: Path) -> None:
    make_config()
    port = _free_port()
    (tmp_path / ".env").write_text(
        f"QCCLASS_EQUIPO=pruebas\nQCCLASS_MLFLOW_URI=http://127.0.0.1:{port}\n"
        f"QCCLASS_OPTUNA_STORAGE=grpc://127.0.0.1:{port}\n"
    )
    result = CliRunner().invoke(app, ["ping", "--config", str(tmp_path / "configs/dataset.yaml")])
    assert result.exit_code == 1
    assert "Machine: pruebas" in result.output
    assert "MLflow: FAILED" in result.output and "Optuna: FAILED" in result.output


def test_flatten_configuration() -> None:
    flat = flatten({"a": 1, "b": {"c": [1, 2], "d": {"e": "x"}}})
    assert flat == {"a": "1", "b.c": "[1, 2]", "b.d.e": "x"}


def test_server_commands(tmp_path: Path) -> None:
    settings = servidores.ServerSettings(tmp_path, "0.0.0.0", 5050, 13000)
    commands = servidores.commands(settings)
    mlflow = commands["mlflow"]
    assert "--serve-artifacts" in mlflow and mlflow[mlflow.index("--port") + 1] == "5050"
    assert mlflow[mlflow.index("--backend-store-uri") + 1].startswith("sqlite:///")
    assert "qcclass.seguimiento.proxy_optuna" in commands["optuna"]
    assert servidores.running_pid(settings, "mlflow") is None
    settings.pid_file("mlflow").parent.mkdir(parents=True)
    settings.pid_file("mlflow").write_text("999999")  # stale pid file
    assert servidores.running_pid(settings, "mlflow") is None
    assert not settings.pid_file("mlflow").exists()


# --------------------------------------------------------------------------------------------
# Seeds and checkpoints
# --------------------------------------------------------------------------------------------


def _train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: range) -> None:
    data = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
    target = data.sum(dim=1, keepdim=True)
    for _ in epochs:
        order = torch.randperm(64)  # global generator: must be restored on resume
        noise = np.random.rand() + random.random()  # numpy and Python generators too
        for chunk in order.split(16):
            optimizer.zero_grad()
            prediction = model(data[chunk]) * (1 + 0.01 * noise)
            torch.nn.functional.mse_loss(prediction, target[chunk]).backward()
            optimizer.step()


def _fresh() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
    return model, torch.optim.AdamW(model.parameters(), lr=0.01)


def test_resuming_from_a_checkpoint_matches_continuous_training(tmp_path: Path) -> None:
    set_global_seeds(7)
    model, optimizer = _fresh()
    _train(model, optimizer, range(6))
    continuous = [p.detach().clone() for p in model.parameters()]

    set_global_seeds(7)
    model, optimizer = _fresh()
    manager = CheckpointManager(tmp_path / "puntos", keep=2)
    _train(model, optimizer, range(3))
    manager.save(3, {"modelo": model.state_dict(), "optimizador": optimizer.state_dict()})
    torch.manual_seed(999)  # the cut: generators move on before the restart

    model, optimizer = _fresh()
    epoch, states = manager.load_latest()
    model.load_state_dict(states["modelo"])
    optimizer.load_state_dict(states["optimizador"])
    _train(model, optimizer, range(epoch, 6))
    for resumed, reference in zip(model.parameters(), continuous, strict=True):
        assert torch.allclose(resumed, reference, atol=1e-6)


def test_checkpoints_are_pruned(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep=2)
    assert manager.load_latest() is None
    for epoch in range(1, 5):
        manager.save(epoch, {"x": torch.tensor([epoch])})
    assert manager.epochs() == [3, 4]
    epoch, states = manager.load_latest()
    assert epoch == 4 and states["x"].item() == 4


# --------------------------------------------------------------------------------------------
# Real servers (slow): start, ping, two machines in one study, stop
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_servers_end_to_end(
    make_config: ConfigFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config()
    config.rutas.metadata.parent.mkdir(parents=True, exist_ok=True)
    config.rutas.metadata.write_text("id,algoritmo\na,ghz\n", encoding="utf-8")
    config.rutas.splits.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["a"], "conjunto": ["train"], "solo_evaluacion": [False]}).to_csv(
        config.rutas.splits / "aleatorio.csv", index=False
    )
    config.rutas.graphs.mkdir(parents=True, exist_ok=True)
    (config.rutas.graphs / "etiquetas.json").write_text("{}", encoding="utf-8")
    write_version(config, "dataset-prueba")
    mlflow_port, optuna_port = _free_port(), _free_port()
    (tmp_path / ".env").write_text(
        f"QCCLASS_SERVIDORES_DIR={tmp_path / 'servidores'}\nQCCLASS_SERVIDORES_HOST=127.0.0.1\n"
        f"QCCLASS_MLFLOW_PUERTO={mlflow_port}\nQCCLASS_OPTUNA_PUERTO={optuna_port}\n"
        f"QCCLASS_MLFLOW_URI=http://127.0.0.1:{mlflow_port}\n"
        f"QCCLASS_OPTUNA_STORAGE=grpc://127.0.0.1:{optuna_port}\n"
    )
    experiment = tmp_path / "prueba.yaml"
    experiment.write_text(
        "experimento_mlflow: e\nestudio_optuna: s\nintentos: 2\nsemilla: 1\n", encoding="utf-8"
    )
    config_path = str(tmp_path / "configs" / "dataset.yaml")
    runner = CliRunner()
    for name in ("QCCLASS_SERVIDORES_DIR", "QCCLASS_MLFLOW_PUERTO", "QCCLASS_OPTUNA_PUERTO"):
        monkeypatch.delenv(name, raising=False)
    try:
        started = runner.invoke(app, ["servidores", "arrancar", "--config", config_path])
        assert started.exit_code == 0, started.output
        assert runner.invoke(app, ["ping", "--config", config_path]).exit_code == 0
        for name in ("m3pro", "rtx3050"):
            monkeypatch.setenv("QCCLASS_EQUIPO", name)
            args = ["prueba-servidores", "--config", config_path, "--experimento", str(experiment)]
            result = runner.invoke(app, args)
            assert result.exit_code == 0, result.output
        assert "m3pro: 2 complete trials" in result.output
        assert "rtx3050: 2 complete trials" in result.output
    finally:
        stopped = runner.invoke(app, ["servidores", "parar", "--config", config_path])
    assert "mlflow" in stopped.output
    status = runner.invoke(app, ["servidores", "estado", "--config", config_path]).output
    assert "mlflow: stopped" in status and "optuna: stopped" in status
