"""End-to-end run of ``qcclass todo`` on the reduced configuration (slow)."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from qcclass.cli import app
from qcclass.config import load_dataset_config, load_splits_config
from qcclass.data.calidad import DATASET_REPORT_NAME, write_dataset_report
from qcclass.data.particiones import PROTOCOLS
from qcclass.metadatos import METADATA_COLUMNS, read_table


def _mini_configs(repo_root: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Copy the mini configurations so that their data directories live under ``tmp_path``."""
    folder = tmp_path / "configs"
    folder.mkdir()
    for name in ("dataset_mini.yaml", "splits_mini.yaml"):
        (folder / name).write_text((repo_root / "configs" / name).read_text(encoding="utf-8"))
    data = yaml.safe_load((folder / "dataset_mini.yaml").read_text(encoding="utf-8"))
    assert data["rutas"]["raiz"] == ".."  # resolves to tmp_path
    return folder / "dataset_mini.yaml", folder / "splits_mini.yaml"


@pytest.mark.slow
def test_whole_pipeline_on_mini_configuration(repo_root: Path, tmp_path: Path) -> None:
    dataset_yaml, splits_yaml = _mini_configs(repo_root, tmp_path)
    args = ["todo", "--config", str(dataset_yaml), "--splits", str(splits_yaml), "--procesos", "2"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Coherence checks: 22/22 ok" in result.output

    config = load_dataset_config(dataset_yaml)
    report = (config.rutas.resultados / DATASET_REPORT_NAME).read_text(encoding="utf-8")
    for section in ("antes y después de deduplicar", "Cúbits y profundidad", "Coherencia"):
        assert section in report
    assert (config.rutas.resultados / "figuras" / "distribucion_cubits_profundidad.png").is_file()
    for protocol in PROTOCOLS:
        assert (config.rutas.splits / f"{protocol}.csv").is_file()
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    assert set(metadata["algoritmo"]) == {"dj", "grover", "qft", "ghz", "randomcircuit"}

    # a second run is incremental: nothing new is generated, everything stays coherent
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 0, again.output
    assert "Circuits written: 0" in again.output
    assert "Oracle circuits written: 0" in again.output
    assert "Canonical circuits written: 0" in again.output

    # the coherence check notices a missing file
    victim = config.rutas.raiz / metadata.loc[metadata["variante_comp"] == "", "ruta_raw"].iloc[0]
    victim.with_suffix(".qpy").unlink()
    checks = write_dataset_report(config, load_splits_config(splits_yaml, dataset=config)).checks
    failing = [check.name for check in checks if not check.ok]
    assert failing == ["raw: filas sin QASM o QPY"]


@pytest.mark.slow
def test_measurement_on_the_mini_dataset(repo_root: Path, tmp_path: Path) -> None:
    """``qcclass medir`` on CPU (1 epoch) over the real reduced dataset writes its JSON."""
    import json

    import pandas as pd

    dataset_yaml, splits_yaml = _mini_configs(repo_root, tmp_path)
    runner = CliRunner()
    args = ["todo", "--config", str(dataset_yaml), "--splits", str(splits_yaml), "--procesos", "2"]
    assert runner.invoke(app, args).exit_code == 0
    config = load_dataset_config(dataset_yaml)
    index = pd.read_csv(config.rutas.graphs / "dag" / "indice.csv")
    assert list(index.columns) == ["id", "n_nodos"]  # written by the formats step
    versioned = runner.invoke(app, ["versionar", "--config", str(dataset_yaml), "--nombre", "m"])
    assert versioned.exit_code == 0, versioned.output
    measure = ["medir", "--config", str(dataset_yaml), "--dispositivo", "cpu", "--epocas", "1"]
    result = runner.invoke(app, [*measure, "--equipo", "ci"])
    assert result.exit_code == 0, result.output
    stored = json.loads((config.rutas.resultados / "medicion" / "ci_cpu.json").read_text())
    assert stored["version_dataset"] == "m" and stored["epocas"][0]["lotes"] > 0
    assert stored["proyeccion"]["fuente"] == "epoca_1"
    assert not (config.rutas.resultados / "accesos_test.log").exists()


SERVER_VARIABLES = (
    "QCCLASS_MLFLOW_URI",
    "QCCLASS_OPTUNA_STORAGE",
    "QCCLASS_EQUIPO",
    "QCCLASS_SERVIDORES_DIR",
    "QCCLASS_SERVIDORES_HOST",
    "QCCLASS_MLFLOW_PUERTO",
    "QCCLASS_OPTUNA_PUERTO",
)

MINI_BASELINE = """
nombre: lb_mini
experimento_mlflow: lb_mini
intentos: 2
semilla: 1
poda: {intentos_iniciales: 10}
objetivos: [algoritmo, familia]
peldanos:
  e7: {fuente: f3, columnas: [n_qubits, profundidad, n_puertas]}
  f3: {fuente: f3, columnas: null}
  wl:
    fuente: wl
    h_max: 2
    bits_hash: 16
    divisiones_pi: 16
    tolerancia_angulo: 1.0e-4
    espacio:
      h: {tipo: int, min: 1, max: 2}
      min_circuitos: {tipo: categorica, valores: [1, 2]}
clasificadores:
  regresion_logistica:
    C: {tipo: float, min: 0.1, max: 10, log: true}
    max_iter: {tipo: fijo, valor: 200}
  random_forest:
    n_estimators: {tipo: int, min: 10, max: 20, paso: 10}
  hist_gradient_boosting:
    max_iter: {tipo: int, min: 10, max: 20, paso: 10}
ajustes_por_peldano:
  wl:
    hist_gradient_boosting:
      columnas_wl: {tipo: categorica, valores: [16]}
importancia: {repeticiones: 2, max_columnas_wl: 5}
"""


@pytest.mark.slow
def test_baseline_on_the_mini_dataset(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``qcclass linea-base`` on the reduced dataset, with real MLflow and Optuna servers."""
    import socket

    from qcclass.lineabase.resumen import SUMMARY_FILE

    def free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    for name in SERVER_VARIABLES:  # .env values of an earlier test stay in os.environ
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QCCLASS_REINTENTO_TOTAL", "60")  # real servers may start slowly
    dataset_yaml, splits_yaml = _mini_configs(repo_root, tmp_path)
    config_args = ["--config", str(dataset_yaml)]
    runner = CliRunner()
    assert runner.invoke(app, ["todo", *config_args, "--splits", str(splits_yaml)]).exit_code == 0
    assert runner.invoke(app, ["versionar", *config_args, "--nombre", "m"]).exit_code == 0
    mlflow_port, optuna_port = free_port(), free_port()
    (tmp_path / ".env").write_text(
        f"QCCLASS_EQUIPO=ci\nQCCLASS_SERVIDORES_DIR={tmp_path / 'servidores'}\n"
        f"QCCLASS_SERVIDORES_HOST=127.0.0.1\nQCCLASS_MLFLOW_PUERTO={mlflow_port}\n"
        f"QCCLASS_OPTUNA_PUERTO={optuna_port}\n"
        f"QCCLASS_MLFLOW_URI=http://127.0.0.1:{mlflow_port}\n"
        f"QCCLASS_OPTUNA_STORAGE=grpc://127.0.0.1:{optuna_port}\n"
    )
    experiment = tmp_path / "lb_mini.yaml"
    experiment.write_text(MINI_BASELINE)
    try:
        started = runner.invoke(app, ["servidores", "arrancar", *config_args])
        assert started.exit_code == 0, started.output
        args = ["linea-base", *config_args, "--splits", str(splits_yaml)]
        result = runner.invoke(app, [*args, "--experimento", str(experiment)])
        assert result.exit_code == 0, result.output
        assert result.output.count("CV macro F1") == 18
        config = load_dataset_config(dataset_yaml)
        summary = (config.rutas.resultados / "baseline" / SUMMARY_FILE).read_text()
        assert "## Objetivo: algoritmo" in summary and "| e7 |" in summary
        assert "**e7**" in summary and "Importancia por permutación: wl" in summary
        assert not (config.rutas.resultados / "accesos_test.log").exists()
        again = runner.invoke(app, [*args, "--experimento", str(experiment), "--peldanos", "e7"])
        assert again.exit_code == 0 and again.output.count("CV macro F1") == 6  # resumed
    finally:
        runner.invoke(app, ["servidores", "parar", *config_args])


@pytest.mark.slow
def test_gnn_on_the_mini_dataset(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``qcclass entrenar-gnn`` on CPU (2 epochs) with a real MLflow server; launching it
    again finds the finished block."""
    import json
    import socket

    def free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    for name in SERVER_VARIABLES:  # .env values of an earlier test stay in os.environ
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QCCLASS_REINTENTO_TOTAL", "60")  # real servers may start slowly
    dataset_yaml, splits_yaml = _mini_configs(repo_root, tmp_path)
    config_args = ["--config", str(dataset_yaml)]
    runner = CliRunner()
    assert runner.invoke(app, ["todo", *config_args, "--splits", str(splits_yaml)]).exit_code == 0
    assert runner.invoke(app, ["versionar", *config_args, "--nombre", "m"]).exit_code == 0
    mlflow_port, optuna_port = free_port(), free_port()
    (tmp_path / ".env").write_text(
        f"QCCLASS_EQUIPO=ci\nQCCLASS_SERVIDORES_DIR={tmp_path / 'servidores'}\n"
        f"QCCLASS_SERVIDORES_HOST=127.0.0.1\nQCCLASS_MLFLOW_PUERTO={mlflow_port}\n"
        f"QCCLASS_OPTUNA_PUERTO={optuna_port}\n"
        f"QCCLASS_MLFLOW_URI=http://127.0.0.1:{mlflow_port}\n"
        f"QCCLASS_OPTUNA_STORAGE=grpc://127.0.0.1:{optuna_port}\n"
    )
    args = [
        "entrenar-gnn",
        *config_args,
        "--splits",
        str(splits_yaml),
        "--dispositivo",
        "cpu",
        "--epocas",
        "2",
        "--procesos",
        "0",
        "--bloque",
        "1",
    ]
    try:
        started = runner.invoke(app, ["servidores", "arrancar", *config_args])
        assert started.exit_code == 0, started.output
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Block 1: algorithm macro F1" in result.output and "2 epochs" in result.output
        directory = Path(result.output.split("Results: ")[1].strip())
        stored = json.loads((directory / "resultado.json").read_text())
        assert len(stored["historial"]) == 2 and stored["identidad"]["bloque"] == 1
        assert not json.loads((directory / "mlflow" / "pendientes.json").read_text())
        again = runner.invoke(app, args)
        assert again.exit_code == 0 and str(directory) in again.output
        config = load_dataset_config(dataset_yaml)
        assert not (config.rutas.resultados / "accesos_test.log").exists()
    finally:
        runner.invoke(app, ["servidores", "parar", *config_args])
