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
    assert "Coherence checks: 21/21 ok" in result.output

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
