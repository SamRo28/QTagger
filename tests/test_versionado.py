"""Tests for the dataset version and the protection of the test sets."""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qcclass.cli import app
from qcclass.config import DatasetConfig
from qcclass.versionado import (
    LockedTestSetError,
    VersionError,
    access_log_path,
    current_hashes,
    file_sha256,
    global_fingerprint,
    read_version,
    split_ids,
    version_differences,
    version_stamp,
    write_version,
)

ConfigFactory = Callable[..., DatasetConfig]  # the ``make_config`` fixture


@pytest.fixture
def versioned(make_config: ConfigFactory) -> DatasetConfig:
    """A tiny fake dataset (metadata, two splits, label vocabulary) with a version."""
    config = make_config()
    config.rutas.metadata.parent.mkdir(parents=True, exist_ok=True)
    config.rutas.metadata.write_text("id,algoritmo\na,ghz\nb,ghz\nc,qft\n", encoding="utf-8")
    config.rutas.splits.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {"id": ["a", "b", "c"], "conjunto": ["train", "val", "test"], "solo_evaluacion": False}
    )
    rows.to_csv(config.rutas.splits / "aleatorio.csv", index=False)
    rows.to_csv(config.rutas.splits / "por_tamano.csv", index=False)
    config.rutas.graphs.mkdir(parents=True, exist_ok=True)
    (config.rutas.graphs / "etiquetas.json").write_text('{"clases": ["ghz"]}', encoding="utf-8")
    write_version(config, "dataset-test")
    return config


def test_version_records_every_file(versioned: DatasetConfig) -> None:
    version = read_version(versioned)
    assert version.nombre == "dataset-test"
    assert set(version.archivos) == {
        "data/metadata.csv",
        "data/splits/aleatorio.csv",
        "data/splits/por_tamano.csv",
        "data/graphs/etiquetas.json",
    }
    assert version.archivos["data/metadata.csv"] == file_sha256(versioned.rutas.metadata)
    assert version.huella == global_fingerprint(current_hashes(versioned))
    assert version_differences(versioned) == []
    assert version_stamp(versioned) == {
        "version_dataset": "dataset-test",
        "huella_dataset": version.huella,
    }
    stored = json.loads(versioned.rutas.version.read_text(encoding="utf-8"))
    assert set(stored) == {"nombre", "huella", "archivos", "creado", "git_commit"}


def test_changes_are_detected(versioned: DatasetConfig) -> None:
    (versioned.rutas.splits / "aleatorio.csv").write_text("id,conjunto\n", encoding="utf-8")
    (versioned.rutas.splits / "nuevo.csv").write_text("id,conjunto\n", encoding="utf-8")
    (versioned.rutas.splits / "por_tamano.csv").unlink()
    assert version_differences(versioned) == [
        "falta data/splits/por_tamano.csv",
        "nuevo data/splits/nuevo.csv",
        "modificado data/splits/aleatorio.csv",
    ]
    with pytest.raises(VersionError, match="does not match"):
        version_stamp(versioned)


def test_incomplete_dataset_cannot_be_versioned(make_config: ConfigFactory) -> None:
    config = make_config()
    with pytest.raises(VersionError, match="incomplete"):
        write_version(config, "dataset-x")


def test_train_and_val_need_no_flag_and_leave_no_trace(versioned: DatasetConfig) -> None:
    rows = split_ids(versioned, "aleatorio", ["train", "val"], experiment="exp-1")
    assert list(rows["id"]) == ["a", "b"]
    assert not access_log_path(versioned).exists()


def test_test_set_is_locked_without_the_flag(versioned: DatasetConfig) -> None:
    with pytest.raises(LockedTestSetError, match="--evaluar-test"):
        split_ids(versioned, "aleatorio", ["test"], experiment="exp-1")
    assert not access_log_path(versioned).exists()


def test_allowed_test_access_is_logged(versioned: DatasetConfig) -> None:
    rows = split_ids(versioned, "aleatorio", ["test"], experiment="exp-1", evaluar_test=True)
    assert list(rows["id"]) == ["c"]
    split_ids(versioned, "por_tamano", ["val", "test"], experiment="exp-2", evaluar_test=True)
    lines = access_log_path(versioned).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    date, experiment, version, fingerprint, protocol = lines[0].split("\t")
    assert (experiment, version, protocol) == ("exp-1", "dataset-test", "aleatorio")
    assert fingerprint == read_version(versioned).huella[:12]
    assert date.startswith("20")
    assert lines[1].split("\t")[1] == "exp-2"


def test_no_access_on_modified_data(versioned: DatasetConfig) -> None:
    versioned.rutas.metadata.write_text("id\n", encoding="utf-8")
    with pytest.raises(VersionError):
        split_ids(versioned, "aleatorio", ["train"], experiment="exp-1")
    with pytest.raises(VersionError):
        split_ids(versioned, "aleatorio", ["test"], experiment="exp-1", evaluar_test=True)
    assert not access_log_path(versioned).exists()


def test_cli_versionar(versioned: DatasetConfig, tmp_path: Path) -> None:
    config_path = str(tmp_path / "configs" / "dataset.yaml")
    ok = CliRunner().invoke(app, ["versionar", "--config", config_path, "--comprobar"])
    assert ok.exit_code == 0 and "matches dataset-test" in ok.output
    versioned.rutas.metadata.write_text("id\n", encoding="utf-8")
    bad = CliRunner().invoke(app, ["versionar", "--config", config_path, "--comprobar"])
    assert bad.exit_code == 1 and "modificado data/metadata.csv" in bad.output
