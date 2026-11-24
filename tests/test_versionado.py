"""Tests for the dataset version and the protection of the test sets."""

import importlib.metadata
import json
import logging
import platform
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qcclass import versionado
from qcclass.cli import app
from qcclass.config import DatasetConfig
from qcclass.versionado import (
    FinalEvaluationPermit,
    LockedTestSetError,
    VersionError,
    access_log_path,
    begin_experiment,
    current_hashes,
    file_sha256,
    final_evaluation_permit,
    global_fingerprint,
    library_differences,
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
    assert set(stored) == {"nombre", "huella", "archivos", "creado", "git_commit", "librerias"}


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


@pytest.fixture
def permit(versioned: DatasetConfig, monkeypatch: pytest.MonkeyPatch) -> FinalEvaluationPermit:
    """A final-evaluation permit, as if the pre-registration tag existed."""
    monkeypatch.setattr(versionado, "_tag_exists", lambda root, tag: True)
    return final_evaluation_permit(versioned, "evaluar-final")


@pytest.mark.parametrize("protected", ["test", "desconocido"])
def test_protected_sets_are_locked_without_a_permit(
    versioned: DatasetConfig, protected: str
) -> None:
    with pytest.raises(LockedTestSetError, match="evaluar-final"):
        split_ids(versioned, "aleatorio", ["val", protected], experiment="exp-1")
    assert not access_log_path(versioned).exists()


def test_permit_only_for_the_final_command_after_preregistration(
    versioned: DatasetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(versionado, "_tag_exists", lambda root, tag: False)
    with pytest.raises(LockedTestSetError, match="preregistro-v1"):
        final_evaluation_permit(versioned, "evaluar-final")
    monkeypatch.setattr(versionado, "_tag_exists", lambda root, tag: True)
    with pytest.raises(LockedTestSetError, match="only"):
        final_evaluation_permit(versioned, "entrenar")
    forged = FinalEvaluationPermit("evaluar-final", "preregistro-v1", object())
    with pytest.raises(LockedTestSetError):
        split_ids(versioned, "aleatorio", ["test"], experiment="exp-1", permit=forged)


def test_allowed_access_is_logged(versioned: DatasetConfig, permit: FinalEvaluationPermit) -> None:
    rows = split_ids(versioned, "aleatorio", ["test"], experiment="exp-1", permit=permit)
    assert list(rows["id"]) == ["c"]
    split_ids(versioned, "por_tamano", ["val", "test"], experiment="exp-2", permit=permit)
    lines = access_log_path(versioned).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    date, experiment, version, fingerprint, protocol, sets = lines[0].split("\t")
    assert (experiment, version, protocol, sets) == ("exp-1", "dataset-test", "aleatorio", "test")
    assert fingerprint == read_version(versioned).huella[:12]
    assert date.startswith("20")
    assert lines[1].split("\t")[1] == "exp-2"


def test_no_access_on_modified_data(
    versioned: DatasetConfig, permit: FinalEvaluationPermit
) -> None:
    versioned.rutas.metadata.write_text("id\n", encoding="utf-8")
    with pytest.raises(VersionError):
        split_ids(versioned, "aleatorio", ["train"], experiment="exp-1")
    with pytest.raises(VersionError):
        split_ids(versioned, "aleatorio", ["test"], experiment="exp-1", permit=permit)
    assert not access_log_path(versioned).exists()


def test_cli_versionar(versioned: DatasetConfig, tmp_path: Path) -> None:
    config_path = str(tmp_path / "configs" / "dataset.yaml")
    ok = CliRunner().invoke(app, ["versionar", "--config", config_path, "--comprobar"])
    assert ok.exit_code == 0 and "matches dataset-test" in ok.output
    versioned.rutas.metadata.write_text("id\n", encoding="utf-8")
    bad = CliRunner().invoke(app, ["versionar", "--config", config_path, "--comprobar"])
    assert bad.exit_code == 1 and "modificado data/metadata.csv" in bad.output


def test_version_records_python_and_libraries(versioned: DatasetConfig) -> None:
    version = read_version(versioned)
    assert version.librerias["python"] == platform.python_version()
    for name in ("qiskit", "mqt.bench", "torch", "torch-geometric", "numpy", "scikit-learn"):
        assert version.librerias[name] == importlib.metadata.version(name)
    assert "optuna" in version.librerias
    assert library_differences(versioned) == []


def test_begin_experiment_compares_fingerprints(versioned: DatasetConfig) -> None:
    stamp = begin_experiment(versioned, "exp-1")
    assert stamp == version_stamp(versioned)
    (versioned.rutas.splits / "aleatorio.csv").write_text("id,conjunto\n", encoding="utf-8")
    with pytest.raises(VersionError, match="modificado data/splits/aleatorio.csv"):
        begin_experiment(versioned, "exp-1")


def test_hand_edited_version_file_is_detected(versioned: DatasetConfig) -> None:
    stored = json.loads(versioned.rutas.version.read_text(encoding="utf-8"))
    stored["huella"] = "0" * 64
    versioned.rutas.version.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(VersionError, match="incoherente"):
        begin_experiment(versioned, "exp-1")


def test_other_library_version_only_warns(
    versioned: DatasetConfig, caplog: pytest.LogCaptureFixture
) -> None:
    stored = json.loads(versioned.rutas.version.read_text(encoding="utf-8"))
    stored["librerias"]["qiskit"] = "0.0.1"
    versioned.rutas.version.write_text(json.dumps(stored), encoding="utf-8")
    installed = importlib.metadata.version("qiskit")
    assert library_differences(versioned) == [f"qiskit: dataset 0.0.1, instalada {installed}"]
    with caplog.at_level(logging.WARNING):
        begin_experiment(versioned, "exp-1")
    assert "library version differs" in caplog.text
