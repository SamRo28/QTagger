"""Shared pytest fixtures."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from qcclass.config import DatasetConfig, load_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[1]

ConfigFactory = Callable[..., DatasetConfig]


@pytest.fixture
def repo_root() -> Path:
    """Repository root directory."""
    return REPO_ROOT


@pytest.fixture
def dataset_yaml() -> Path:
    """Path to the project's dataset configuration."""
    return REPO_ROOT / "configs" / "dataset.yaml"


@pytest.fixture
def splits_yaml() -> Path:
    """Path to the project's split configuration."""
    return REPO_ROOT / "configs" / "splits.yaml"


@pytest.fixture
def make_config(tmp_path: Path, dataset_yaml: Path) -> ConfigFactory:
    """Factory of dataset configurations whose data directories live under ``tmp_path``.

    The project's ``dataset.yaml`` is copied to ``tmp_path/configs/`` (so ``raiz: ..``
    resolves to ``tmp_path``) after applying the given top-level overrides.
    """

    def factory(**overrides: Any) -> DatasetConfig:
        data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        data.update(overrides)
        path = tmp_path / "configs" / "dataset.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return load_dataset_config(path)

    return factory
