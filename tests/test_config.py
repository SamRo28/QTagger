"""Tests for loading and validating the YAML configuration."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from mqt.bench import BenchmarkLevel
from mqt.bench.benchmarks import get_available_benchmark_names

from qcclass.config import ConfigError, load_dataset_config, load_splits_config

Mutation = Callable[[dict[str, Any]], None]


def _read(path: Path) -> dict[str, Any]:
    """Read a YAML file into a plain dictionary."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(directory: Path, name: str, data: dict[str, Any]) -> Path:
    """Write a dictionary as YAML and return the file path."""
    path = directory / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------
# The shipped configuration files
# --------------------------------------------------------------------------------------------


def test_dataset_config_loads(dataset_yaml: Path, repo_root: Path) -> None:
    config = load_dataset_config(dataset_yaml)

    assert config.semilla_global == 42
    assert (config.tamanos.min, config.tamanos.max) == (2, 40)
    assert config.niveles == ("alg", "indep", "nativegates", "mapped")
    assert config.benchmark_levels == tuple(BenchmarkLevel)
    assert config.nativegates.gateset == "ibm_heron"
    assert config.mapped.dispositivo == "ibm_heron_156"
    assert config.opt_level == 2
    assert config.niveles_opt_compilacion == (1, 0, 2)
    assert config.variantes_parametros == 5
    assert config.max_nodos == 20000
    assert len(config.familias) == 6
    assert len(config.all_algorithms) == 31
    assert config.reservados_sin_familia == ("qwalk",)
    assert config.oraculos.por_tamano == 20
    assert config.etiquetas.fusiones == {"full_adder": "cdkm_ripple_carry_adder", "bell": "ghz"}
    assert config.etiquetas.clase_propia_con_minimo == ("vbe_ripple_carry_adder",)
    assert len(config.etiquetas.solo_familia) == 6
    assert config.etiquetas.minimo_bases_clase_propia == 22
    assert config.family_of("grover") == "oraculos"
    assert config.control_negativo == ("randomcircuit",)
    assert set(config.excluidos) == {
        "dynamic_qft",
        "ghz_dynamic",
        "iqpe",
        "mlqae",
        "quantum_metropolis",
        "seven_qubit_steane_code",
        "shors_nine_qubit_code",
        "teleportation",
        "shor",
    }
    assert config.rutas.raiz == repo_root
    assert config.rutas.raw == repo_root / "data" / "raw"
    assert config.rutas.metadata == repo_root / "data" / "metadata.csv"
    assert config.rutas.resultados == repo_root / "results"


def test_dataset_config_covers_every_installed_benchmark(dataset_yaml: Path) -> None:
    """Classes, negative control and exclusions account for the whole mqt.bench catalogue."""
    config = load_dataset_config(dataset_yaml)
    assert set(config.referenced_benchmarks) == set(get_available_benchmark_names())


def test_splits_config_loads(splits_yaml: Path, dataset_yaml: Path) -> None:
    splits = load_splits_config(splits_yaml, dataset=load_dataset_config(dataset_yaml))

    assert splits.semilla == 42
    assert (splits.aleatorio.train, splits.aleatorio.val, splits.aleatorio.test) == pytest.approx(
        (0.70, 0.15, 0.15)
    )
    assert (splits.por_tamano.train.min, splits.por_tamano.train.max) == (3, 16)
    assert (splits.por_tamano.val.min, splits.por_tamano.val.max) == (17, 20)
    assert (splits.por_tamano.test.min, splits.por_tamano.test.max) == (21, 30)
    assert splits.algoritmos_reservados == ("qwalk", "hrs_cumulative_multiplier", "hhl")
    assert splits.minimo_por_clase_train == 15
    assert splits.balanceo.modo == "muestreo"
    assert splits.balanceo.max_bases_por_clase is None
    assert splits.por_nivel.train == ("alg", "indep")
    assert splits.por_nivel.test == ("nativegates", "mapped")


# --------------------------------------------------------------------------------------------
# Invalid dataset configurations
# --------------------------------------------------------------------------------------------


def test_unknown_benchmark_fails_and_warns(
    dataset_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _read(dataset_yaml)
    data["familias"]["oraculos"].append("not_a_benchmark")
    path = _write(tmp_path, "dataset.yaml", data)

    with caplog.at_level(logging.WARNING), pytest.raises(ConfigError, match="not_a_benchmark"):
        load_dataset_config(path)
    assert "not_a_benchmark" in caplog.text


def _set_gateset(data: dict[str, Any]) -> None:
    data["nativegates"]["gateset"] = "ibm_falcon"


def _set_device(data: dict[str, Any]) -> None:
    data["mapped"]["dispositivo"] = "ibm_falcon_27"


def _add_unknown_level(data: dict[str, Any]) -> None:
    data["niveles"].append("hardware")


def _repeat_algorithm_in_two_families(data: dict[str, Any]) -> None:
    data["familias"]["qft_fase"].append("grover")


def _unknown_variational(data: dict[str, Any]) -> None:
    data["variacionales_con_parametros"].append("ghz_not_a_class")


def _excluded_also_class(data: dict[str, Any]) -> None:
    data["excluidos"].append("ghz")


def _drop_key(data: dict[str, Any]) -> None:
    del data["max_nodos"]


def _invert_sizes(data: dict[str, Any]) -> None:
    data["tamanos"] = {"min": 10, "max": 3}


def _bad_opt_level(data: dict[str, Any]) -> None:
    data["opt_level"] = 7


def _boolean_seed(data: dict[str, Any]) -> None:
    data["semilla_global"] = True


def _merge_into_unknown(data: dict[str, Any]) -> None:
    data["etiquetas"]["fusiones"]["bell"] = "not_a_class"


def _chained_merge(data: dict[str, Any]) -> None:
    data["etiquetas"]["fusiones"]["ghz"] = "wstate"


def _unfamiliar_also_class(data: dict[str, Any]) -> None:
    data["reservados_sin_familia"].append("ghz")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_gateset, "ibm_falcon"),
        (_set_device, "ibm_falcon_27"),
        (_add_unknown_level, "hardware"),
        (_repeat_algorithm_in_two_families, "more than one family"),
        (_unknown_variational, "ghz_not_a_class"),
        (_excluded_also_class, "excluidos"),
        (_drop_key, "max_nodos"),
        (_invert_sizes, "invalid size range"),
        (_bad_opt_level, "opt_level"),
        (_boolean_seed, "semilla_global"),
        (_unfamiliar_also_class, "reservados_sin_familia"),
        (_merge_into_unknown, "not_a_class"),
        (_chained_merge, "merged classes"),
    ],
)
def test_invalid_dataset_config_fails(
    dataset_yaml: Path, tmp_path: Path, mutate: Mutation, message: str
) -> None:
    data = _read(dataset_yaml)
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        load_dataset_config(_write(tmp_path, "dataset.yaml", data))


# --------------------------------------------------------------------------------------------
# Invalid split configurations
# --------------------------------------------------------------------------------------------


def _fractions_not_summing_to_one(data: dict[str, Any]) -> None:
    data["aleatorio"]["train"] = 0.80


def _overlapping_size_ranges(data: dict[str, Any]) -> None:
    data["por_tamano"]["val"] = [16, 20]


def _unknown_reserved_algorithm(data: dict[str, Any]) -> None:
    data["algoritmos_reservados"].append("not_a_benchmark")


def _unknown_balancing_mode(data: dict[str, Any]) -> None:
    data["balanceo"]["modo"] = "smote"


def _zero_cap(data: dict[str, Any]) -> None:
    data["balanceo"]["max_bases_por_clase"] = 0


def _level_in_train_and_test(data: dict[str, Any]) -> None:
    data["por_nivel"]["test"].append("indep")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_fractions_not_summing_to_one, "add up to 1"),
        (_overlapping_size_ranges, "overlap"),
        (_unknown_reserved_algorithm, "not_a_benchmark"),
        (_unknown_balancing_mode, "modo"),
        (_zero_cap, "max_bases_por_clase"),
        (_level_in_train_and_test, "both train and test"),
    ],
)
def test_invalid_splits_config_fails(
    splits_yaml: Path, dataset_yaml: Path, tmp_path: Path, mutate: Mutation, message: str
) -> None:
    data = _read(splits_yaml)
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        load_splits_config(
            _write(tmp_path, "splits.yaml", data), dataset=load_dataset_config(dataset_yaml)
        )
