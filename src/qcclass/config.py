"""Load and validate the YAML configuration files.

The YAML keys keep the names fixed by the project plan, and the dataclass attributes mirror
them one-to-one, so ``config.<key>`` always matches the file. Benchmark, level, gateset and
device names are checked against the installed ``mqt.bench`` instead of a hand-copied list.
"""

import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from mqt.bench import BenchmarkLevel
from mqt.bench.benchmarks import get_available_benchmark_names
from mqt.bench.targets import get_available_device_names, get_available_gateset_names

logger = logging.getLogger(__name__)

DATASET_KEYS = (
    "semilla_global",
    "tamanos",
    "niveles",
    "nativegates",
    "mapped",
    "opt_level",
    "niveles_opt_compilacion",
    "variantes_parametros",
    "max_nodos",
    "familias",
    "control_negativo",
    "reservados_sin_familia",
    "excluidos",
    "variacionales_con_parametros",
    "etiquetas",
    "oraculos",
    "rutas",
)
PATH_KEYS = (
    "raw",
    "canonical",
    "features",
    "graphs",
    "tokens",
    "gate_labels",
    "splits",
    "logs",
    "metadata",
    "resultados",
)
SPLITS_KEYS = (
    "semilla",
    "aleatorio",
    "por_tamano",
    "algoritmos_reservados",
    "minimo_por_clase_train",
    "balanceo",
    "por_nivel",
)
BALANCING_MODES = ("muestreo", "ponderacion")
SPLIT_SETS = ("train", "val", "test")
QISKIT_OPT_LEVELS = range(4)


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------------------------
# Configuration objects
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeRange:
    """Inclusive range of circuit sizes, in qubits."""

    min: int
    max: int


@dataclass(frozen=True)
class NativeGatesConfig:
    """Options for the ``nativegates`` level."""

    gateset: str


@dataclass(frozen=True)
class MappedConfig:
    """Options for the ``mapped`` level."""

    dispositivo: str


@dataclass(frozen=True)
class LabelsConfig:
    """How algorithms map to training class labels."""

    fusiones: Mapping[str, str]
    solo_familia: tuple[str, ...]
    clase_propia_con_minimo: tuple[str, ...]
    minimo_bases_clase_propia: int


@dataclass(frozen=True)
class OraclesConfig:
    """Options for the in-house oracle circuits."""

    por_tamano: int


@dataclass(frozen=True)
class PathsConfig:
    """Absolute dataset locations, resolved from the ``rutas`` section."""

    raiz: Path
    raw: Path
    canonical: Path
    features: Path
    graphs: Path
    tokens: Path
    gate_labels: Path
    splits: Path
    logs: Path
    metadata: Path
    resultados: Path


@dataclass(frozen=True)
class DatasetConfig:
    """Contents of ``configs/dataset.yaml``."""

    semilla_global: int
    tamanos: SizeRange
    niveles: tuple[str, ...]
    nativegates: NativeGatesConfig
    mapped: MappedConfig
    opt_level: int
    niveles_opt_compilacion: tuple[int, ...]
    variantes_parametros: int
    max_nodos: int
    familias: Mapping[str, tuple[str, ...]]
    control_negativo: tuple[str, ...]
    reservados_sin_familia: tuple[str, ...]
    excluidos: tuple[str, ...]
    variacionales_con_parametros: tuple[str, ...]
    etiquetas: LabelsConfig
    oraculos: OraclesConfig
    rutas: PathsConfig

    @property
    def all_algorithms(self) -> tuple[str, ...]:
        """Every class algorithm, in family order."""
        return tuple(name for names in self.familias.values() for name in names)

    @property
    def referenced_benchmarks(self) -> tuple[str, ...]:
        """Every benchmark name mentioned anywhere in the configuration, without repeats."""
        names = (
            *self.all_algorithms,
            *self.control_negativo,
            *self.reservados_sin_familia,
            *self.excluidos,
            *self.variacionales_con_parametros,
        )
        return tuple(dict.fromkeys(names))

    @property
    def benchmark_levels(self) -> tuple[BenchmarkLevel, ...]:
        """The configured levels as ``mqt.bench`` enum members."""
        return tuple(BenchmarkLevel[name.upper()] for name in self.niveles)

    def family_of(self, algorithm: str) -> str:
        """Return the family an algorithm belongs to; raise ``KeyError`` if it is not a class."""
        for family, names in self.familias.items():
            if algorithm in names:
                return family
        raise KeyError(f"'{algorithm}' is not a class algorithm of any family")


@dataclass(frozen=True)
class RandomSplitConfig:
    """Fractions of the ``aleatorio`` protocol."""

    train: float
    val: float
    test: float


@dataclass(frozen=True)
class SizeSplitConfig:
    """Qubit ranges of the ``por_tamano`` protocol."""

    train: SizeRange
    val: SizeRange
    test: SizeRange


@dataclass(frozen=True)
class LevelSplitConfig:
    """Levels used for training and testing in the ``por_nivel`` protocol."""

    train: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class BalancingConfig:
    """Class balancing applied when building the splits."""

    modo: str
    max_bases_por_clase: int | None


@dataclass(frozen=True)
class SplitsConfig:
    """Contents of ``configs/splits.yaml``."""

    semilla: int
    aleatorio: RandomSplitConfig
    por_tamano: SizeSplitConfig
    algoritmos_reservados: tuple[str, ...]
    minimo_por_clase_train: int
    balanceo: BalancingConfig
    por_nivel: LevelSplitConfig


# --------------------------------------------------------------------------------------------
# Low-level parsing helpers. ``ctx`` is a human-readable location such as
# ``dataset.yaml[tamanos][min]`` used in error messages.
# --------------------------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping."""
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return data


def _get(data: Mapping[str, Any], key: str, ctx: str) -> tuple[Any, str]:
    """Return ``data[key]`` together with its context, failing if the key is missing."""
    if key not in data:
        raise ConfigError(f"{ctx}: missing key '{key}'")
    return data[key], f"{ctx}[{key}]"


def _warn_unknown_keys(data: Mapping[str, Any], known: Iterable[str], ctx: str) -> None:
    """Log a warning for keys that are not part of the schema (usually typos)."""
    for key in sorted(set(data) - set(known)):
        logger.warning("%s: unknown key '%s' is ignored", ctx, key)


def _as_mapping(value: Any, ctx: str) -> Mapping[str, Any]:
    """Check that a value is a mapping."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"{ctx}: expected a mapping, got {value!r}")
    return value


def _as_int(value: Any, ctx: str, minimum: int | None = None) -> int:
    """Check that a value is an integer (booleans excluded), optionally bounded below."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{ctx}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{ctx}: must be >= {minimum}, got {value}")
    return value


def _as_optional_int(value: Any, ctx: str, minimum: int | None = None) -> int | None:
    """Like ``_as_int`` but ``null`` is allowed."""
    return None if value is None else _as_int(value, ctx, minimum)


def _as_fraction(value: Any, ctx: str) -> float:
    """Check that a value is a number in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{ctx}: expected a number, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{ctx}: must be in [0, 1], got {value}")
    return float(value)


def _as_str(value: Any, ctx: str) -> str:
    """Check that a value is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{ctx}: expected a non-empty string, got {value!r}")
    return value


def _as_str_tuple(value: Any, ctx: str, *, non_empty: bool = False) -> tuple[str, ...]:
    """Check that a value is a list of distinct non-empty strings."""
    if not isinstance(value, list):
        raise ConfigError(f"{ctx}: expected a list, got {value!r}")
    items = tuple(_as_str(item, f"{ctx}[{i}]") for i, item in enumerate(value))
    if non_empty and not items:
        raise ConfigError(f"{ctx}: must not be empty")
    repeated = sorted(name for name, count in Counter(items).items() if count > 1)
    if repeated:
        raise ConfigError(f"{ctx}: repeated entries {repeated}")
    return items


def _size_range(low: int, high: int, ctx: str) -> SizeRange:
    """Build a size range, checking ``1 <= low <= high``."""
    if low < 1 or low > high:
        raise ConfigError(f"{ctx}: invalid size range [{low}, {high}]")
    return SizeRange(min=low, max=high)


def _parse_size_mapping(value: Any, ctx: str) -> SizeRange:
    """Parse a size range written as ``{min: a, max: b}``."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, ("min", "max"), ctx)
    low = _as_int(*_get(data, "min", ctx))
    high = _as_int(*_get(data, "max", ctx))
    return _size_range(low, high, ctx)


def _parse_size_pair(value: Any, ctx: str) -> SizeRange:
    """Parse a size range written as ``[a, b]``."""
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{ctx}: expected a [min, max] pair, got {value!r}")
    low = _as_int(value[0], f"{ctx}[0]")
    high = _as_int(value[1], f"{ctx}[1]")
    return _size_range(low, high, ctx)


# --------------------------------------------------------------------------------------------
# dataset.yaml
# --------------------------------------------------------------------------------------------


def _parse_levels(value: Any, ctx: str) -> tuple[str, ...]:
    """Parse the level list, validating the names against ``mqt.bench.BenchmarkLevel``."""
    levels = _as_str_tuple(value, ctx, non_empty=True)
    valid = [member.name.lower() for member in BenchmarkLevel]
    unknown = [level for level in levels if level not in valid]
    if unknown:
        raise ConfigError(f"{ctx}: unknown levels {unknown}; valid levels are {valid}")
    return levels


def _parse_families(value: Any, ctx: str) -> dict[str, tuple[str, ...]]:
    """Parse the ``familias`` mapping of family name to benchmark names."""
    data = _as_mapping(value, ctx)
    if not data:
        raise ConfigError(f"{ctx}: at least one family is required")
    return {
        _as_str(family, ctx): _as_str_tuple(names, f"{ctx}[{family}]", non_empty=True)
        for family, names in data.items()
    }


def _parse_paths(value: Any, ctx: str, config_dir: Path) -> PathsConfig:
    """Parse ``rutas``: ``raiz`` is relative to the config file, the rest to ``raiz``."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, ("raiz", *PATH_KEYS), ctx)
    root = (config_dir / _as_str(*_get(data, "raiz", ctx))).resolve()
    entries = {key: root / _as_str(*_get(data, key, ctx)) for key in PATH_KEYS}
    return PathsConfig(raiz=root, **entries)


def _parse_opt_levels(value: Any, ctx: str) -> tuple[int, ...]:
    """Parse the transpiler optimisation level of each compilation variant (c0, c1, ...)."""
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{ctx}: expected a non-empty list, got {value!r}")
    levels = tuple(_as_int(level, f"{ctx}[{i}]") for i, level in enumerate(value))
    if invalid := [level for level in levels if level not in QISKIT_OPT_LEVELS]:
        raise ConfigError(f"{ctx}: invalid optimisation levels {invalid}")
    if len(set(levels)) != len(levels):
        raise ConfigError(f"{ctx}: repeated optimisation levels {list(levels)}")
    return levels


def _parse_labels(value: Any, ctx: str) -> LabelsConfig:
    """Parse the ``etiquetas`` section (consistency is checked later)."""
    data = _as_mapping(value, ctx)
    keys = ("fusiones", "solo_familia", "clase_propia_con_minimo", "minimo_bases_clase_propia")
    _warn_unknown_keys(data, keys, ctx)
    merges, merges_ctx = _get(data, "fusiones", ctx)
    merges = _as_mapping(merges, merges_ctx)
    return LabelsConfig(
        fusiones={
            _as_str(source, merges_ctx): _as_str(target, f"{merges_ctx}[{source}]")
            for source, target in merges.items()
        },
        solo_familia=_as_str_tuple(*_get(data, "solo_familia", ctx)),
        clase_propia_con_minimo=_as_str_tuple(*_get(data, "clase_propia_con_minimo", ctx)),
        minimo_bases_clase_propia=_as_int(*_get(data, "minimo_bases_clase_propia", ctx), minimum=1),
    )


def _parse_oracles(value: Any, ctx: str) -> OraclesConfig:
    """Parse the ``oraculos`` section."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, ("por_tamano",), ctx)
    return OraclesConfig(por_tamano=_as_int(*_get(data, "por_tamano", ctx), minimum=1))


def _parse_dataset(data: Mapping[str, Any], ctx: str, config_dir: Path) -> DatasetConfig:
    """Build a ``DatasetConfig`` from the raw YAML mapping (structure and types only)."""
    _warn_unknown_keys(data, DATASET_KEYS, ctx)
    nativegates = _as_mapping(*_get(data, "nativegates", ctx))
    mapped = _as_mapping(*_get(data, "mapped", ctx))
    return DatasetConfig(
        semilla_global=_as_int(*_get(data, "semilla_global", ctx), minimum=0),
        tamanos=_parse_size_mapping(*_get(data, "tamanos", ctx)),
        niveles=_parse_levels(*_get(data, "niveles", ctx)),
        nativegates=NativeGatesConfig(
            gateset=_as_str(*_get(nativegates, "gateset", f"{ctx}[nativegates]"))
        ),
        mapped=MappedConfig(dispositivo=_as_str(*_get(mapped, "dispositivo", f"{ctx}[mapped]"))),
        opt_level=_as_int(*_get(data, "opt_level", ctx)),
        niveles_opt_compilacion=_parse_opt_levels(*_get(data, "niveles_opt_compilacion", ctx)),
        variantes_parametros=_as_int(*_get(data, "variantes_parametros", ctx), minimum=1),
        max_nodos=_as_int(*_get(data, "max_nodos", ctx), minimum=1),
        familias=_parse_families(*_get(data, "familias", ctx)),
        control_negativo=_as_str_tuple(*_get(data, "control_negativo", ctx)),
        reservados_sin_familia=_as_str_tuple(*_get(data, "reservados_sin_familia", ctx)),
        excluidos=_as_str_tuple(*_get(data, "excluidos", ctx)),
        variacionales_con_parametros=_as_str_tuple(
            *_get(data, "variacionales_con_parametros", ctx)
        ),
        etiquetas=_parse_labels(*_get(data, "etiquetas", ctx)),
        oraculos=_parse_oracles(*_get(data, "oraculos", ctx)),
        rutas=_parse_paths(*_get(data, "rutas", ctx), config_dir=config_dir),
    )


def _dataset_consistency_errors(config: DatasetConfig) -> list[str]:
    """Return the internal inconsistencies of a dataset configuration."""
    errors = []
    if config.opt_level not in QISKIT_OPT_LEVELS:
        errors.append(f"opt_level must be one of {list(QISKIT_OPT_LEVELS)}, got {config.opt_level}")
    classes = set(config.all_algorithms)
    in_several = sorted(n for n, c in Counter(config.all_algorithms).items() if c > 1)
    if in_several:
        errors.append(f"algorithms listed in more than one family: {in_several}")
    negative = set(config.control_negativo)
    unfamiliar = set(config.reservados_sin_familia)
    if negative_as_class := sorted(classes & negative):
        errors.append(f"control_negativo entries are also class algorithms: {negative_as_class}")
    if unfamiliar_used := sorted(unfamiliar & (classes | negative)):
        errors.append(f"reservados_sin_familia entries are also used elsewhere: {unfamiliar_used}")
    if excluded_used := sorted(set(config.excluidos) & (classes | negative | unfamiliar)):
        errors.append(f"excluidos entries are also used elsewhere: {excluded_used}")
    merges = config.etiquetas.fusiones
    if bad := sorted(name for name in [*merges, *merges.values()] if name not in classes):
        errors.append(f"etiquetas.fusiones mentions non-class algorithms: {bad}")
    if chained := sorted(set(merges) & set(merges.values())):
        errors.append(f"etiquetas.fusiones merges into merged classes: {chained}")
    if bad := sorted(set(config.etiquetas.clase_propia_con_minimo) - classes):
        errors.append(f"etiquetas.clase_propia_con_minimo mentions non-classes: {bad}")
    if bad := sorted(set(config.etiquetas.solo_familia) - classes):
        errors.append(f"etiquetas.solo_familia mentions non-classes: {bad}")
    if bad := sorted(set(config.etiquetas.solo_familia) & {*merges, *merges.values()}):
        errors.append(f"etiquetas.solo_familia entries are also merged: {bad}")
    if not_classes := sorted(set(config.variacionales_con_parametros) - classes):
        errors.append(f"variacionales_con_parametros entries are not classes: {not_classes}")
    return errors


def _mqt_bench_errors(config: DatasetConfig) -> list[str]:
    """Return benchmark, gateset and device names that the installed mqt.bench does not know."""
    mqt_version = version("mqt.bench")
    errors = []
    available = set(get_available_benchmark_names())
    unknown = [name for name in config.referenced_benchmarks if name not in available]
    for name in unknown:
        logger.warning("benchmark '%s' does not exist in mqt.bench %s", name, mqt_version)
    if unknown:
        errors.append(f"benchmarks not available in mqt.bench {mqt_version}: {unknown}")
    gatesets = get_available_gateset_names()
    if config.nativegates.gateset not in gatesets:
        errors.append(
            f"gateset '{config.nativegates.gateset}' not available in mqt.bench {mqt_version}; "
            f"available gatesets: {gatesets}"
        )
    devices = get_available_device_names()
    if config.mapped.dispositivo not in devices:
        errors.append(
            f"device '{config.mapped.dispositivo}' not available in mqt.bench {mqt_version}; "
            f"available devices: {devices}"
        )
    if unreferenced := sorted(available - set(config.referenced_benchmarks)):
        logger.info("mqt.bench benchmarks not referenced in the configuration: %s", unreferenced)
    return errors


def load_dataset_config(path: Path | str) -> DatasetConfig:
    """Load ``dataset.yaml`` and validate it, including every name against mqt.bench.

    Raises:
        ConfigError: listing every problem found in the file.
    """
    path = Path(path)
    config = _parse_dataset(_read_yaml(path), path.name, config_dir=path.parent)
    errors = _dataset_consistency_errors(config) + _mqt_bench_errors(config)
    if errors:
        raise ConfigError(f"{path}:\n  - " + "\n  - ".join(errors))
    return config


# --------------------------------------------------------------------------------------------
# splits.yaml
# --------------------------------------------------------------------------------------------


def _parse_random_split(value: Any, ctx: str) -> RandomSplitConfig:
    """Parse the ``aleatorio`` fractions and check that they add up to 1."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, SPLIT_SETS, ctx)
    fractions = {name: _as_fraction(*_get(data, name, ctx)) for name in SPLIT_SETS}
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ConfigError(f"{ctx}: fractions must add up to 1, got {sum(fractions.values())}")
    if fractions["train"] == 0.0:
        raise ConfigError(f"{ctx}: the train fraction must be positive")
    return RandomSplitConfig(**fractions)


def _parse_size_split(value: Any, ctx: str) -> SizeSplitConfig:
    """Parse the ``por_tamano`` ranges and check that they do not overlap."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, SPLIT_SETS, ctx)
    ranges = {name: _parse_size_pair(*_get(data, name, ctx)) for name in SPLIT_SETS}
    ordered = sorted(ranges.items(), key=lambda item: item[1].min)
    for (name_a, a), (name_b, b) in pairwise(ordered):
        if b.min <= a.max:
            raise ConfigError(f"{ctx}: ranges '{name_a}' and '{name_b}' overlap")
    return SizeSplitConfig(**ranges)


def _parse_level_split(value: Any, ctx: str) -> LevelSplitConfig:
    """Parse ``por_nivel``: disjoint, valid level lists for train and test."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, ("train", "test"), ctx)
    train = _parse_levels(*_get(data, "train", ctx))
    test = _parse_levels(*_get(data, "test", ctx))
    if overlap := sorted(set(train) & set(test)):
        raise ConfigError(f"{ctx}: levels in both train and test: {overlap}")
    return LevelSplitConfig(train=train, test=test)


def _parse_balancing(value: Any, ctx: str) -> BalancingConfig:
    """Parse the ``balanceo`` section."""
    data = _as_mapping(value, ctx)
    _warn_unknown_keys(data, ("modo", "max_bases_por_clase"), ctx)
    mode = _as_str(*_get(data, "modo", ctx))
    if mode not in BALANCING_MODES:
        raise ConfigError(f"{ctx}[modo]: must be one of {list(BALANCING_MODES)}, got {mode!r}")
    cap = _as_optional_int(*_get(data, "max_bases_por_clase", ctx), minimum=1)
    return BalancingConfig(modo=mode, max_bases_por_clase=cap)


def _parse_splits(data: Mapping[str, Any], ctx: str) -> SplitsConfig:
    """Build a ``SplitsConfig`` from the raw YAML mapping."""
    _warn_unknown_keys(data, SPLITS_KEYS, ctx)
    return SplitsConfig(
        semilla=_as_int(*_get(data, "semilla", ctx), minimum=0),
        aleatorio=_parse_random_split(*_get(data, "aleatorio", ctx)),
        por_tamano=_parse_size_split(*_get(data, "por_tamano", ctx)),
        algoritmos_reservados=_as_str_tuple(*_get(data, "algoritmos_reservados", ctx)),
        minimo_por_clase_train=_as_int(*_get(data, "minimo_por_clase_train", ctx), minimum=0),
        balanceo=_parse_balancing(*_get(data, "balanceo", ctx)),
        por_nivel=_parse_level_split(*_get(data, "por_nivel", ctx)),
    )


def _check_splits_against_dataset(splits: SplitsConfig, dataset: DatasetConfig) -> list[str]:
    """Cross-check the split protocols with the dataset; return errors, log soft issues."""
    errors = []
    available = set(get_available_benchmark_names())
    if unknown := [n for n in splits.algoritmos_reservados if n not in available]:
        errors.append(f"algoritmos_reservados not available in mqt.bench: {unknown}")
    generated = set(dataset.all_algorithms) | set(dataset.reservados_sin_familia)
    for name in splits.algoritmos_reservados:
        if name in available and name not in generated:
            logger.warning("reserved algorithm '%s' is not generated by the dataset", name)
    for name in dataset.reservados_sin_familia:
        if name not in splits.algoritmos_reservados:
            logger.warning("'%s' has no family but is not in algoritmos_reservados", name)
    for name in SPLIT_SETS:
        size_range = getattr(splits.por_tamano, name)
        if size_range.min < dataset.tamanos.min or size_range.max > dataset.tamanos.max:
            logger.warning(
                "por_tamano[%s] = [%d, %d] exceeds the dataset sizes [%d, %d]",
                name,
                size_range.min,
                size_range.max,
                dataset.tamanos.min,
                dataset.tamanos.max,
            )
    return errors


def load_splits_config(path: Path | str, dataset: DatasetConfig | None = None) -> SplitsConfig:
    """Load ``splits.yaml``; if ``dataset`` is given, cross-check it with that configuration.

    Raises:
        ConfigError: if the file is malformed or inconsistent with the dataset.
    """
    path = Path(path)
    splits = _parse_splits(_read_yaml(path), path.name)
    if dataset is not None and (errors := _check_splits_against_dataset(splits, dataset)):
        raise ConfigError(f"{path}:\n  - " + "\n  - ".join(errors))
    return splits
