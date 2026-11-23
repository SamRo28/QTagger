"""F1: generate raw OpenQASM 3 circuits from MQT Bench.

Work is split into tasks, one per (algorithm, parameter variant), and each task walks the
sizes in increasing order. For every size (a *base circuit*, identified by its ``group_id``):

1. The ALG circuit is built first. If MQT Bench rejects the size, the error is logged.
2. Circuits with mid-circuit measurements, resets or classical control flow are rejected.
3. Size probe: the NATIVEGATES circuit (already flattened) is built. If it has more than
   ``max_nodos`` gates, no level is generated for this base and the larger sizes of the
   same task are skipped, because circuit size grows with the number of qubits.
4. Every requested level is generated and written as OpenQASM 3.

Seeds: when no seed is given, MQT Bench 2.4 always uses the same one (10), so a seed
derived from ``semilla_global``, the algorithm, the size and the variant is passed
explicitly. It sets the random parameter values and the factories that accept a seed
(graphstate, qaoa, qpeexact, qpeinexact, randomcircuit). All levels of a base share it.

Generation is incremental: a circuit is skipped when its file exists and it is already
listed in the metadata, unless ``force`` is set.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pandas as pd
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device, get_target_for_gateset
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

from qcclass.circuitos import count_gates, is_dynamic, qpy_path, write_circuit
from qcclass.config import DatasetConfig
from qcclass.metadatos import METADATA_COLUMNS, read_table, replace_scope, upsert
from qcclass.paralelo import run_tasks
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

ORIGIN = "mqt"
NEGATIVE_CONTROL_FAMILY = "control_negativo"
NO_FAMILY = "sin_familia"
ERRORS_FILE_NAME = "generacion_errores.csv"
ALL_LEVELS = "todos"
PROBE_LEVEL = "nativegates"
ERROR_COLUMNS = (
    "algoritmo",
    "n",
    "variante_param",
    "nivel",
    "familia",
    "semilla",
    "tipo",
    "mensaje",
)
ERROR_SCOPE = ("algoritmo", "n", "variante_param")
MAX_MESSAGE_LENGTH = 300

# Error types written to the ``tipo`` column of the error log.
ERROR_GENERATION = "generacion"
ERROR_DYNAMIC = "dinamico"
ERROR_TOO_LARGE = "demasiado_grande"
ERROR_SKIPPED_SIZE = "omitido_por_tamano"
ERROR_SKIPPED_DEVICE = "omitido_dispositivo"
ERROR_EXPORT = "exportacion"


# --------------------------------------------------------------------------------------------
# What to generate
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """Subset of the configuration to generate (command-line overrides applied)."""

    algorithms: tuple[str, ...]
    sizes: tuple[int, ...]
    levels: tuple[str, ...]


@dataclass(frozen=True)
class Task:
    """Generation work for one algorithm and parameter variant, over increasing sizes."""

    algorithm: str
    family: str
    variant: int
    sizes: tuple[int, ...]
    levels: tuple[str, ...]
    existing_ids: frozenset[str]
    force: bool


@dataclass
class TaskResult:
    """Metadata rows of the written circuits and error-log rows of one task."""

    rows: list[dict[str, object]] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)
    skipped: int = 0


@dataclass(frozen=True)
class GenerationReport:
    """Outcome of a generation run."""

    rows: pd.DataFrame
    errors: pd.DataFrame
    skipped: int


def generatable_algorithms(config: DatasetConfig) -> dict[str, str]:
    """Map every benchmark to generate to its family.

    Classes keep their family; the negative control and the family-less reserved
    algorithms get the pseudo-families ``control_negativo`` and ``sin_familia``.
    """
    families = {name: family for family, names in config.familias.items() for name in names}
    families.update(dict.fromkeys(config.control_negativo, NEGATIVE_CONTROL_FAMILY))
    families.update(dict.fromkeys(config.reservados_sin_familia, NO_FAMILY))
    return families


def parameter_variants(config: DatasetConfig, algorithm: str) -> range:
    """Parameter variants of an algorithm: several for variational ones, a single one otherwise."""
    if algorithm in config.variacionales_con_parametros:
        return range(config.variantes_parametros)
    return range(1)


def _select_algorithms(config: DatasetConfig, requested: Sequence[str] | None) -> tuple[str, ...]:
    """Validate the requested algorithms (default: every generatable one)."""
    available = generatable_algorithms(config)
    if not requested:
        return tuple(available)
    if excluded := [name for name in requested if name in config.excluidos]:
        raise ValueError(f"excluded benchmarks cannot be generated: {excluded}")
    if unknown := [name for name in requested if name not in available]:
        raise ValueError(f"unknown algorithms {unknown}; available: {sorted(available)}")
    return tuple(dict.fromkeys(requested))


def _select_sizes(config: DatasetConfig, low: int | None, high: int | None) -> tuple[int, ...]:
    """Size range from the configuration, optionally overridden."""
    low = config.tamanos.min if low is None else low
    high = config.tamanos.max if high is None else high
    if low < 1 or low > high:
        raise ValueError(f"invalid size range [{low}, {high}]")
    return tuple(range(low, high + 1))


def _select_levels(config: DatasetConfig, requested: Sequence[str] | None) -> tuple[str, ...]:
    """Levels to generate; they must be among the configured ones."""
    if not requested:
        return config.niveles
    if unknown := [level for level in requested if level not in config.niveles]:
        raise ValueError(f"levels {unknown} are not configured; configured: {config.niveles}")
    return tuple(level for level in config.niveles if level in requested)


def build_selection(
    config: DatasetConfig,
    algorithms: Sequence[str] | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    levels: Sequence[str] | None = None,
) -> Selection:
    """Build and validate a selection; ``None`` means "as configured".

    Raises:
        ValueError: if an algorithm, size range or level is not valid.
    """
    return Selection(
        algorithms=_select_algorithms(config, algorithms),
        sizes=_select_sizes(config, min_size, max_size),
        levels=_select_levels(config, levels),
    )


def build_tasks(
    config: DatasetConfig, selection: Selection, existing_ids: frozenset[str], force: bool
) -> list[Task]:
    """One task per selected algorithm and parameter variant."""
    families = generatable_algorithms(config)
    return [
        Task(
            algorithm=algorithm,
            family=families[algorithm],
            variant=variant,
            sizes=selection.sizes,
            levels=selection.levels,
            existing_ids=existing_ids,
            force=force,
        )
        for algorithm in selection.algorithms
        for variant in parameter_variants(config, algorithm)
    ]


# --------------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------------


def group_id(algorithm: str, n: int, variant: int) -> str:
    """Identifier shared by every version (level, compilation) of a base circuit."""
    return f"{algorithm}_n{n}_p{variant}"


def circuit_id(level: str, group: str) -> str:
    """Identifier of a raw circuit file."""
    return f"{level}_{group}"


def raw_path(config: DatasetConfig, level: str, family: str, algorithm: str, group: str) -> Path:
    """``<raw>/<level>/<family>/<algorithm>/<group_id>.qasm``."""
    return config.rutas.raw / level / family / algorithm / f"{group}.qasm"


def relative_to_root(config: DatasetConfig, path: Path) -> str:
    """Path relative to ``rutas.raiz`` (as stored in the metadata), or absolute if outside it."""
    return (
        path.relative_to(config.rutas.raiz) if path.is_relative_to(config.rutas.raiz) else path
    ).as_posix()


def base_seed(config: DatasetConfig, algorithm: str, n: int, variant: int) -> int:
    """Seed of a base circuit, shared by all its levels."""
    return derive_seed(config.semilla_global, algorithm, n, variant)


# --------------------------------------------------------------------------------------------
# Generation of one task
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Base:
    """A base circuit being generated: one (algorithm, size, variant)."""

    task: Task
    n: int
    seed: int
    group: str


def _error_row(base: _Base, level: str, kind: str, detail: str | Exception) -> dict[str, object]:
    """Build an error-log row; exceptions are reduced to a single line."""
    if isinstance(detail, Exception):
        detail = f"{type(detail).__name__}: {detail}"
    message = " ".join(detail.split())[:MAX_MESSAGE_LENGTH]
    logger.debug("%s %s [%s] %s: %s", base.task.algorithm, base.group, level, kind, message)
    return {
        "algoritmo": base.task.algorithm,
        "n": base.n,
        "variante_param": base.task.variant,
        "nivel": level,
        "familia": base.task.family,
        "semilla": base.seed,
        "tipo": kind,
        "mensaje": message,
    }


def _metadata_row(
    config: DatasetConfig, base: _Base, level: str, width: int, path: Path
) -> dict[str, object]:
    """Metadata row of a written raw circuit; ``n_qubits`` is the logical width."""
    return {
        "id": circuit_id(level, base.group),
        "algoritmo": base.task.algorithm,
        "familia": base.task.family,
        "nivel": level,
        "n_qubits": width,
        "variante_param": base.task.variant,
        "variante_comp": "",
        "semilla": base.seed,
        "origen": ORIGIN,
        "group_id": base.group,
        "ruta_raw": relative_to_root(config, path),
    }


def _build_level(
    config: DatasetConfig, base: _Base, level: str, width: int, device: Target | None
) -> QuantumCircuit:
    """Call MQT Bench for one level of a base circuit."""
    benchmark_level = BenchmarkLevel[level.upper()]
    target = None
    if benchmark_level is BenchmarkLevel.NATIVEGATES:
        target = get_target_for_gateset(config.nativegates.gateset, width)
    elif benchmark_level is BenchmarkLevel.MAPPED:
        target = device
    return get_benchmark(
        base.task.algorithm,
        benchmark_level,
        base.n,
        target=target,
        opt_level=config.opt_level,
        seed=base.seed,
    )


def _raw_files_exist(qasm_file: Path) -> bool:
    """True if both the QASM file and its QPY twin exist."""
    return qasm_file.is_file() and qpy_path(qasm_file).is_file()


def _pending_levels(config: DatasetConfig, base: _Base) -> list[str]:
    """Levels of a base circuit that still have to be written."""
    task = base.task
    return [
        level
        for level in task.levels
        if task.force
        or circuit_id(level, base.group) not in task.existing_ids
        or not _raw_files_exist(raw_path(config, level, task.family, task.algorithm, base.group))
    ]


def _write_level(
    config: DatasetConfig,
    base: _Base,
    level: str,
    width: int,
    built: dict[str, QuantumCircuit],
    device: Target | None,
    result: TaskResult,
) -> None:
    """Build (unless already built), export and record one level of a base circuit."""
    if level == "mapped" and device is not None and width > device.num_qubits:
        detail = f"{width} qubits > {device.num_qubits} qubits of the device"
        result.errors.append(_error_row(base, level, ERROR_SKIPPED_DEVICE, detail))
        return
    try:
        if level in built:
            circuit = built[level]
        else:
            circuit = _build_level(config, base, level, width, device)
    except Exception as exc:
        result.errors.append(_error_row(base, level, ERROR_GENERATION, exc))
        return
    path = raw_path(config, level, base.task.family, base.task.algorithm, base.group)
    try:
        write_circuit(circuit, path)
    except Exception as exc:
        result.errors.append(_error_row(base, level, ERROR_EXPORT, exc))
        return
    result.rows.append(_metadata_row(config, base, level, width, path))


def _generate_base(
    config: DatasetConfig, base: _Base, device: Target | None, result: TaskResult
) -> bool:
    """Generate the pending levels of a base circuit.

    Returns:
        False if the base exceeds ``max_nodos`` (larger sizes must be skipped), else True.
    """
    pending = _pending_levels(config, base)
    result.skipped += len(base.task.levels) - len(pending)
    if not pending:
        return True
    try:
        algorithmic = get_benchmark(base.task.algorithm, BenchmarkLevel.ALG, base.n, seed=base.seed)
    except Exception as exc:
        result.errors.append(_error_row(base, ALL_LEVELS, ERROR_GENERATION, exc))
        return True
    if is_dynamic(algorithmic):
        detail = "mid-circuit measurement, reset or classical control flow"
        result.errors.append(_error_row(base, ALL_LEVELS, ERROR_DYNAMIC, detail))
        return True
    width = algorithmic.num_qubits
    try:
        probe = _build_level(config, base, PROBE_LEVEL, width, device)
    except Exception as exc:
        result.errors.append(_error_row(base, PROBE_LEVEL, ERROR_GENERATION, exc))
        return True
    if (gates := count_gates(probe)) > config.max_nodos:
        detail = f"{PROBE_LEVEL} circuit has {gates} gates > max_nodos={config.max_nodos}"
        result.errors.append(_error_row(base, ALL_LEVELS, ERROR_TOO_LARGE, detail))
        return False
    built = {"alg": algorithmic, PROBE_LEVEL: probe}
    for level in pending:
        _write_level(config, base, level, width, built, device, result)
    return True


def run_task(task: Task, config: DatasetConfig) -> TaskResult:
    """Generate every size of a task, stopping at the first size that exceeds ``max_nodos``."""
    result = TaskResult()
    device = get_device(config.mapped.dispositivo) if "mapped" in task.levels else None
    too_large_at: int | None = None
    for n in task.sizes:
        variant_group = group_id(task.algorithm, n, task.variant)
        base = _Base(task, n, base_seed(config, task.algorithm, n, task.variant), variant_group)
        if too_large_at is not None:
            detail = f"n={too_large_at} already exceeds max_nodos"
            result.errors.append(_error_row(base, ALL_LEVELS, ERROR_SKIPPED_SIZE, detail))
        elif not _generate_base(config, base, device, result):
            too_large_at = n
    return result


# --------------------------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------------------------


def errors_path(config: DatasetConfig) -> Path:
    """Location of the generation error log."""
    return config.rutas.logs / ERRORS_FILE_NAME


def existing_circuit_ids(config: DatasetConfig) -> frozenset[str]:
    """Ids of the MQT circuits already listed in the metadata."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    return frozenset(metadata.loc[metadata["origen"] == ORIGIN, "id"])


def generate(
    config: DatasetConfig, selection: Selection, *, force: bool = False, processes: int = 1
) -> GenerationReport:
    """Generate the selected circuits, update the metadata and the error log.

    Every circuit is kept on disk; class balancing happens in the split step (Phase 7).
    """
    tasks = build_tasks(config, selection, existing_circuit_ids(config), force)
    logger.info(
        "generating %d algorithms, sizes %d-%d, levels %s (%d tasks)",
        len(selection.algorithms),
        selection.sizes[0],
        selection.sizes[-1],
        ",".join(selection.levels),
        len(tasks),
    )
    results = run_tasks(partial(run_task, config=config), tasks, processes, "generar")
    rows = pd.DataFrame([row for r in results for row in r.rows], columns=METADATA_COLUMNS)
    errors = pd.DataFrame([row for r in results for row in r.errors], columns=ERROR_COLUMNS)
    if len(rows):
        upsert(config.rutas.metadata, rows, "id", METADATA_COLUMNS)
    scopes = [(task.algorithm, n, task.variant) for task in tasks for n in task.sizes]
    replace_scope(errors_path(config), errors, ERROR_SCOPE, scopes, ERROR_COLUMNS)
    skipped = sum(result.skipped for result in results)
    logger.info(
        "%d circuits written, %d already present, %d errors", len(rows), skipped, len(errors)
    )
    return GenerationReport(rows, errors, skipped)
