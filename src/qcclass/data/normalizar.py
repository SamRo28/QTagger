"""F2: flatten and normalise circuits to the canonical gate set (rz, sx, x, cx).

For every raw circuit (read from its QPY twin):

1. Final measurements and barriers are removed; ``mapped`` circuits also lose their idle
   physical qubits, so their width is the logical one.
2. The circuit is transpiled to the canonical basis once per compilation variant. Variants
   differ in optimisation level (``niveles_opt_compilacion``, e.g. c0 = 1, c1 = 0, c2 = 2):
   without a coupling map ``transpile`` is deterministic and seeds alone would produce
   identical circuits. Only ``alg`` and ``indep`` circuits get every variant; ``nativegates``
   and ``mapped`` ones only c0. At optimisation level >= 2 Qiskit may elide SWAP gates into
   an output permutation (``layout.final_layout``, kept in the QPY file).
3. The result must contain only canonical gates and no ``gate`` definitions.
4. Circuits with more than ``max_nodos`` gates are discarded and logged.
5. The result is written as ``<canonical>/<level>/<family>/<algorithm>/<id>_c<k>.qasm`` plus
   its QPY twin, and a metadata row is added per variant (``group_id`` inherited).
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import RemoveBarriers

from qcclass.circuitos import (
    CANONICAL_BASIS,
    count_gates,
    qpy_path,
    read_qpy,
    write_qpy,
    write_text_atomic,
)
from qcclass.config import DatasetConfig
from qcclass.metadatos import METADATA_COLUMNS, read_table, replace_scope, upsert
from qcclass.paralelo import run_tasks
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

ALL_VARIANT_LEVELS = ("alg", "indep")
LOG_FILE_NAME = "normalizacion_descartes.csv"
LOG_COLUMNS = ("id_raw", "variante_comp", "algoritmo", "nivel", "tipo", "mensaje")
LOG_SCOPE = ("id_raw",)
DISCARD_TOO_LARGE = "max_nodos"
DISCARD_NOT_CANONICAL = "no_canonico"
DISCARD_ERROR = "error"
_GATE_DEFINITION = re.compile(r"^\s*gate\s", re.MULTILINE)


# --------------------------------------------------------------------------------------------
# Circuit transformations
# --------------------------------------------------------------------------------------------


def remove_idle_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
    """Drop the qubits that no operation touches."""
    dag = circuit_to_dag(circuit)
    idle = [wire for wire in dag.idle_wires() if isinstance(wire, Qubit)]
    if idle:
        dag.remove_qubits(*idle)
    return dag_to_circuit(dag)


def strip_non_unitary(circuit: QuantumCircuit, *, drop_idle_qubits: bool) -> QuantumCircuit:
    """Remove final measurements and barriers (and idle qubits if requested)."""
    body = RemoveBarriers()(circuit.remove_final_measurements(inplace=False))
    return remove_idle_qubits(body) if drop_idle_qubits else body


def to_canonical(circuit: QuantumCircuit, optimization_level: int, seed: int) -> QuantumCircuit:
    """Transpile to the canonical basis ``rz, sx, x, cx`` (no coupling map)."""
    return transpile(
        circuit,
        basis_gates=list(CANONICAL_BASIS),
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )


def non_canonical_content(circuit: QuantumCircuit, qasm_text: str) -> list[str]:
    """Gate names outside the canonical basis, plus ``gate`` if QASM defines any gate."""
    problems = sorted(set(circuit.count_ops()) - set(CANONICAL_BASIS))
    if _GATE_DEFINITION.search(qasm_text):
        problems.append("gate")
    return problems


# --------------------------------------------------------------------------------------------
# Dataset step
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawCircuit:
    """A raw circuit to normalise (one metadata row) and the variants still to produce."""

    row: dict[str, str]
    variants: tuple[int, ...]


@dataclass
class NormalizationResult:
    """Metadata rows of the canonical circuits and discard-log rows of one raw circuit."""

    rows: list[dict[str, object]] = field(default_factory=list)
    log: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizationReport:
    """Outcome of a normalisation run."""

    rows: pd.DataFrame
    log: pd.DataFrame
    skipped: int


def variant_indices(config: DatasetConfig, level: str) -> tuple[int, ...]:
    """Compilation variants of a level: all for alg/indep, only c0 otherwise."""
    if level in ALL_VARIANT_LEVELS:
        return tuple(range(len(config.niveles_opt_compilacion)))
    return (0,)


def canonical_id(raw_id: str, variant: int) -> str:
    """Identifier of a canonical circuit."""
    return f"{raw_id}_c{variant}"


def canonical_path(config: DatasetConfig, row: dict[str, str], variant: int) -> Path:
    """``<canonical>/<level>/<family>/<algorithm>/<id>_c<k>.qasm``."""
    name = f"{canonical_id(row['id'], variant)}.qasm"
    return config.rutas.canonical / row["nivel"] / row["familia"] / row["algoritmo"] / name


def _log(row: dict[str, str], variant: int, kind: str, message: str) -> dict[str, object]:
    """Discard-log row."""
    return {
        "id_raw": row["id"],
        "variante_comp": f"c{variant}",
        "algoritmo": row["algoritmo"],
        "nivel": row["nivel"],
        "tipo": kind,
        "mensaje": " ".join(message.split())[:300],
    }


def _canonical_row(
    config: DatasetConfig, row: dict[str, str], variant: int, circuit: QuantumCircuit, path: Path
) -> dict[str, object]:
    """Metadata row of a canonical circuit, inheriting the raw row (``group_id`` included)."""
    return {
        **row,
        "id": canonical_id(row["id"], variant),
        "n_qubits": circuit.num_qubits,
        "variante_comp": f"c{variant}",
        "opt_level_comp": config.niveles_opt_compilacion[variant],
        "ruta_canonical": path.relative_to(config.rutas.raiz).as_posix(),
        "n_puertas": count_gates(circuit),
        "profundidad": circuit.depth(),
        "huella_sha": "",
        "huella_wl": "",
        "duplicado_de": "",
    }


def _normalize_variant(
    config: DatasetConfig, item: RawCircuit, body: QuantumCircuit, variant: int
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Produce one variant; return (metadata row, None) or (None, discard-log row)."""
    row = item.row
    seed = derive_seed(config.semilla_global, "normalizar", row["id"], variant)
    canonical = to_canonical(body, config.niveles_opt_compilacion[variant], seed)
    if (gates := count_gates(canonical)) > config.max_nodos:
        message = f"{gates} gates > max_nodos={config.max_nodos}"
        return None, _log(row, variant, DISCARD_TOO_LARGE, message)
    text = qasm3.dumps(canonical)
    if problems := non_canonical_content(canonical, text):
        return None, _log(row, variant, DISCARD_NOT_CANONICAL, f"found {problems}")
    path = canonical_path(config, row, variant)
    write_text_atomic(text, path)
    write_qpy(canonical, qpy_path(path))
    return _canonical_row(config, row, variant, canonical, path), None


def normalize_raw(item: RawCircuit, config: DatasetConfig) -> NormalizationResult:
    """Normalise every pending variant of one raw circuit."""
    result = NormalizationResult()
    row = item.row
    try:
        raw = read_qpy(qpy_path(config.rutas.raiz / row["ruta_raw"]))
        body = strip_non_unitary(raw, drop_idle_qubits=row["nivel"] == "mapped")
    except Exception as exc:
        result.log.append(
            _log(row, item.variants[0], DISCARD_ERROR, f"{type(exc).__name__}: {exc}")
        )
        return result
    for variant in item.variants:
        try:
            metadata_row, log_row = _normalize_variant(config, item, body, variant)
        except Exception as exc:
            metadata_row, log_row = None, _log(row, variant, DISCARD_ERROR, f"{exc!r}")
        if metadata_row is not None:
            result.rows.append(metadata_row)
        if log_row is not None:
            result.log.append(log_row)
    return result


def _raw_rows(metadata: pd.DataFrame, levels: Sequence[str] | None) -> pd.DataFrame:
    """Raw-circuit rows (no compilation variant), optionally restricted to some levels."""
    raw = metadata[metadata["variante_comp"] == ""]
    return raw if not levels else raw[raw["nivel"].isin(list(levels))]


def _pending(
    config: DatasetConfig, raw: pd.DataFrame, existing: set[str], force: bool
) -> tuple[list[RawCircuit], int]:
    """Raw circuits with variants to produce, and the number of variants already present."""
    items, skipped = [], 0
    for row in raw.to_dict("records"):
        todo = []
        for variant in variant_indices(config, row["nivel"]):
            path = canonical_path(config, row, variant)
            done = canonical_id(row["id"], variant) in existing and qpy_path(path).is_file()
            if done and not force:
                skipped += 1
            else:
                todo.append(variant)
        if todo:
            items.append(RawCircuit(row=row, variants=tuple(todo)))
    return items, skipped


def log_path(config: DatasetConfig) -> Path:
    """Location of the normalisation discard log."""
    return config.rutas.logs / LOG_FILE_NAME


def normalize(
    config: DatasetConfig,
    *,
    levels: Sequence[str] | None = None,
    force: bool = False,
    processes: int = 1,
) -> NormalizationReport:
    """Normalise every raw circuit listed in the metadata (incrementally)."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    raw = _raw_rows(metadata, levels)
    existing = set(metadata.loc[metadata["variante_comp"] != "", "id"])
    items, skipped = _pending(config, raw, existing, force)
    logger.info("normalising %d raw circuits (%d variants already present)", len(items), skipped)
    results = run_tasks(partial(normalize_raw, config=config), items, processes, "normalizar")
    rows = pd.DataFrame([r for res in results for r in res.rows], columns=METADATA_COLUMNS)
    log = pd.DataFrame([r for res in results for r in res.log], columns=LOG_COLUMNS)
    if len(rows):
        upsert(config.rutas.metadata, rows, "id", METADATA_COLUMNS)
    scopes = [(item.row["id"],) for item in items]
    replace_scope(log_path(config), log, LOG_SCOPE, scopes, LOG_COLUMNS)
    logger.info("%d canonical circuits written, %d discarded", len(rows), len(log))
    return NormalizationReport(rows=rows, log=log, skipped=skipped)
