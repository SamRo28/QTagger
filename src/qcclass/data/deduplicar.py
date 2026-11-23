"""Detect duplicated canonical circuits.

Two fingerprints are computed per canonical circuit:

* ``huella_sha``: SHA-256 of the canonical OpenQASM 3 text without comments or redundant
  whitespace (exact identity, angles included).
* ``huella_wl``: Weisfeiler-Lehman hash of the gate DAG with the gate type as node
  attribute (structural identity: ignores angles and qubit labels).

Labels are the training classes: algorithm names after ``etiquetas.fusiones``, or
``familia:<family>`` for algorithms that only keep their family label. Only exact (SHA)
matches exclude circuits: within a group of identical texts with a single label, every
circuit but a representative gets ``duplicado_de``; groups whose circuits have different
labels are excluded entirely (from every class involved) and listed in
``duplicados_conflictivos.csv`` (``tipo = exacto``), together with structural (WL) groups
that mix labels (``tipo = estructural``), which are only reported.
"""

import hashlib
import logging
import re
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import networkx as nx
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from qcclass.circuitos import qpy_path, read_qpy
from qcclass.config import DatasetConfig
from qcclass.metadatos import METADATA_COLUMNS, read_table, upsert, write_table
from qcclass.paralelo import run_tasks

logger = logging.getLogger(__name__)

CONFLICTS_FILE_NAME = "duplicados_conflictivos.csv"
CONFLICT_COLUMNS = ("tipo", "huella", "id", "etiqueta", "algoritmo", "familia", "nivel", "n_qubits")
CONFLICT_EXACT = "exacto"
CONFLICT_STRUCTURAL = "estructural"
EXCLUDED_DUPLICATE = "duplicado"
EXCLUDED_CONFLICT = "conflicto_etiquetas"
FAMILY_LABEL_PREFIX = "familia:"
LABELLED_VARIANT = "etiquetada"  # F7 variant (see qcclass.formatos.etiquetas_puerta)
LEVEL_ORDER = {"alg": 0, "indep": 1, "nativegates": 2, "mapped": 3}
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------------------------


def normalized_qasm(text: str) -> str:
    """QASM text without comments and with every whitespace run collapsed to one space."""
    text = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
    return _WHITESPACE.sub(" ", text).strip()


def text_fingerprint(text: str) -> str:
    """SHA-256 of the normalised QASM text."""
    return hashlib.sha256(normalized_qasm(text).encode("utf-8")).hexdigest()


def gate_graph(circuit: QuantumCircuit) -> nx.DiGraph:
    """Gate dependency DAG: one node per gate (attribute ``tipo``), edges in time order."""
    dag = circuit_to_dag(circuit)
    index = {node: i for i, node in enumerate(dag.topological_op_nodes())}
    graph = nx.DiGraph()
    for node, i in index.items():
        graph.add_node(i, tipo=node.op.name)
    for source, target, _ in dag.edges():
        if isinstance(source, DAGOpNode) and isinstance(target, DAGOpNode):
            graph.add_edge(index[source], index[target])
    return graph


def structural_fingerprint(circuit: QuantumCircuit) -> str:
    """Weisfeiler-Lehman hash of :func:`gate_graph`.

    networkx >= 3.5 warns on every directed graph that its hashes changed in 3.5; the
    version is pinned in ``requirements.lock``, so the warning is silenced here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The hashes produced for directed graphs")
        return nx.weisfeiler_lehman_graph_hash(gate_graph(circuit), node_attr="tipo")


@dataclass(frozen=True)
class Fingerprints:
    """Fingerprints of one canonical circuit."""

    id: str
    sha: str
    wl: str


def fingerprint(row: dict[str, str], config: DatasetConfig) -> Fingerprints:
    """Compute both fingerprints of a canonical circuit (QASM text and QPY circuit)."""
    qasm_file = config.rutas.raiz / row["ruta_canonical"]
    sha = text_fingerprint(qasm_file.read_text(encoding="utf-8"))
    return Fingerprints(row["id"], sha, structural_fingerprint(read_qpy(qpy_path(qasm_file))))


# --------------------------------------------------------------------------------------------
# Duplicate marking
# --------------------------------------------------------------------------------------------


def _representative_order(group: pd.DataFrame) -> pd.DataFrame:
    """Order in which the representative of a group is chosen (first row wins)."""
    key = group["nivel"].map(LEVEL_ORDER).fillna(len(LEVEL_ORDER))
    return group.assign(_key=key).sort_values(["_key", "id"]).drop(columns="_key")


def mark_duplicates(canonical: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return ``duplicado_de`` per id and the conflict table.

    ``canonical`` needs the columns ``id``, ``etiqueta`` (training label), ``algoritmo``,
    ``familia``, ``nivel``, ``n_qubits``, ``huella_sha`` and ``huella_wl``.
    """
    duplicate_of = pd.Series("", index=canonical["id"].to_numpy(), dtype=object)
    conflicts = []
    for sha, group in canonical.groupby("huella_sha", sort=True):
        if len(group) < 2:
            continue
        if group["etiqueta"].nunique() > 1:
            conflicts.append(group.assign(tipo=CONFLICT_EXACT, huella=sha))
            continue
        ordered = _representative_order(group)
        representative = ordered["id"].iloc[0]
        duplicate_of[ordered["id"].iloc[1:].to_numpy()] = representative
    for wl, group in canonical.groupby("huella_wl", sort=True):
        if group["etiqueta"].nunique() > 1:
            conflicts.append(group.assign(tipo=CONFLICT_STRUCTURAL, huella=wl))
    table = pd.concat(conflicts, ignore_index=True) if conflicts else pd.DataFrame()
    return duplicate_of, table.reindex(columns=list(CONFLICT_COLUMNS))


# --------------------------------------------------------------------------------------------
# Training labels
# --------------------------------------------------------------------------------------------


def merged_labels(config: DatasetConfig, algorithms: pd.Series) -> pd.Series:
    """Algorithm names with ``etiquetas.fusiones`` applied."""
    return algorithms.map(lambda name: config.etiquetas.fusiones.get(name, name))


def family_label(family: str) -> str:
    """Label of a circuit that only keeps its family."""
    return f"{FAMILY_LABEL_PREFIX}{family}"


def _family_only_algorithms(
    config: DatasetConfig, canonical: pd.DataFrame, excluded: pd.Series
) -> set[str]:
    """Algorithms of ``clase_propia_con_minimo`` with too few unique base circuits."""
    kept = canonical[~excluded]
    bases = kept.groupby("algoritmo")["group_id"].nunique()
    minimum = config.etiquetas.minimo_bases_clase_propia
    return {a for a in config.etiquetas.clase_propia_con_minimo if bases.get(a, 0) < minimum}


def _label_and_mark(
    config: DatasetConfig, canonical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign labels and exclusions, iterating until the family-only set is stable."""
    fixed = set(config.etiquetas.solo_familia)
    family_only: set[str] = set()
    for _ in range(len(config.etiquetas.clase_propia_con_minimo) + 1):
        labels = merged_labels(config, canonical["algoritmo"])
        demoted = canonical["algoritmo"].isin(list(family_only | fixed))
        labels[demoted] = canonical.loc[demoted, "familia"].map(family_label)
        canonical = canonical.assign(etiqueta=labels)
        duplicate_of, conflicts = mark_duplicates(canonical)
        conflicting = canonical["id"].isin(conflicts.loc[conflicts["tipo"] == CONFLICT_EXACT, "id"])
        duplicated = canonical["id"].map(duplicate_of) != ""
        updated = _family_only_algorithms(config, canonical, conflicting | duplicated)
        if updated == family_only:
            break
        family_only = updated
    exclusion = pd.Series("", index=canonical.index, dtype=object)
    exclusion[duplicated] = EXCLUDED_DUPLICATE
    exclusion[conflicting] = EXCLUDED_CONFLICT
    labelled = canonical.assign(
        clase=canonical["etiqueta"].where(
            ~canonical["etiqueta"].str.startswith(FAMILY_LABEL_PREFIX), ""
        ),
        duplicado_de=canonical["id"].map(duplicate_of),
        exclusion=exclusion,
    ).drop(columns="etiqueta")
    return labelled, conflicts


@dataclass(frozen=True)
class DeduplicationReport:
    """Outcome of a deduplication run."""

    canonical: pd.DataFrame
    conflicts: pd.DataFrame


def conflicts_path(config: DatasetConfig) -> Path:
    """Location of the conflicting-duplicates list."""
    return config.rutas.logs / CONFLICTS_FILE_NAME


def deduplicate(
    config: DatasetConfig, *, force: bool = False, processes: int = 1
) -> DeduplicationReport:
    """Fingerprint every canonical circuit (cached in the metadata), label and mark it.

    Columns written: ``huella_sha``, ``huella_wl``, ``duplicado_de``, ``clase`` (training
    label after merges; empty when only the family label is kept) and ``exclusion``
    (``duplicado``, ``conflicto_etiquetas`` or empty). Exact duplicates with different
    labels are excluded from both classes.
    """
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    # labelled F7 variants are auxiliary (never trained on): they keep their own exclusion
    canonical = metadata[
        (metadata["ruta_canonical"] != "") & (metadata["variante_comp"] != LABELLED_VARIANT)
    ].copy()
    missing = canonical if force else canonical[canonical["huella_sha"] == ""]
    logger.info("fingerprinting %d of %d canonical circuits", len(missing), len(canonical))
    prints = run_tasks(
        partial(fingerprint, config=config), missing.to_dict("records"), processes, "deduplicar"
    )
    by_id = {p.id: p for p in prints}
    has_new = canonical["id"].isin(list(by_id))
    canonical.loc[has_new, "huella_sha"] = canonical.loc[has_new, "id"].map(lambda i: by_id[i].sha)
    canonical.loc[has_new, "huella_wl"] = canonical.loc[has_new, "id"].map(lambda i: by_id[i].wl)
    canonical, conflicts = _label_and_mark(config, canonical)
    upsert(config.rutas.metadata, canonical, "id", METADATA_COLUMNS)
    write_table(conflicts, conflicts_path(config))
    logger.info(
        "%d duplicates and %d label conflicts excluded",
        int((canonical["exclusion"] == EXCLUDED_DUPLICATE).sum()),
        int((canonical["exclusion"] == EXCLUDED_CONFLICT).sum()),
    )
    return DeduplicationReport(canonical=canonical, conflicts=conflicts)
