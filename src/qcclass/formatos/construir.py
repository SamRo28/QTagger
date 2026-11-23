"""Build every input format (F3-F7) from the canonical circuits.

Steps:

1. F7: for every raw ``alg`` circuit, build the block-by-block canonical circuit, store it
   as the variant ``etiquetada`` (QASM 3 + QPY, metadata row with
   ``exclusion = etiquetada`` so that it never enters the training splits) and write its
   labels to ``<gate_labels>/<id>.json``.
2. F3-F6 for every usable canonical circuit (not excluded) and every labelled variant.
3. Consistency check on a few random labelled circuits: F4 nodes = F2 gates = F6 tokens =
   F7 labels.

The formats are rebuilt completely on every run; graph files of circuits that are no longer
part of the dataset are removed.
"""

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from qiskit import qasm3

from qcclass.circuitos import count_gates, qpy_path, read_qpy, write_qpy, write_text_atomic
from qcclass.config import DatasetConfig
from qcclass.data.deduplicar import text_fingerprint
from qcclass.data.normalizar import non_canonical_content
from qcclass.formatos import caracteristicas, grafo_cubits, grafo_dag, tokens
from qcclass.formatos.comun import LabelVocabulary, gate_sequence, label_vocabulary
from qcclass.formatos.etiquetas_puerta import LABELLED_VARIANT, labelled_canonical
from qcclass.metadatos import METADATA_COLUMNS, read_table, upsert
from qcclass.paralelo import run_tasks
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

FEATURES_FILE = "features.parquet"
TOKENS_FILE = "tokens.jsonl"
TOKEN_VOCABULARY_FILE = "vocabulario.json"
LABEL_VOCABULARY_FILE = "etiquetas.json"
DAG_DIR = "dag"
QUBIT_DIR = "cubits"
CHECK_SAMPLE = 5


# --------------------------------------------------------------------------------------------
# F7: labelled variants
# --------------------------------------------------------------------------------------------


def labelled_id(raw_id: str) -> str:
    """Identifier of the labelled variant of a raw circuit."""
    return f"{raw_id}_c{LABELLED_VARIANT}"


def _labelled_row(
    config: DatasetConfig, row: dict[str, str], label: str
) -> dict[str, object] | None:
    """Build, store and describe the labelled variant of one raw ``alg`` circuit."""
    raw = read_qpy(qpy_path(config.rutas.raiz / row["ruta_raw"]))
    labelled = labelled_canonical(raw)
    circuit = labelled.circuit
    if count_gates(circuit) > config.max_nodos:
        return None
    circuit_id = labelled_id(row["id"])
    folder = config.rutas.canonical / row["nivel"] / row["familia"] / row["algoritmo"]
    path = folder / f"{circuit_id}.qasm"
    text = qasm3.dumps(circuit)
    if problems := non_canonical_content(circuit, text):
        raise ValueError(f"{circuit_id}: non-canonical content {problems}")
    write_text_atomic(text, path)
    write_qpy(circuit, qpy_path(path))
    labels = {"id": circuit_id, "id_raw": row["id"], "etiquetas": labelled.labels}
    labels["bloques"] = labelled.blocks
    write_text_atomic(
        json.dumps(labels, ensure_ascii=False), config.rutas.gate_labels / f"{circuit_id}.json"
    )
    return {
        **row,
        "id": circuit_id,
        "n_qubits": circuit.num_qubits,
        "variante_comp": LABELLED_VARIANT,
        "opt_level_comp": 0,
        "ruta_canonical": path.relative_to(config.rutas.raiz).as_posix(),
        "n_puertas": count_gates(circuit),
        "profundidad": circuit.depth(),
        "huella_sha": text_fingerprint(text),
        "huella_wl": "",
        "duplicado_de": "",
        "clase": label,
        "exclusion": LABELLED_VARIANT,
    }


def build_labelled_variants(config: DatasetConfig, processes: int) -> pd.DataFrame:
    """F7 for every raw ``alg`` circuit with a usable class; returns the new metadata rows."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    canonical = metadata[metadata["ruta_canonical"] != ""]
    class_of = canonical.drop_duplicates("algoritmo").set_index("algoritmo")["clase"]
    raw = metadata[(metadata["variante_comp"] == "") & (metadata["nivel"] == "alg")]
    raw = raw[raw["algoritmo"].isin(class_of.index)]
    tasks = [(row, class_of[row["algoritmo"]]) for row in raw.to_dict("records")]
    results = run_tasks(partial(_labelled_task, config=config), tasks, processes, "F7")
    rows = pd.DataFrame([r for r in results if r is not None], columns=METADATA_COLUMNS)
    if len(rows):
        upsert(config.rutas.metadata, rows, "id", METADATA_COLUMNS)
    logger.info("%d labelled variants written (%d too large)", len(rows), len(tasks) - len(rows))
    return rows


def _labelled_task(task: tuple[dict, str], config: DatasetConfig) -> dict | None:
    """Picklable wrapper of :func:`_labelled_row`."""
    row, label = task
    return _labelled_row(config, row, label)


# --------------------------------------------------------------------------------------------
# F3-F6
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitFormats:
    """Per-circuit outputs collected by the main process."""

    id: str
    features: dict[str, float]
    tokens: list[str]
    nodes: int


def _with_targets(graph, circuit_id: str, label: tuple[int, int]):
    """Attach the identifier and the class and family targets to a graph."""
    graph.y_algoritmo = torch.tensor([label[0]], dtype=torch.long)
    graph.y_familia = torch.tensor([label[1]], dtype=torch.long)
    graph.id = circuit_id
    return graph


def circuit_formats(
    row: dict[str, str], config: DatasetConfig, vocabulary: LabelVocabulary
) -> CircuitFormats:
    """Compute F3 and F6 and write the F4 and F5 graphs of one canonical circuit."""
    circuit = read_qpy(qpy_path(config.rutas.raiz / row["ruta_canonical"]))
    gates = gate_sequence(circuit)
    label = vocabulary.targets(row)
    dag = _with_targets(grafo_dag.dag_graph(gates, circuit.num_qubits), row["id"], label)
    qubits = _with_targets(grafo_cubits.qubit_graph(gates, circuit.num_qubits), row["id"], label)
    grafo_dag.save_graph(dag, config.rutas.graphs / DAG_DIR / f"{row['id']}.pt")
    grafo_dag.save_graph(qubits, config.rutas.graphs / QUBIT_DIR / f"{row['id']}.pt")
    features = {"id": row["id"], **caracteristicas.features(gates, circuit.num_qubits)}
    return CircuitFormats(row["id"], features, tokens.tokens(gates), dag.num_nodes)


def _write_index(directory: Path, ids: list[str]) -> int:
    """Write ``indice.csv`` and delete graph files not listed in it; return files removed."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids}).to_csv(directory / grafo_dag.INDEX_FILE, index=False)
    keep = {f"{circuit_id}.pt" for circuit_id in ids}
    stale = [path for path in directory.glob("*.pt") if path.name not in keep]
    for path in stale:
        path.unlink()
    return len(stale)


def _write_outputs(
    config: DatasetConfig, results: list[CircuitFormats], vocabulary: LabelVocabulary
) -> None:
    """Write the aggregated F3 and F6 files, the graph indexes and the label vocabulary."""
    results = sorted(results, key=lambda r: r.id)
    ids = [r.id for r in results]
    config.rutas.features.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.features for r in results]).to_parquet(
        config.rutas.features / FEATURES_FILE, index=False
    )
    lines = (json.dumps({"id": r.id, "tokens": r.tokens}) for r in results)
    write_text_atomic("\n".join(lines) + "\n", config.rutas.tokens / TOKENS_FILE)
    counts = tokens.vocabulary(r.tokens for r in results)
    write_text_atomic(json.dumps(counts), config.rutas.tokens / TOKEN_VOCABULARY_FILE)
    removed = sum(_write_index(config.rutas.graphs / d, ids) for d in (DAG_DIR, QUBIT_DIR))
    labels = vocabulary.as_json()
    write_text_atomic(
        json.dumps(labels, ensure_ascii=False, indent=2),
        config.rutas.graphs / LABEL_VOCABULARY_FILE,
    )
    logger.info("formats written for %d circuits (%d stale graph files removed)", len(ids), removed)


# --------------------------------------------------------------------------------------------
# Consistency check and whole step
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsistencyRow:
    """Sizes of one circuit in every format."""

    id: str
    gates_f2: int
    nodes_f4: int
    tokens_f6: int
    labels_f7: int | None

    @property
    def consistent(self) -> bool:
        """True if every available format has the same number of elements."""
        sizes = {self.gates_f2, self.nodes_f4, self.tokens_f6}
        if self.labels_f7 is not None:
            sizes.add(self.labels_f7)
        return len(sizes) == 1


def check_consistency(config: DatasetConfig, sample: int = CHECK_SAMPLE) -> list[ConsistencyRow]:
    """Compare the formats of ``sample`` random labelled circuits (seeded) and of ``sample``
    random training circuits."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    index = pd.read_csv(config.rutas.graphs / DAG_DIR / grafo_dag.INDEX_FILE, dtype=str)
    rows = metadata[metadata["id"].isin(index["id"])].set_index("id")
    rng = random.Random(derive_seed(config.semilla_global, "comprobacion_formatos"))
    labelled = sorted(rows.index[rows["variante_comp"] == LABELLED_VARIANT])
    usable = sorted(rows.index[rows["exclusion"] == ""])
    chosen = rng.sample(labelled, min(sample, len(labelled))) + rng.sample(
        usable, min(sample, len(usable))
    )
    token_lengths = {}
    with (config.rutas.tokens / TOKENS_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["id"] in chosen:
                token_lengths[record["id"]] = len(record["tokens"])
    report = []
    for circuit_id in chosen:
        graph = grafo_dag.load_graph(config.rutas.graphs / DAG_DIR / f"{circuit_id}.pt")
        labels_file = config.rutas.gate_labels / f"{circuit_id}.json"
        labels = None
        if labels_file.is_file():
            labels = len(json.loads(labels_file.read_text(encoding="utf-8"))["etiquetas"])
        report.append(
            ConsistencyRow(
                circuit_id,
                int(rows.loc[circuit_id, "n_puertas"]),
                graph.num_nodes,
                token_lengths[circuit_id],
                labels,
            )
        )
    return report


@dataclass(frozen=True)
class FormatsReport:
    """Outcome of the formats step."""

    circuits: int
    labelled: int
    classes: tuple[str, ...]
    families: tuple[str, ...]
    consistency: list[ConsistencyRow]


def build_formats(
    config: DatasetConfig, reserved: Sequence[str], *, processes: int = 1
) -> FormatsReport:
    """Run the whole formats step (see the module docstring)."""
    labelled = build_labelled_variants(config, processes)
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    canonical = metadata[metadata["ruta_canonical"] != ""]
    usable = canonical[canonical["exclusion"] == ""]
    selected = canonical[canonical["exclusion"].isin(["", LABELLED_VARIANT])]
    vocabulary = label_vocabulary(config, reserved, list(usable["clase"]))
    worker = partial(circuit_formats, config=config, vocabulary=vocabulary)
    results = run_tasks(worker, selected.to_dict("records"), processes, "formatos")
    _write_outputs(config, results, vocabulary)
    return FormatsReport(
        circuits=len(results),
        labelled=len(labelled),
        classes=vocabulary.classes,
        families=vocabulary.families,
        consistency=check_consistency(config),
    )
