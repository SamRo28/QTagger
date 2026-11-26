"""Data contract with QTaggerDemo: pydantic models of the exported files (schema 0.1.0).

Two kinds of files, each with its JSON Schema in ``esquema/demo/`` (written by
:func:`write_schemas`, ``qcclass esquema-demo``):

* ``circuito.schema.json`` (:class:`CircuitExport`): one file per gallery circuit, with the
  whole path of the laboratory: the F1 circuit with its blocks, the flattened F2 circuit, the
  F4 graph, the model traces (in a binary file), the prediction, the explanation and the
  unknown check.
* ``global.schema.json`` (:class:`GlobalExport`): classes, families, the 2D reference map,
  confusion matrices, baseline ladder, training curves, Optuna history, ablations, decisions,
  dataset statistics, measurements and versions.

Binary traces (``trazas.bin``): float16 little-endian tensors stored one after the other in
row-major order; :class:`TraceTensor` gives the name, byte offset and shape of each one.

Conventions: keys in English; class, family and block names as in the dataset. Every file
carries ``schema_version`` and ``provisional`` (``true`` for the phase 11b export).
Sections of the global file that do not exist yet are :class:`Section` with
``status = "pending"`` and ``data = null``.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"
SCHEMA_DIR = Path("esquema") / "demo"
CIRCUIT_SCHEMA_FILE = "circuito.schema.json"
GLOBAL_SCHEMA_FILE = "global.schema.json"
TRACE_DTYPE = "float16"
TRACE_BYTE_ORDER = "little"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------------------------


class LayerDescription(_Model):
    """One layer of the model (``CircuitGNN.describe``)."""

    name: str
    type: str
    inputs: int
    outputs: int
    parameters: int


class Architecture(_Model):
    """Configuration, layers and parameters of the model."""

    config: dict[str, Any]
    layers: list[LayerDescription]
    parameters: int


class Versions(_Model):
    """Versions behind an export."""

    schema_version: str = SCHEMA_VERSION
    qcclass_commit: str
    code_dirty: bool
    dataset: str
    dataset_fingerprint: str
    libraries: dict[str, str]


class ModelInfo(_Model):
    """The model that produced the traces, predictions and explanations."""

    name: str
    description: str
    fingerprint: str = Field(description="Fingerprint of the training run (run_identity).")
    training_code: str = Field(description="Version of the training code (source hash).")
    cv_block: int = Field(description="Cross-validation block held out when training.")
    device: str
    precision: str
    epochs: int
    f1_macro_algorithm: float = Field(description="Macro F1 on the held-out block.")
    f1_macro_family: float
    architecture: Architecture
    node_features: list[str]
    edge_features: list[str] = Field(description="Edge attributes the model reads.")


class Selection(_Model):
    """Why a circuit is in the gallery."""

    role: Literal["main", "family", "error", "unknown"]
    reason: str


# --------------------------------------------------------------------------------------------
# Circuit file
# --------------------------------------------------------------------------------------------


class CircuitInfo(_Model):
    """Labels and size of the circuit."""

    id: str = Field(description="Dataset id of the canonical circuit (or of the generated one).")
    group_id: str | None
    algorithm: str
    family: str | None
    class_label: str | None = Field(description="Trainable class; null if family only/unknown.")
    level: str
    num_qubits: int
    depth: int
    num_gates: int
    source: Literal["mqt", "propio", "generado"]
    cv_block: int | None = Field(description="Block of its group (null if generated).")
    canonical_variant: str


class Block(_Model):
    """A top-level instruction of F1 (a block of the circuit)."""

    id: int = Field(description="Index of the instruction in F1.")
    name: str
    qubits: list[int]
    start_column: int
    end_column: int
    num_canonical_gates: int


class Instruction(_Model):
    """An instruction of F1 placed in its drawing column."""

    index: int
    name: str
    qubits: list[int]
    params: list[float]
    column: int
    block: int = Field(description="Id of its block (every F1 instruction is a block).")


class OriginalCircuit(_Model):
    """F1: the circuit as produced by its source, with its blocks."""

    num_columns: int
    instructions: list[Instruction]
    blocks: list[Block]


class CanonicalGate(_Model):
    """A gate of F2; ``node`` is its node index in F4 (the same order)."""

    node: int
    name: Literal["rz", "sx", "x", "cx"]
    qubits: list[int]
    column: int
    angle: float | None
    block: int | None = Field(description="Id of the F1 block it comes from (null if unknown).")


class BlockExtent(_Model):
    """Columns of F2 covered by the gates of one block (to animate the flattening)."""

    block: int
    start_column: int
    end_column: int


class CanonicalCircuit(_Model):
    """F2: the flattened circuit (rz, sx, x, cx)."""

    num_columns: int
    gates: list[CanonicalGate]
    block_extents: list[BlockExtent]
    blocks_known: bool = Field(description="True if every gate knows its F1 block.")


class Edges(_Model):
    """Stored F4 edges ``source -> target`` (time direction), with the roles of the qubit."""

    source: list[int]
    target: list[int]
    roles: list[list[float]] = Field(description="[origen_control, origen_destino, ...]")


class Graph(_Model):
    """F4: gate dependency graph."""

    num_nodes: int
    num_edges: int
    direction: Literal["forward"] = "forward"
    model_uses_reverse_edges: bool = Field(
        description="The model also reads every edge backwards (attribute atras = 1)."
    )
    role_names: list[str]
    edges: Edges
    node_feature_names: list[str]
    node_features_tensor: str = Field(description="Name of the [N, F] tensor in the binary.")


class TraceTensor(_Model):
    """A tensor of ``trazas.bin``."""

    name: str
    offset: int = Field(description="Byte offset in the binary.")
    shape: list[int]


class LayerCoordinates(_Model):
    """2D position of every node in one stage of the model."""

    name: str
    explained_variance: list[float]
    points: list[list[float]] = Field(description="[N][2]")


class AttentionTrace(_Model):
    """Attention weights (GATv2 per layer and/or the pooling gate)."""

    layers: list[dict[str, Any]] | None
    pooling: list[float] | None


class Trace(_Model):
    """Model traces: index of the binary and 2D coordinates per stage."""

    binary: str
    dtype: Literal["float16"] = TRACE_DTYPE
    byte_order: Literal["little"] = TRACE_BYTE_ORDER
    tensors: list[TraceTensor]
    stages: list[str] = Field(description="Node stages in order: entrada, capa_1, ..., JK.")
    coordinates_method: str
    coordinates: list[LayerCoordinates]
    attention: AttentionTrace | None


class HeadPrediction(_Model):
    """Probabilities of one head."""

    classes: list[str]
    probabilities: list[float]
    predicted: str
    true: str | None
    correct: bool | None


class Prediction(_Model):
    """Both heads."""

    algorithm: HeadPrediction
    family: HeadPrediction


class Importance(_Model):
    """Mean and standard deviation over models (``std`` null with a single model)."""

    mean: list[float]
    std: list[float] | None


class BlockImportance(_Model):
    """Importance of one F1 block (sum over its gates)."""

    block: int
    name: str
    mean: float
    std: float | None


class Explanation(_Model):
    """Importance of every gate for the predicted class."""

    method: str
    target_class: str
    steps: int
    baseline: str
    num_models: int
    gates: Importance
    blocks: list[BlockImportance]
    completeness_error: float = Field(
        description="|sum of attributions - (f(x) - f(baseline))| relative to |f(x) - f(b)|."
    )


class ClassDistance(_Model):
    """Mahalanobis distance to one class."""

    class_name: str
    distance: float


class UnknownCheck(_Model):
    """Is the circuit unknown? Distances to the training classes and the threshold."""

    method: str
    reference: str
    distances: list[ClassDistance]
    nearest_class: str
    min_distance: float
    threshold: float
    acceptance: float = Field(description="Share of known validation circuits accepted.")
    verdict: Literal["known", "unknown"]
    map_position: list[float] = Field(description="[x, y] in the global reference map.")


class CircuitExport(_Model):
    """One gallery circuit (``circuito.json``)."""

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    provisional: bool
    slug: str
    selection: Selection
    model: ModelInfo
    circuit: CircuitInfo
    original: OriginalCircuit
    canonical: CanonicalCircuit
    graph: Graph
    trace: Trace
    prediction: Prediction
    explanation: Explanation
    unknown: UnknownCheck


# --------------------------------------------------------------------------------------------
# Global file
# --------------------------------------------------------------------------------------------


class Section[T](_Model):
    """A global section: ``available``, ``provisional`` (from the provisional model) or
    ``pending`` (does not exist yet; ``data`` null)."""

    status: Literal["available", "provisional", "pending"]
    note: str | None = None
    data: T | None = None


class ClassInfo(_Model):
    """A label of the dataset."""

    name: str
    family: str | None
    status: Literal["trainable", "family_only", "reserved", "unknown_without_family"]


class FamilyInfo(_Model):
    """A family and its algorithms."""

    name: str
    classes: list[str]
    algorithms: list[str]


class MapPoint(_Model):
    """A circuit in the 2D reference map."""

    id: str
    x: float
    y: float
    class_label: str | None
    family: str
    predicted: str


class ReferenceMap(_Model):
    """2D projection of the summary vectors of known circuits."""

    method: str
    source: str
    explained_variance: list[float]
    points: list[MapPoint]


class ConfusionMatrix(_Model):
    """Rows = true class, columns = predicted class."""

    labels: list[str]
    matrix: list[list[int]]


class Confusions(_Model):
    """Confusion matrices of both heads."""

    source: str
    algorithm: ConfusionMatrix
    family: ConfusionMatrix


class BaselineRow(_Model):
    """A model of the baseline ladder (cross-validation, never test)."""

    rung: str
    classifier: str
    target: str
    cv_f1_macro: float
    cv_f1_macro_std: float
    oof_f1_macro: float
    oof_f1_macro_ci: list[float]


class TrainingPoint(_Model):
    """An epoch of a training curve."""

    epoch: int
    loss: float
    loss_algorithm: float
    loss_family: float
    learning_rate: float
    f1_macro_algorithm: float | None
    f1_macro_family: float | None


class TrainingCurve(_Model):
    """Training curve of one run."""

    run: str
    points: list[TrainingPoint]


class Decision(_Model):
    """A row of the decision log (docs/DECISIONES.md)."""

    date: str
    phase: str
    decision: str
    reason: str
    alternative: str


class DatasetStats(_Model):
    """Counts of the usable canonical circuits."""

    version: str
    circuits: int
    groups: int
    by_level: dict[str, int]
    by_family: dict[str, int]
    by_class: dict[str, int]
    qubits: dict[str, float]
    gates: dict[str, float]


class MachineMeasurement(_Model):
    """Speed of a machine (results/medicion)."""

    name: str
    device: str
    mixed_precision: bool
    seconds_per_epoch: float | None
    hours_per_epoch: float | None


class GalleryEntry(_Model):
    """A circuit file of the export."""

    slug: str
    role: Literal["main", "family", "error", "unknown"]
    path: str
    algorithm: str
    family: str | None


class GlobalExport(_Model):
    """Global views (``global.json``)."""

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    provisional: bool
    generated_at: str
    versions: Versions
    model: ModelInfo
    classes: list[ClassInfo]
    families: list[FamilyInfo]
    gallery: list[GalleryEntry]
    selection_criteria: list[str]
    reference_map: Section[ReferenceMap]
    confusion: Section[Confusions]
    baseline_ladder: Section[list[BaselineRow]]
    training_curves: Section[list[TrainingCurve]]
    optuna_history: Section[list[dict[str, Any]]]
    ablations: Section[list[dict[str, Any]]]
    decisions: Section[list[Decision]]
    dataset: Section[DatasetStats]
    measurements: Section[list[MachineMeasurement]]


SCHEMAS = {CIRCUIT_SCHEMA_FILE: CircuitExport, GLOBAL_SCHEMA_FILE: GlobalExport}


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of a model, with its id and version."""
    schema = model.model_json_schema()
    name = next(k for k, v in SCHEMAS.items() if v is model)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"qtagger/demo/{SCHEMA_VERSION}/{name}",
        "x-schema-version": SCHEMA_VERSION,
        **schema,
    }


def schema_text(model: type[BaseModel]) -> str:
    """Text of a schema file (stable: sorted keys are not used, pydantic order is)."""
    return json.dumps(json_schema(model), indent=2, ensure_ascii=False) + "\n"


def write_schemas(directory: Path) -> list[Path]:
    """Write every schema file to ``directory``; returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in SCHEMAS.items():
        path = directory / name
        path.write_text(schema_text(model), encoding="utf-8")
        paths.append(path)
    return paths
