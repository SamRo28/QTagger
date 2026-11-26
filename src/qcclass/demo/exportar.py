"""``qcclass exportar-demo --provisional``: data of QTaggerDemo from the phase 11 reference.

Only the train + val pool of ``aleatorio`` is read (through ``load_pool``, which never touches
the test split) plus circuits generated on the spot; never test, never the unknown set.

Steps:

1. Load the finished reference run of ``modelo.experimento`` on ``modelo.bloque``.
2. Summary vectors and probabilities of the training part (after the cap) and of the
   held-out block of that cross-validation block.
3. Mahalanobis detector (training part) with the threshold that accepts ``aceptacion`` of
   the known held-out circuits; 2D reference map = PCA of the held-out summaries.
4. Gallery, all from the groups of the held-out block, which the model never saw, using
   their ``etiquetada`` variant (F2 built block by block, so every gate knows its F1 block):
   the main circuit (``galeria.principal``), one per family, one error of the network and a
   random circuit generated now (not part of the dataset) as the unknown example. The
   criteria are written to the global file.
5. One ``circuitos/<slug>/circuito.json`` + ``trazas.bin`` per gallery circuit and
   ``global.json``; every file is validated against the schema and its size checked against
   the limits of the configuration (a warning, the limits are a guide).
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from sklearn.decomposition import PCA

from qcclass.circuitos import qpy_path, read_qpy, write_text_atomic
from qcclass.config import ConfigError, DatasetConfig, SplitsConfig, load_experiment_config
from qcclass.demo import esquema as s
from qcclass.demo.desconocidos import Mahalanobis
from qcclass.demo.disposicion import Expanded, block_extents, blocks_of, check_same_gates, columns
from qcclass.demo.modelo import (
    TrainedModel,
    graph_input,
    integrated_gradients,
    load_trained,
    run_traced,
    summaries,
)
from qcclass.demo.trazas import aligned_pca, write_binary
from qcclass.entrenamiento.bloques import CrossValidationPool, fold_training, load_pool
from qcclass.entrenamiento.gnn import RESULT_FILE, complete_model_settings
from qcclass.entrenamiento.gnn_experimento import OUTPUT_DIR, load_gnn_experiment
from qcclass.evaluacion.metricas import confusion
from qcclass.formatos.comun import gate_sequence
from qcclass.formatos.construir import DAG_DIR
from qcclass.formatos.etiquetas_puerta import LABELLED_VARIANT
from qcclass.formatos.grafo_dag import EDGE_FEATURES, NODE_FEATURES, dag_graph, load_graph
from qcclass.metadatos import METADATA_COLUMNS, read_table
from qcclass.modelos.aristas import EDGE_FEATURES as MODEL_EDGE_FEATURES
from qcclass.seguimiento.mlflow_cliente import git_state
from qcclass.semillas import derive_seed
from qcclass.versionado import library_versions, version_stamp

logger = logging.getLogger(__name__)

REQUIRED = ("nombre", "semilla", "salida", "modelo", "galeria", "explicacion", "desconocidos")
CIRCUITS_DIR = "circuitos"
CIRCUIT_FILE = "circuito.json"
BINARY_FILE = "trazas.bin"
GLOBAL_FILE = "global.json"
GENERATED = "aleatorio_generado"
DECISION_ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|(.*)\|\s*$")


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoExportConfig:
    """``configs/experimentos/demo_provisional.yaml``."""

    name: str
    seed: int
    output: Path
    experiment: Path
    block: int
    run_directory: Path | None
    main_algorithm: str
    main_qubits: tuple[int, ...]
    min_gates: int
    max_gates: int
    readable_qubits: int
    readable_min_gates: int
    readable_max_gates: int
    target_gates: int
    family_algorithms: dict[str, tuple[str, ...]]
    span: bool
    random_qubits: int
    random_depth: int
    random_max_operands: int
    ig_steps: int
    acceptance: float
    max_json_mb: float
    max_binary_mb: float
    raw: dict[str, Any] = field(default_factory=dict)


def load_demo_config(path: Path | str, root: Path) -> DemoExportConfig:
    """Load the export configuration (relative paths are relative to ``root``).

    Raises:
        ConfigError: if a key is missing or a value is invalid.
    """
    data = load_experiment_config(path, REQUIRED)
    try:
        model, gallery = data["modelo"], data["galeria"]
        random = gallery.get("aleatorio") or {}
        limits = data.get("limites") or {}
        directory = model.get("directorio")
        return DemoExportConfig(
            name=str(data["nombre"]),
            seed=int(data["semilla"]),
            output=root / str(data["salida"]),
            experiment=root / str(model["experimento"]),
            block=int(model["bloque"]),
            run_directory=root / directory if directory else None,
            main_algorithm=str(gallery["principal"]["algoritmo"]),
            main_qubits=tuple(int(q) for q in gallery["principal"]["cubits"]),
            min_gates=int(gallery["candidatos"]["min_puertas"]),
            max_gates=int(gallery["candidatos"]["max_puertas"]),
            readable_qubits=int(gallery["legibilidad"]["max_cubits"]),
            readable_min_gates=int(gallery["legibilidad"]["min_puertas"]),
            readable_max_gates=int(gallery["legibilidad"]["max_puertas"]),
            target_gates=int(gallery["legibilidad"]["puertas_objetivo"]),
            family_algorithms={
                str(f): tuple(str(a) for a in names)
                for f, names in (gallery.get("algoritmos_por_familia") or {}).items()
            },
            span=bool(gallery.get("columnas_por_rango", True)),
            random_qubits=int(random.get("cubits", 4)),
            random_depth=int(random.get("profundidad", 8)),
            random_max_operands=int(random.get("max_operandos", 2)),
            ig_steps=int(data["explicacion"]["pasos"]),
            acceptance=float(data["desconocidos"]["aceptacion"]),
            max_json_mb=float(limits.get("json_mb", 2)),
            max_binary_mb=float(limits.get("binario_mb", 5)),
            raw=data,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


def find_run(config: DatasetConfig, demo: DemoExportConfig) -> Path:
    """Directory of the finished reference run (``modelo.directorio`` or the newest finished
    run of the experiment on the block with the same model and number of epochs).

    Raises:
        FileNotFoundError: if there is none.
    """
    if demo.run_directory is not None:
        return demo.run_directory
    experiment = load_gnn_experiment(demo.experiment)
    model = complete_model_settings(experiment.model)
    root = config.rutas.resultados / OUTPUT_DIR / experiment.name
    found = []
    for path in root.glob(f"*/bloque_{demo.block}/{RESULT_FILE}"):
        identity = json.loads(path.read_text(encoding="utf-8"))["identidad"]
        if (
            identity["modelo"] == model
            and identity["entrenamiento"]["epocas"] == experiment.training.epocas
        ):
            found.append(path.parent)
    if not found:
        raise FileNotFoundError(
            f"no finished run of {experiment.name} on block {demo.block} in {root}: "
            f"train it with qcclass entrenar-gnn --experimento {demo.experiment.name}"
        )
    return max(found, key=lambda p: (p / RESULT_FILE).stat().st_mtime)


# --------------------------------------------------------------------------------------------
# Data of the pool
# --------------------------------------------------------------------------------------------


@dataclass
class PoolView:
    """Summaries and predictions of the training part and of the held-out block."""

    pool: CrossValidationPool
    train: pd.DataFrame
    held: pd.DataFrame
    train_vectors: np.ndarray
    held_vectors: np.ndarray
    held_algorithm: np.ndarray
    held_family: np.ndarray


def _stored_graphs(config: DatasetConfig, ids: list[str], trained: TrainedModel) -> list:
    directory = config.rutas.graphs / DAG_DIR
    return [trained.transform(load_graph(directory / f"{i}.pt")) for i in ids]


def pool_view(
    config: DatasetConfig, splits: SplitsConfig, trained: TrainedModel, demo: DemoExportConfig
) -> PoolView:
    """Step 2 of the module docstring."""
    pool = load_pool(config, splits, demo.name)
    fold = fold_training(pool, demo.block, splits)
    train = pool.rows.iloc[fold.train_index].reset_index(drop=True)
    held = pool.rows.iloc[fold.held_out_index].reset_index(drop=True)
    logger.info("summaries of %d training and %d held-out circuits", len(train), len(held))
    train_vectors, _, _ = summaries(trained, _stored_graphs(config, list(train["id"]), trained))
    held_vectors, held_algorithm, held_family = summaries(
        trained, _stored_graphs(config, list(held["id"]), trained)
    )
    return PoolView(pool, train, held, train_vectors, held_vectors, held_algorithm, held_family)


# --------------------------------------------------------------------------------------------
# Gallery
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """A gallery circuit before it is exported."""

    slug: str
    role: str
    reason: str
    row: dict[str, Any] | None  # metadata row (None for the generated circuit)
    original: QuantumCircuit


def _candidates(
    config: DatasetConfig, view: PoolView, trained: TrainedModel, demo: DemoExportConfig
) -> pd.DataFrame:
    """``etiquetada`` variants of the held-out groups with their prediction."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    groups = set(view.held["group_id"])
    rows = metadata[
        (metadata["variante_comp"] == LABELLED_VARIANT) & metadata["group_id"].isin(groups)
    ].copy()
    rows["n"] = rows["n_puertas"].astype(int)
    rows = rows[(rows["n"] >= demo.min_gates) & (rows["n"] <= demo.max_gates)]
    rows = rows.sort_values(["n", "id"]).reset_index(drop=True)
    _, algorithm, _ = summaries(trained, _stored_graphs(config, list(rows["id"]), trained))
    classes = view.pool.class_names("algoritmo")
    rows["pred"] = [classes[i] for i in algorithm.argmax(axis=1)]
    rows["p_pred"] = algorithm.max(axis=1)
    rows["correct"] = rows["pred"] == rows["clase"]
    return rows


def choose_gallery(
    config: DatasetConfig, view: PoolView, trained: TrainedModel, demo: DemoExportConfig
) -> tuple[list[Choice], list[str]]:
    """Gallery circuits and the criteria used (see the module docstring)."""
    rows = _candidates(config, view, trained, demo)
    reserved = set(view.pool.vocabulary.reserved) | set(view.pool.vocabulary.unknown_without_family)
    trainable = rows[(rows["clase"] != "") & ~rows["algoritmo"].isin(reserved)]
    readable = trainable[
        (trainable["n_qubits"].astype(int) <= demo.readable_qubits)
        & (trainable["n"] >= demo.readable_min_gates)
        & (trainable["n"] <= demo.readable_max_gates)
    ]
    limits = (
        f"at most {demo.readable_qubits} qubits and {demo.readable_min_gates}-"
        f"{demo.readable_max_gates} F2 gates, the closest to {demo.target_gates} gates"
    )
    criteria = [
        f"Every dataset circuit comes from a group of the held-out block {demo.block}, never "
        "seen by the model, and is its 'etiquetada' variant (F2 decomposed block by block).",
        f"Candidates: a trainable class (never a reserved, unknown or family-only algorithm) "
        f"and between {demo.min_gates} and {demo.max_gates} F2 gates.",
    ]
    chosen: list[Choice] = []

    def add(row: pd.Series, role: str, reason: str) -> None:
        raw = read_qpy(qpy_path(config.rutas.raiz / row["ruta_raw"]))
        slug = f"{role}_{row['algoritmo']}_{row['group_id']}"
        chosen.append(Choice(slug, role, reason, row.to_dict(), raw))

    def by_target(options: pd.DataFrame) -> pd.DataFrame:
        distance = (options["n"] - demo.target_gates).abs()
        return options.assign(_d=distance).sort_values(["_d", "id"]).drop(columns="_d")

    def pick(options: pd.DataFrame, fallback: pd.DataFrame, what: str):
        """Readable option closest to the target, correctly predicted first; without a
        readable option, the same among ``fallback`` (readability limits ignored)."""
        note = ""
        if options.empty:
            logger.warning("no readable candidate for %s: fallback", what)
            options, note = fallback, " (no readable candidate: limits ignored)"
        if options.empty:
            logger.warning("no candidate for %s", what)
            return None
        options = by_target(options)
        correct = options[options["correct"]]
        if correct.empty:
            return options.iloc[0], f"{what} closest to {demo.target_gates} gates{note}, " + (
                "none predicted correctly"
            )
        return correct.iloc[
            0
        ], f"correctly predicted {what} closest to {demo.target_gates} gates{note}"

    main = trainable[
        (trainable["algoritmo"] == demo.main_algorithm)
        & trainable["n_qubits"].astype(int).isin(demo.main_qubits)
    ].sort_values(["n", "id"])
    if main.empty:
        raise ValueError(f"no {demo.main_algorithm} with {demo.main_qubits} qubits")
    main_correct = main[main["correct"]]
    row = (main_correct if len(main_correct) else main).iloc[0]
    add(
        row,
        "main",
        f"smallest {demo.main_algorithm} with {list(demo.main_qubits)} qubits"
        + (" predicted correctly" if len(main_correct) else " (none predicted correctly)"),
    )
    criteria.append(
        f"Main: the smallest {demo.main_algorithm} with {list(demo.main_qubits)} qubits "
        "predicted correctly (the smallest one if none is)."
    )
    for family in view.pool.class_names("familia"):
        allowed = demo.family_algorithms.get(family)

        def of_family(table: pd.DataFrame, family: str = family, allowed=allowed) -> pd.DataFrame:
            keep = (table["familia"] == family) & (table["algoritmo"] != demo.main_algorithm)
            if allowed:
                keep &= table["algoritmo"].isin(allowed)
            return table[keep]

        what = f"circuit of {family}" + (f" ({', '.join(allowed)})" if allowed else "")
        if found := pick(of_family(readable), of_family(trainable), what):
            add(found[0], "family", found[1])
    restricted = "; ".join(f"{f}: {', '.join(a)}" for f, a in demo.family_algorithms.items())
    criteria.append(
        f"One per family, readable: {limits} (ties by id), correctly predicted first; not "
        f"{demo.main_algorithm} (the main one)"
        + (f"; algorithms allowed: {restricted}" if restricted else "")
        + f". Without a readable candidate, the candidate of the family closest to "
        f"{demo.target_gates} gates."
    )

    def wrong(table: pd.DataFrame) -> pd.DataFrame:
        return table[~table["correct"]].sort_values(["p_pred", "id"], ascending=[False, True])

    errors, note = wrong(readable), ""
    if errors.empty:
        errors, note = wrong(trainable), " (no readable mistake: limits ignored)"
    if errors.empty:
        criteria.append("Error: none of the candidates is mispredicted (no error circuit).")
    else:
        add(errors.iloc[0], "error", f"the most confident readable mistake of the network{note}")
        criteria.append(
            "Error: the readable mispredicted candidate with the highest wrong probability "
            "(any candidate if none is readable)."
        )
    seed = derive_seed(demo.seed, "demo", "aleatorio")
    generated = random_circuit(
        demo.random_qubits,
        demo.random_depth,
        max_operands=demo.random_max_operands,
        measure=False,
        seed=seed,
    )
    reason = (
        f"random circuit generated now (qiskit random_circuit, {demo.random_qubits} qubits, "
        f"depth {demo.random_depth}, seed {seed}); not part of the dataset"
    )
    chosen.append(Choice(f"unknown_{GENERATED}", "unknown", reason, None, generated))
    criteria.append(f"Unknown: a {reason}.")
    return chosen, criteria


# --------------------------------------------------------------------------------------------
# One circuit
# --------------------------------------------------------------------------------------------


def _head(classes: tuple[str, ...], probabilities: np.ndarray, true: str | None):
    predicted = classes[int(np.argmax(probabilities))]
    return s.HeadPrediction(
        classes=list(classes),
        probabilities=[round(float(p), 6) for p in probabilities],
        predicted=predicted,
        true=true,
        correct=None if true is None else predicted == true,
    )


def _round(points: np.ndarray, digits: int = 4) -> list[list[float]]:
    return np.round(points.astype(float), digits).tolist()


@dataclass
class Context:
    """What every circuit export needs."""

    config: DatasetConfig
    demo: DemoExportConfig
    trained: TrainedModel
    model_info: s.ModelInfo
    classes: tuple[str, ...]
    families: tuple[str, ...]
    detector: Mahalanobis
    map_pca: PCA
    blocks: dict[str, int]


def export_circuit(choice: Choice, context: Context, directory: Path) -> s.CircuitExport:
    """Write ``circuito.json`` and ``trazas.bin`` of one gallery circuit."""
    trained, demo = context.trained, context.demo
    expanded: Expanded = blocks_of(choice.original)
    row = choice.row
    if row is not None:
        stored = read_qpy(qpy_path(context.config.rutas.raiz / row["ruta_canonical"]))
        check_same_gates(gate_sequence(stored), expanded.gates)
        canonical = stored
    else:
        canonical = expanded.circuit
    gates = gate_sequence(canonical)
    num_qubits = canonical.num_qubits
    graph = graph_input(gates, num_qubits, trained.transform)
    if row is not None:  # the input built now is the stored F4 graph
        reference = trained.transform(
            load_graph(context.config.rutas.graphs / DAG_DIR / f"{row['id']}.pt")
        )
        for name in ("x", "edge_index", "edge_attr"):
            if not bool((getattr(reference, name) == getattr(graph, name)).all()):
                raise ValueError(f"{row['id']}: rebuilt graph differs from the stored one ({name})")

    # F1 and F2 layout
    f1_columns = columns([q for _, q, _ in expanded.instructions], num_qubits, span=demo.span)
    f2_columns = columns([g.qubits for g in gates], num_qubits, span=demo.span)
    extents = block_extents(expanded.block_of_gate, f2_columns)
    counts = pd.Series(expanded.block_of_gate).value_counts().to_dict()
    instructions = [
        s.Instruction(index=i, name=name, qubits=list(q), params=p, column=f1_columns[i], block=i)
        for i, (name, q, p) in enumerate(expanded.instructions)
    ]
    blocks = [
        s.Block(
            id=i,
            name=name,
            qubits=list(q),
            start_column=f1_columns[i],
            end_column=f1_columns[i],
            num_canonical_gates=int(counts.get(i, 0)),
        )
        for i, (name, q, _) in enumerate(expanded.instructions)
    ]
    canonical_part = s.CanonicalCircuit(
        num_columns=max(f2_columns) + 1 if f2_columns else 0,
        gates=[
            s.CanonicalGate(
                node=i,
                name=g.name,
                qubits=list(g.qubits),
                column=f2_columns[i],
                angle=None if g.angle is None else round(g.angle, 6),
                block=expanded.block_of_gate[i],
            )
            for i, g in enumerate(gates)
        ],
        block_extents=[
            s.BlockExtent(block=b, start_column=lo, end_column=hi)
            for b, (lo, hi) in sorted(extents.items())
        ],
        blocks_known=True,
    )

    # graph (stored direction) and traces
    stored_graph = dag_graph(gates, num_qubits)
    traces = run_traced(trained, graph)
    layers = trained.config.capas
    tensors: dict[str, np.ndarray] = {"atributos_entrada": graph.x.numpy()}
    stages = {"atributos_entrada": graph.x.numpy(), "entrada": traces["nodos"][0]}
    tensors["entrada"] = traces["nodos"][0]
    for k in range(1, layers + 1):
        tensors[f"capa_{k}"] = traces["nodos"][k]
        stages[f"capa_{k}"] = traces["nodos"][k]
    if traces["nodo_virtual"] is not None:
        tensors["nodo_virtual"] = np.concatenate(traces["nodo_virtual"], axis=0)
    if traces["jumping_knowledge"] is not None:
        tensors["jumping_knowledge"] = traces["jumping_knowledge"]
        stages["jumping_knowledge"] = traces["jumping_knowledge"]
    tensors["resumen"] = traces["resumen"][0]
    directory.mkdir(parents=True, exist_ok=True)
    index = write_binary(tensors, directory / BINARY_FILE)
    coordinates = [
        s.LayerCoordinates(
            name=name, explained_variance=[round(v, 4) for v in variance], points=_round(points)
        )
        for name, points, variance in aligned_pca(stages, seed=demo.seed)
    ]
    attention = None
    if traces["atencion_capas"] or traces["atencion_agregacion"] is not None:
        attention = s.AttentionTrace(
            layers=[
                {"edge_index": a["edge_index"].astype(int).tolist(), "pesos": _round(a["pesos"])}
                for a in traces["atencion_capas"]
            ]
            if traces["atencion_capas"]
            else None,
            pooling=None
            if traces["atencion_agregacion"] is None
            else np.round(traces["atencion_agregacion"], 5).tolist(),
        )

    # prediction, explanation, unknown
    p_alg, p_fam = traces["prob_algoritmo"][0], traces["prob_familia"][0]
    true_class = (row or {}).get("clase") or None
    true_family = (row or {}).get("familia") or None
    prediction = s.Prediction(
        algorithm=_head(context.classes, p_alg, true_class),
        family=_head(context.families, p_fam, true_family),
    )
    target = int(np.argmax(p_alg))
    importance, error = integrated_gradients(trained, graph, target, steps=demo.ig_steps)
    per_block = pd.Series(importance).groupby(expanded.block_of_gate).sum()
    explanation = s.Explanation(
        method="integrated_gradients",
        target_class=context.classes[target],
        steps=demo.ig_steps,
        baseline="zeros (every node attribute set to 0; edges unchanged)",
        num_models=1,
        gates=s.Importance(mean=[round(float(v), 6) for v in importance], std=None),
        blocks=[
            s.BlockImportance(
                block=int(b), name=expanded.instructions[b][0], mean=float(v), std=None
            )
            for b, v in per_block.items()
        ],
        completeness_error=round(error, 6),
    )
    summary = traces["resumen"]
    distances = context.detector.distances(summary)[0]
    nearest = int(np.argmin(distances))
    unknown = s.UnknownCheck(
        method="mahalanobis",
        reference=f"class means and Ledoit-Wolf covariance of the training part of block "
        f"{demo.block}; threshold accepting {demo.acceptance:.0%} of its held-out circuits",
        distances=[
            s.ClassDistance(class_name=c, distance=round(float(d), 4))
            for c, d in zip(context.detector.classes, distances, strict=True)
        ],
        nearest_class=context.detector.classes[nearest],
        min_distance=round(float(distances[nearest]), 4),
        threshold=round(context.detector.threshold, 4),
        acceptance=demo.acceptance,
        verdict="unknown" if distances[nearest] > context.detector.threshold else "known",
        map_position=[round(float(v), 4) for v in context.map_pca.transform(summary)[0]],
    )

    edges = stored_graph.edge_index.numpy()
    info = s.CircuitInfo(
        id=row["id"] if row else GENERATED,
        group_id=row["group_id"] if row else None,
        algorithm=row["algoritmo"] if row else GENERATED,
        family=true_family,
        class_label=true_class,
        level=row["nivel"] if row else "generado",
        num_qubits=num_qubits,
        depth=canonical.depth(),
        num_gates=len(gates),
        source=row["origen"] if row else "generado",
        cv_block=context.blocks.get(row["group_id"]) if row else None,
        canonical_variant=LABELLED_VARIANT if row else "por_bloques",
    )
    export = s.CircuitExport(
        provisional=True,
        slug=choice.slug,
        selection=s.Selection(role=choice.role, reason=choice.reason),
        model=context.model_info,
        circuit=info,
        original=s.OriginalCircuit(
            num_columns=max(f1_columns) + 1 if f1_columns else 0,
            instructions=instructions,
            blocks=blocks,
        ),
        canonical=canonical_part,
        graph=s.Graph(
            num_nodes=len(gates),
            num_edges=int(edges.shape[1]),
            model_uses_reverse_edges=trained.transform.variant.reverse,
            role_names=list(EDGE_FEATURES),
            edges=s.Edges(
                source=edges[0].tolist(),
                target=edges[1].tolist(),
                roles=stored_graph.edge_attr.numpy().astype(int).tolist(),
            ),
            node_feature_names=list(NODE_FEATURES),
            node_features_tensor="atributos_entrada",
        ),
        trace=s.Trace(
            binary=BINARY_FILE,
            tensors=index,
            stages=list(stages),
            coordinates_method="pca_2d_per_stage_aligned_with_previous_by_orthogonal_procrustes",
            coordinates=coordinates,
            attention=attention,
        ),
        prediction=prediction,
        explanation=explanation,
        unknown=unknown,
    )
    write_text_atomic(export.model_dump_json(), directory / CIRCUIT_FILE)
    return export


# --------------------------------------------------------------------------------------------
# Global file
# --------------------------------------------------------------------------------------------


def model_info(trained: TrainedModel, demo: DemoExportConfig) -> s.ModelInfo:
    """Description of the model of the export."""
    result, identity = trained.result, trained.result["identidad"]
    architecture = trained.architecture
    return s.ModelInfo(
        name=f"{load_gnn_experiment(demo.experiment).name}.bloque{demo.block}",
        description="Phase 11 reference (provisional model of the demo export)",
        fingerprint=result["huella"],
        training_code=identity["codigo"],
        cv_block=int(result["bloque"]),
        device=result["dispositivo"],
        precision=result["precision_mixta"],
        epochs=int(identity["entrenamiento"]["epocas"]),
        f1_macro_algorithm=float(result["f1_macro_algoritmo"]),
        f1_macro_family=float(result["f1_macro_familia"]),
        architecture=s.Architecture(
            config=architecture["configuracion"],
            layers=[
                s.LayerDescription(
                    name=layer["nombre"],
                    type=layer["tipo"],
                    inputs=layer["entrada"],
                    outputs=layer["salida"],
                    parameters=layer["parametros"],
                )
                for layer in architecture["capas"]
            ],
            parameters=architecture["parametros"],
        ),
        node_features=list(NODE_FEATURES),
        edge_features=list(MODEL_EDGE_FEATURES),
    )


def _classes(config: DatasetConfig, pool: CrossValidationPool):
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    usable = metadata[(metadata["ruta_canonical"] != "")]
    family_of = usable.drop_duplicates("algoritmo").set_index("algoritmo")["familia"].to_dict()
    vocabulary = pool.vocabulary
    classes = [
        s.ClassInfo(name=c, family=family_of.get(c), status="trainable") for c in vocabulary.classes
    ]
    family_only = sorted(
        set(usable.loc[(usable["clase"] == "") & (usable["exclusion"] == ""), "algoritmo"])
        - set(vocabulary.reserved)
        - set(vocabulary.unknown_without_family)
    )
    classes += [
        s.ClassInfo(name=a, family=family_of.get(a), status="family_only") for a in family_only
    ]
    classes += [
        s.ClassInfo(name=a, family=family_of.get(a), status="reserved") for a in vocabulary.reserved
    ]
    classes += [
        s.ClassInfo(name=a, family=None, status="unknown_without_family")
        for a in vocabulary.unknown_without_family
    ]
    families = [
        s.FamilyInfo(
            name=f,
            classes=[c for c in vocabulary.classes if family_of.get(c) == f],
            algorithms=sorted(a for a, fam in family_of.items() if fam == f),
        )
        for f in vocabulary.families
    ]
    return classes, families


def _baseline(config: DatasetConfig) -> s.Section:
    path = config.rutas.resultados / "baseline" / "resumen.csv"
    if not path.is_file():
        return s.Section(status="pending", note="no baseline summary on this machine")
    table = pd.read_csv(path)
    table = table.sort_values("revision").drop_duplicates(
        ["peldano", "clasificador", "objetivo"], keep="last"
    )
    rows = [
        s.BaselineRow(
            rung=r.peldano,
            classifier=r.clasificador,
            target=r.objetivo,
            cv_f1_macro=round(r.cv_f1_macro, 5),
            cv_f1_macro_std=round(r.cv_f1_macro_desviacion, 5),
            oof_f1_macro=round(r.oof_f1_macro, 5),
            oof_f1_macro_ci=[round(r.oof_f1_macro_ic_inf, 5), round(r.oof_f1_macro_ic_sup, 5)],
        )
        for r in table.sort_values(["peldano", "objetivo", "clasificador"]).itertuples()
    ]
    return s.Section(
        status="available",
        note="Phase 10, cross-validation over train + val of 'aleatorio' (never test)",
        data=rows,
    )


def _decisions(root: Path) -> s.Section:
    path = root / "docs" / "DECISIONES.md"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := DECISION_ROW.match(line):
            cells = [c.strip() for c in match.group(2).split("|")]
            if len(cells) >= 4:
                rows.append(
                    s.Decision(
                        date=match.group(1),
                        phase=cells[0],
                        decision=cells[1],
                        reason=cells[2],
                        alternative=" | ".join(cells[3:]),
                    )
                )
    return s.Section(status="available", note="docs/DECISIONES.md", data=rows)


def _dataset(config: DatasetConfig, stamp: dict[str, str]) -> s.Section:
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    usable = metadata[(metadata["ruta_canonical"] != "") & (metadata["exclusion"] == "")]
    qubits = usable["n_qubits"].astype(int)
    gates = usable["n_puertas"].astype(int)
    classes = usable["clase"].replace("", "(solo familia)")
    return s.Section(
        status="available",
        note="usable canonical circuits of every split (counts only)",
        data=s.DatasetStats(
            version=stamp["version_dataset"],
            circuits=len(usable),
            groups=usable["group_id"].nunique(),
            by_level=usable["nivel"].value_counts().sort_index().to_dict(),
            by_family=usable["familia"].value_counts().sort_index().to_dict(),
            by_class=classes.value_counts().sort_index().to_dict(),
            qubits={
                "min": int(qubits.min()),
                "median": float(qubits.median()),
                "max": int(qubits.max()),
            },
            gates={
                "min": int(gates.min()),
                "p25": float(gates.quantile(0.25)),
                "median": float(gates.median()),
                "p75": float(gates.quantile(0.75)),
                "max": int(gates.max()),
            },
        ),
    )


def _measurements(config: DatasetConfig) -> s.Section:
    rows = []
    for path in sorted((config.rutas.resultados / "medicion").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "proyeccion" not in data or not data.get("epocas"):
            continue
        rows.append(
            s.MachineMeasurement(
                name=path.stem,
                device=data["dispositivo"],
                mixed_precision=bool(data["opciones"].get("amp")),
                seconds_per_epoch=data["epocas"][-1].get("tiempo_total_s"),
                hours_per_epoch=data["proyeccion"].get("horas_por_epoca"),
            )
        )
    note = "phase M: reference GNN, training only (results/medicion)"
    return s.Section(status="available", note=note, data=rows)


def _confusions(view: PoolView, demo: DemoExportConfig) -> s.Confusions:
    def matrix(head: str, probabilities: np.ndarray) -> s.ConfusionMatrix:
        labels = view.pool.class_names(head)
        y = view.held[f"y_{head}"].to_numpy()
        keep = y >= 0
        values = confusion(y[keep], probabilities[keep].argmax(axis=1), len(labels))
        return s.ConfusionMatrix(labels=list(labels), matrix=np.asarray(values).tolist())

    return s.Confusions(
        source=f"held-out block {demo.block} (validation of the cross-validation)",
        algorithm=matrix("algoritmo", view.held_algorithm),
        family=matrix("familia", view.held_family),
    )


def _training_curve(trained: TrainedModel) -> list[s.TrainingCurve]:
    def value(row: dict, key: str) -> float | None:
        v = row.get(key)
        return None if v is None or not np.isfinite(v) else round(float(v), 6)

    points = [
        s.TrainingPoint(
            epoch=int(h["epoca"]),
            loss=round(h["perdida"], 6),
            loss_algorithm=round(h["perdida_algoritmo"], 6),
            loss_family=round(h["perdida_familia"], 6),
            learning_rate=h["tasa_aprendizaje"],
            f1_macro_algorithm=value(h, "f1_macro_algoritmo"),
            f1_macro_family=value(h, "f1_macro_familia"),
        )
        for h in trained.result["historial"]
    ]
    return [s.TrainingCurve(run=trained.result["huella"], points=points)]


# --------------------------------------------------------------------------------------------
# Whole export
# --------------------------------------------------------------------------------------------


@dataclass
class ExportReport:
    """Outcome of an export."""

    directory: Path
    files: dict[str, int]
    over_limit: list[str]
    gallery: list[s.GalleryEntry]
    global_export: s.GlobalExport
    circuits: list[s.CircuitExport]


def export_demo(
    config: DatasetConfig, splits: SplitsConfig, demo: DemoExportConfig, *, provisional: bool
) -> ExportReport:
    """Run the export (see the module docstring)."""
    if not provisional:
        raise ValueError("only the provisional export exists before phase 18 (--provisional)")
    stamp = version_stamp(config)
    run = find_run(config, demo)
    logger.info("model: %s", run)
    trained = load_trained(run)
    view = pool_view(config, splits, trained, demo)
    classes = view.pool.class_names("algoritmo")
    families = view.pool.class_names("familia")
    detector = Mahalanobis.fit(
        view.train_vectors,
        view.train["y_algoritmo"].to_numpy(),
        classes,
        view.held_vectors,
        demo.acceptance,
    )
    map_pca = PCA(n_components=2, random_state=demo.seed).fit(view.held_vectors)
    info = model_info(trained, demo)
    blocks = (
        view.pool.rows.drop_duplicates("group_id").set_index("group_id")["bloque"].astype(int)
    ).to_dict()
    context = Context(config, demo, trained, info, classes, families, detector, map_pca, blocks)
    gallery_choices, criteria = choose_gallery(config, view, trained, demo)

    output = demo.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    circuits, gallery = [], []
    for choice in gallery_choices:
        logger.info("exporting %s", choice.slug)
        directory = output / CIRCUITS_DIR / choice.slug
        export = export_circuit(choice, context, directory)
        circuits.append(export)
        gallery.append(
            s.GalleryEntry(
                slug=choice.slug,
                role=choice.role,
                path=f"{CIRCUITS_DIR}/{choice.slug}/{CIRCUIT_FILE}",
                algorithm=export.circuit.algorithm,
                family=export.circuit.family,
            )
        )

    points = map_pca.transform(view.held_vectors)
    predicted = view.held_algorithm.argmax(axis=1)
    reference_map = s.ReferenceMap(
        method="pca_2d",
        source=f"summary vectors of the held-out block {demo.block}",
        explained_variance=[round(float(v), 4) for v in map_pca.explained_variance_ratio_],
        points=[
            s.MapPoint(
                id=r.id,
                x=round(float(p[0]), 4),
                y=round(float(p[1]), 4),
                class_label=r.clase or None,
                family=r.familia,
                predicted=classes[k],
            )
            for r, p, k in zip(view.held.itertuples(), points, predicted, strict=True)
        ],
    )
    git = git_state(config.rutas.raiz)
    classes_info, families_info = _classes(config, view.pool)
    provisional_note = "provisional: phase 11 reference model on one block"
    global_export = s.GlobalExport(
        provisional=True,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        versions=s.Versions(
            qcclass_commit=git.get("git_commit", ""),
            code_dirty=git.get("git_sucio") == "True",
            dataset=stamp["version_dataset"],
            dataset_fingerprint=stamp["huella_dataset"],
            libraries=library_versions(),
        ),
        model=info,
        classes=classes_info,
        families=families_info,
        gallery=gallery,
        selection_criteria=criteria,
        reference_map=s.Section(status="provisional", note=provisional_note, data=reference_map),
        confusion=s.Section(
            status="provisional", note=provisional_note, data=_confusions(view, demo)
        ),
        baseline_ladder=_baseline(config),
        training_curves=s.Section(
            status="provisional", note=provisional_note, data=_training_curve(trained)
        ),
        optuna_history=s.Section(status="pending", note="phase 12 search still running"),
        ablations=s.Section(status="pending", note="phase 13"),
        decisions=_decisions(config.rutas.raiz),
        dataset=_dataset(config, stamp),
        measurements=_measurements(config),
    )
    write_text_atomic(global_export.model_dump_json(), output / GLOBAL_FILE)
    files, over = check_sizes(output, demo)
    return ExportReport(output, files, over, gallery, global_export, circuits)


def check_sizes(output: Path, demo: DemoExportConfig) -> tuple[dict[str, int], list[str]]:
    """Size of every file and those above the limits (JSON / binary per circuit)."""
    files = {
        p.relative_to(output).as_posix(): p.stat().st_size
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    over = []
    for name, size in files.items():
        limit = demo.max_binary_mb if name.endswith(".bin") else demo.max_json_mb
        if name.startswith(CIRCUITS_DIR) and size > limit * 2**20:
            over.append(name)
            logger.warning("%s: %.2f MB, above the %.0f MB guide", name, size / 2**20, limit)
    return files, over


def validate_export(output: Path) -> int:
    """Validate every exported JSON against the JSON Schema files; returns files checked.

    Raises:
        jsonschema.ValidationError: on the first invalid file.
    """
    import jsonschema

    validators = {
        name: jsonschema.Draft202012Validator(s.json_schema(model))
        for name, model in s.SCHEMAS.items()
    }
    checked = 0
    for path in sorted(Path(output).rglob("*.json")):
        name = s.GLOBAL_SCHEMA_FILE if path.name == GLOBAL_FILE else s.CIRCUIT_SCHEMA_FILE
        validators[name].validate(json.loads(path.read_text(encoding="utf-8")))
        checked += 1
    return checked
