"""Calibration of the GNN on one machine (``qcclass calibrar-gnn``): memory model and real time
per configuration, with the micro-batches of ``entrenamiento.microlotes``.

1. **Memory.** For every ``diseno`` configuration, ``memoria.lotes`` training steps on
   micro-batches of ``microlote_medida`` nodes (after one warm-up step). Per node:
   ``(peak allocated - memory before the step) / nodes``. A
   :class:`~qcclass.modelos.memoria.MemoryModel` is fitted and saved as
   ``<resultados>/medicion/memoria_gnn_<machine>.json``.
2. **Time.** For every ``estimar`` configuration, the micro-batch chosen by the model, one
   warm-up batch and ``tiempo.lotes`` whole batches timed (GPU synchronised); training
   seconds per epoch = training nodes of the block / nodes per second. Evaluation: forward
   passes of ``lotes_evaluacion`` batches of the held-out block. Hours per trial =
   ``bloques * (epochs * s/epoch + evaluations * s/evaluation)`` with one evaluation every
   ``evaluar_cada`` epochs. Peaks of allocated and reserved memory are reported, so a run
   that spills over the useful memory shows up.

Output: ``<resultados>/medicion/calibracion_gnn_<machine>.json`` and an MLflow run.
"""

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qcclass.circuitos import write_text_atomic
from qcclass.config import ConfigError, DatasetConfig, SplitsConfig, load_experiment_config
from qcclass.entrenamiento.bloques import fold_training, load_pool
from qcclass.entrenamiento.gnn import (
    PoolGraphs,
    TrainingSettings,
    _loader,
    _weights,
    mixed_precision,
    precision_name,
)
from qcclass.entrenamiento.gnn_experimento import configure_cpu_threads
from qcclass.entrenamiento.microlotes import AccumulatingStep
from qcclass.formatos.construir import DAG_DIR
from qcclass.formatos.grafo_dag import NODE_FEATURES
from qcclass.modelos.aristas import EDGE_DIM, EdgeTransform
from qcclass.modelos.gnn import CircuitGNN, GNNConfig, MultiTaskLoss
from qcclass.modelos.lotes import graph_sizes
from qcclass.modelos.memoria import FEATURES, MemoryModel, model_path
from qcclass.seguimiento.mlflow_cliente import tracked_run
from qcclass.seguimiento.procedencia import start_provenance

logger = logging.getLogger(__name__)

REQUIRED = (
    "nombre",
    "experimento_mlflow",
    "semilla",
    "bloque",
    "entrenamiento",
    "memoria",
    "tiempo",
    "diseno",
    "estimar",
)
REPORT_FILE = "calibracion_gnn_{machine}.json"
MB = 2**20


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class Calibration:
    """State shared by the measurements of one calibration run."""

    def __init__(
        self, config: DatasetConfig, splits: SplitsConfig, data: dict[str, Any], device: str
    ) -> None:
        self.data = data
        self.device = device
        self.training = TrainingSettings.from_mapping(dict(data["entrenamiento"]))
        self.amp_dtype = mixed_precision(self.training, device)
        self.amp = self.amp_dtype is not None
        self.fixed = dict(data.get("fijos_modelo") or {})
        pool = load_pool(config, splits, data["nombre"])
        self.pool = pool
        self.fold = fold_training(pool, int(data["bloque"]), splits)
        graphs = config.rutas.graphs / DAG_DIR
        sizes = graph_sizes(graphs)
        transform = EdgeTransform(self.training.variante_aristas)
        self.parts = {}
        for name, index in (("train", self.fold.train_index), ("held", self.fold.held_out_index)):
            rows = pool.rows.iloc[index]
            dataset = PoolGraphs(
                graphs,
                list(rows["id"]),
                rows["y_algoritmo"].to_numpy(),
                rows["y_familia"].to_numpy(),
                transform,
            )
            self.parts[name] = (dataset, sizes.loc[rows["id"]].to_numpy())
        self.train_nodes = int(self.parts["train"][1].sum())
        self.held_nodes = int(self.parts["held"][1].sum())

    def batches(self, part: str, budget: int, count: int, seed: int) -> list[Any]:
        """The first ``count`` batches of ``budget`` nodes of a part (shuffled, seeded)."""
        dataset, sizes = self.parts[part]
        loader, _ = _loader(
            dataset,
            sizes,
            budget,
            shuffle=True,
            seed=seed,
            workers=0,
            device=self.device,
            max_batches=count,
        )
        return list(itertools.islice(loader, count))

    def build(self, model: dict[str, Any]) -> tuple[CircuitGNN, AccumulatingStep, dict]:
        settings = {**self.fixed, **model}
        config = GNNConfig.from_mapping(
            settings,
            node_dim=len(NODE_FEATURES),
            edge_dim=EDGE_DIM,
            num_classes=len(self.pool.class_names("algoritmo")),
            num_families=len(self.pool.class_names("familia")),
        )
        torch.manual_seed(int(self.data["semilla"]))
        network = CircuitGNN(config).to(self.device).train()
        loss = MultiTaskLoss(
            _weights(self.fold.weights.get("algoritmo"), self.pool.class_names("algoritmo")),
            _weights(self.fold.weights.get("familia"), self.pool.class_names("familia")),
            self.training.lambda_familia,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.training.tasa_aprendizaje,
            weight_decay=self.training.regularizacion_pesos,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16)
        memory = self.data["memoria"]
        step = AccumulatingStep(
            network,
            loss,
            optimizer,
            scaler,
            device=self.device,
            amp_dtype=self.amp_dtype,
            micro_nodes=self.training.presupuesto_nodos,
            min_micro_nodes=int(memory["microlote_minimo"]),
        )
        return network, step, settings


def measure_memory(cal: Calibration, model: dict[str, Any], batches: list[Any]) -> dict[str, Any]:
    """Peak MB per node of training micro-batches of one configuration."""
    network, step, settings = cal.build(model)
    largest = max(b.num_nodes for b in batches)
    step.micro_nodes = step.min_micro_nodes = largest  # measure whole micro-batches, no halving
    try:
        step(batches[0])  # warm-up: optimiser states, kernels
        _sync()
        per_node, peaks = [], []
        for batch in batches[1:]:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            step(batch)
            _sync()
            peak = torch.cuda.max_memory_allocated()
            per_node.append((peak - before) / MB / batch.num_nodes)
            peaks.append(peak / MB)
        result = {
            "modelo": settings,
            "mb_por_nodo": float(max(per_node)),
            "pico_asignado_mb": float(max(peaks)),
            "base_mb": float(before / MB),
            "nodos_microlote": int(largest),
        }
    except torch.OutOfMemoryError:
        result = {"modelo": settings, "error": "sin memoria"}
    del network, step
    torch.cuda.empty_cache()
    return result


def measure_time(
    cal: Calibration,
    name: str,
    model: dict[str, Any],
    memory: MemoryModel | None,
    train_batches: list[Any],
    held_batches: list[Any],
) -> dict[str, Any]:
    """Real seconds per epoch and hours per trial of one configuration."""
    network, step, settings = cal.build(model)
    budget = cal.training.presupuesto_nodos
    minimum = int(cal.data["memoria"]["microlote_minimo"])
    step.micro_nodes = budget if memory is None else memory.micro_nodes(settings, budget, minimum)
    gpu = cal.device == "cuda"
    chosen = step.micro_nodes
    timing = cal.data["tiempo"]
    try:
        step(train_batches[0])
        _sync()
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        nodes = 0
        for batch in train_batches[1:]:
            step(batch)
            nodes += batch.num_nodes
        _sync()
        train_rate = nodes / (time.perf_counter() - start)
        allocated = torch.cuda.max_memory_allocated() / MB if gpu else 0.0
        reserved = torch.cuda.max_memory_reserved() / MB if gpu else 0.0
        network.eval()
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=cal.amp_dtype or torch.float16, enabled=cal.amp),
        ):
            network(held_batches[0].to(cal.device))
            _sync()
            start, nodes = time.perf_counter(), 0
            for batch in held_batches[1:]:
                network(batch.to(cal.device))
                nodes += batch.num_nodes
            _sync()
        eval_rate = nodes / (time.perf_counter() - start)
    except torch.OutOfMemoryError:
        micro = step.micro_nodes
        del network, step
        torch.cuda.empty_cache()
        return {
            "nombre": name,
            "modelo": settings,
            "error": "sin memoria",
            "microlote_nodos": micro,
        }
    parameters = sum(p.numel() for p in network.parameters())
    micro = step.micro_nodes
    del network, step
    torch.cuda.empty_cache()
    epoch = cal.train_nodes / train_rate
    evaluation = cal.held_nodes / eval_rate
    every = cal.training.evaluar_cada
    hours = {
        str(e): timing["bloques"] * (e * epoch + -(-e // every) * evaluation) / 3600
        for e in timing["epocas"]
    }
    useful = float(cal.data["memoria"]["memoria_util_mb"])
    return {
        "nombre": name,
        "modelo": settings,
        "microlote_inicial": chosen,
        "microlote_nodos": micro,
        "segundos_entrenamiento_epoca": epoch,
        "segundos_evaluacion": evaluation,
        "horas_por_intento": hours,
        "pico_asignado_mb": allocated,
        "pico_reservado_mb": reserved,
        "desborda": gpu and reserved > useful,
        "dispositivo": cal.device,
        "precision_mixta": precision_name(cal.amp_dtype),
        "hilos_cpu": torch.get_num_threads() if not gpu else None,
        "parametros": parameters,
    }


def _fit_memory(
    cal: Calibration, data: dict[str, Any], machine: str, results: Path, seed: int
) -> tuple[list[dict[str, Any]], MemoryModel, Path]:
    memory_settings = data["memoria"]
    micro = cal.batches(
        "train", int(memory_settings["microlote_medida"]), int(memory_settings["lotes"]) + 1, seed
    )
    measurements = []
    for model in data["diseno"]:
        result = measure_memory(cal, model, micro)
        logger.info("memory %s: %s", model, result.get("mb_por_nodo", result.get("error")))
        measurements.append(result)
    fitted = [m for m in measurements if "mb_por_nodo" in m]
    memory = MemoryModel.fit(
        machine,
        fitted,
        useful_mb=float(memory_settings["memoria_util_mb"]),
        margin=float(memory_settings["margen"]),
    )
    for m in fitted:
        m["mb_por_nodo_predicho"] = memory.per_node_mb(m["modelo"])
    model_file = model_path(results, machine)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    memory.save(model_file)
    return measurements, memory, model_file


def run_calibration(
    config: DatasetConfig, splits: SplitsConfig, path: Path | str, *, device: str = "cuda"
) -> tuple[dict[str, Any], Path]:
    """Run the calibration (module docstring) and write the report. On ``cpu`` only the time
    part runs (float32, ``QCCLASS_NUCLEOS_LIBRES`` cores left free, ``tiempo_cpu`` counts).

    Raises:
        ConfigError: if the configuration is invalid.
        ValueError: if the device is not available.
    """
    data = load_experiment_config(path, REQUIRED)
    if device not in ("cuda", "cpu"):
        raise ValueError("the calibration runs on cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine")
    configure_cpu_threads(device)
    provenance = start_provenance(config, data["nombre"], device=device)
    try:
        cal = Calibration(config, splits, data, device)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc
    seed = int(data["semilla"])
    measurements, memory, model_file = [], None, None
    if device == "cuda":
        measurements, memory, model_file = _fit_memory(
            cal, data, provenance.machine, config.rutas.resultados, seed
        )
    timing = (
        {**data["tiempo"], **(data.get("tiempo_cpu") or {})}
        if device == "cpu"
        else (data["tiempo"])
    )
    budget = cal.training.presupuesto_nodos
    train_batches = cal.batches("train", budget, int(timing["lotes"]) + 1, seed + 1)
    held_batches = cal.batches("held", budget, int(timing["lotes_evaluacion"]) + 1, seed + 2)
    estimates = []
    for name, model in data["estimar"].items():
        result = measure_time(cal, name, model, memory, train_batches, held_batches)
        logger.info("time %s: %s", name, {k: v for k, v in result.items() if k != "modelo"})
        estimates.append(result)
    report = {
        "equipo": provenance.machine,
        "bloque": int(data["bloque"]),
        "nodos_entrenamiento": cal.train_nodes,
        "nodos_evaluacion": cal.held_nodes,
        "memoria": measurements,
        "dispositivo": device,
        "modelo_memoria": str(model_file) if model_file else None,
        "coeficientes": (
            dict(zip(FEATURES, memory.coefficients, strict=True)) if memory is not None else None
        ),
        "estimaciones": estimates,
        **{k: provenance.tags[k] for k in ("version_dataset", "git_commit")},
    }
    report_path = (
        config.rutas.resultados / "medicion" / REPORT_FILE.format(machine=provenance.machine)
    )
    write_text_atomic(json.dumps(report, indent=2, ensure_ascii=False) + "\n", report_path)
    with tracked_run(
        experiment=data["experimento_mlflow"],
        run_name=f"{data['nombre']}.{provenance.machine}",
        configuration=data,
        provenance=provenance.tags,
        machine_description=provenance.machine_description,
    ) as run:
        for e in estimates:
            if "segundos_entrenamiento_epoca" in e:
                run.log_metrics(
                    {
                        f"{e['nombre']}.segundos_epoca": e["segundos_entrenamiento_epoca"],
                        f"{e['nombre']}.horas_intento_100": e["horas_por_intento"].get(
                            "100", np.nan
                        ),
                        f"{e['nombre']}.pico_reservado_mb": e["pico_reservado_mb"],
                    }
                )
        run.log_artifact(report_path)
        if model_file is not None:
            run.log_artifact(model_file)
    return report, report_path
