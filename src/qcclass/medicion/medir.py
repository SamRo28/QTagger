"""Training-cost measurement of the reference GNN on one machine (``qcclass medir``).

Only the ``train`` set of the ``aleatorio`` protocol is read (without ``solo_evaluacion``
variants); no validation or test data is touched. Steps:

1. Fingerprints of the dataset checked against ``VERSION.json`` (abort if they differ).
2. Worst case: one forward and backward pass with the largest training graph.
3. ``epocas`` epochs with batches limited by a node budget, timing separately the wait for
   data and the computation, plus throughput, loss, GPU memory, RAM and GPU temperature.
4. Optional sweep (``--barrido``) of several node budgets with a fixed number of batches;
   out-of-memory errors are recorded, not raised.
5. Projection of the hours of full trainings and of an Optuna search from the nodes per
   second of the last epoch (epoch 2 by default, with a warm disk cache).

The result is written to ``<resultados>/medicion/<machine>_<device>[_amp].json``.
"""

import gc
import itertools
import json
import logging
import re
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from qcclass.config import DatasetConfig
from qcclass.data.particiones import balance_path
from qcclass.formatos.construir import DAG_DIR
from qcclass.formatos.grafo_dag import EDGE_FEATURES, NODE_FEATURES, IndexedGraphDataset
from qcclass.medicion.equipo import machine_info, machine_name
from qcclass.medicion.monitor import ResourceMonitor
from qcclass.metadatos import METADATA_COLUMNS, read_table
from qcclass.modelos.lotes import NodeBudgetBatchSampler, graph_sizes
from qcclass.modelos.referencia import (
    Labels,
    MultiTaskLoss,
    ReferenceGNN,
    class_weight_tensors,
    count_parameters,
)
from qcclass.semillas import derive_seed
from qcclass.versionado import LABEL_VOCABULARY_FILE, begin_experiment, split_ids

logger = logging.getLogger(__name__)

PROTOCOL = "aleatorio"
MEASUREMENT_DIR = "medicion"
SUMMARY_FILE = "RESUMEN.md"
SWEEP_BUDGETS = (25_000, 50_000, 100_000, 200_000, 400_000)  # 25k: 4 GB laptop GPUs
SWEEP_BATCHES = 50
PROJECTION_EPOCHS = (50, 100, 200)
OPTUNA_TRIALS = (20, 50)
OPTUNA_FOLDS = 5
OPTUNA_EPOCHS = 100
# each block of a 5-fold search trains on 4/5 of the training set
OPTUNA_FOLD_TRAIN_FRACTION = (OPTUNA_FOLDS - 1) / OPTUNA_FOLDS
LEARNING_RATE = 1e-3
MPS_FALLBACK_PATTERN = re.compile(r"operator '([^']+)' is not currently supported on the MPS")


@dataclass(frozen=True)
class MeasurementOptions:
    """Options of ``qcclass medir``."""

    dispositivo: str = "auto"
    presupuesto_nodos: int = 100_000
    epocas: int = 2
    max_lotes: int | None = None
    num_workers: int = 0
    amp: bool = False
    dim: int = 64
    capas: int = 3
    barrido: bool = False
    equipo: str | None = None


@dataclass(frozen=True)
class TrainingSet:
    """Training graphs of the measured protocol, their sizes and the loss weights."""

    dataset: IndexedGraphDataset
    sizes: np.ndarray
    labels: Labels
    algorithm_weights: torch.Tensor
    family_weights: torch.Tensor
    excluded_unknown: int


# --------------------------------------------------------------------------------------------
# Device helpers
# --------------------------------------------------------------------------------------------


def synchronize(device: str) -> None:
    """Wait for the queued kernels of ``device`` (accurate timings)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def is_out_of_memory(error: BaseException) -> bool:
    """True for CUDA and MPS out-of-memory errors."""
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def release_memory(device: str) -> None:
    """Free cached device memory after an error or between sweep budgets."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class GpuMemory:
    """Peak device memory: CUDA allocator statistics, MPS sampled after every step."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._mps_peak = 0

    def reset(self) -> None:
        """Start a new peak."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._mps_peak = 0
        self.sample()

    def sample(self) -> None:
        """Record the current MPS driver memory (no-op elsewhere)."""
        if self.device == "mps":
            self._mps_peak = max(self._mps_peak, torch.mps.driver_allocated_memory())

    def peak(self) -> dict:
        """Peak memory in MiB (``None`` on CPU)."""
        if self.device == "cuda":
            return {
                "memoria_gpu_max_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "memoria_gpu_reservada_max_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            }
        if self.device == "mps":
            return {"memoria_gpu_max_mb": round(self._mps_peak / 2**20, 1)}
        return {"memoria_gpu_max_mb": None}


def startup_memory(device: str) -> dict | None:
    """Device memory once the model and optimiser are on it, before the first batch.

    On CUDA: what the caching allocator has reserved (``memory_reserved``) and allocated,
    and the free and total memory seen by the driver (``mem_get_info``), whose difference
    also includes the CUDA context. ``util_mb`` is what is left for batches.
    """
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        return {
            "memoria_reservada_mb": round(reserved / 2**20, 1),
            "memoria_asignada_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
            "memoria_total_mb": round(total / 2**20, 1),
            "memoria_libre_driver_mb": round(free / 2**20, 1),
            "contexto_y_otros_mb": round((total - free - reserved) / 2**20, 1),
            "util_mb": round((free + reserved - torch.cuda.memory_allocated()) / 2**20, 1),
        }
    if device == "mps":
        return {
            "memoria_asignada_mb": round(torch.mps.current_allocated_memory() / 2**20, 1),
            "memoria_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20, 1),
            "memoria_recomendada_mb": round(torch.mps.recommended_max_memory() / 2**20, 1),
        }
    return None


# --------------------------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------------------------


def load_training_set(config: DatasetConfig, experiment: str) -> TrainingSet:
    """Train graphs of :data:`PROTOCOL` without evaluation-only variants or unknown labels."""
    rows = split_ids(config, PROTOCOL, ["train"], experiment=experiment)
    rows = rows[~rows["solo_evaluacion"]]
    labels = Labels.read(config.rutas.graphs / LABEL_VOCABULARY_FILE)
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS).set_index("id")
    unknown = metadata.loc[rows["id"], "algoritmo"].isin(labels.unknown).to_numpy()
    if unknown.any():  # the split never puts them in train; guard anyway
        logger.warning("%d unknown (-2) graphs removed from train", int(unknown.sum()))
    ids = list(rows.loc[~unknown, "id"])
    all_sizes = graph_sizes(config.rutas.graphs / DAG_DIR)
    if missing := sorted(set(ids) - set(all_sizes.index)):
        raise FileNotFoundError(f"{len(missing)} train graphs are not in the index: {missing[:5]}")
    algorithm_weights, family_weights = class_weight_tensors(balance_path(config, PROTOCOL), labels)
    return TrainingSet(
        dataset=IndexedGraphDataset(config.rutas.graphs / DAG_DIR, ids),
        sizes=all_sizes.loc[ids].to_numpy(),
        labels=labels,
        algorithm_weights=algorithm_weights,
        family_weights=family_weights,
        excluded_unknown=int(unknown.sum()),
    )


def build_model(training: TrainingSet, options: MeasurementOptions, seed: int) -> ReferenceGNN:
    """Reference model with a reproducible initialisation."""
    torch.manual_seed(seed)
    return ReferenceGNN(
        len(NODE_FEATURES),
        len(EDGE_FEATURES),
        len(training.labels.classes),
        len(training.labels.families),
        dim=options.dim,
        layers=options.capas,
    )


def make_loader(
    training: TrainingSet,
    budget: int,
    options: MeasurementOptions,
    device: str,
    seed: int,
    max_batches: int | None,
) -> DataLoader:
    """DataLoader with node-budget batches, reshuffled every epoch."""
    sampler = NodeBudgetBatchSampler(
        training.sizes, budget, shuffle=True, seed=seed, max_batches=max_batches
    )
    return DataLoader(
        training.dataset,
        batch_sampler=sampler,
        num_workers=options.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=options.num_workers > 0,
    )


class Trainer:
    """Model, loss, optimiser and (on CUDA with ``--amp``) mixed precision."""

    def __init__(self, model: ReferenceGNN, training: TrainingSet, device: str, amp: bool) -> None:
        self.device = device
        self.amp = amp
        self.model = model.to(device)
        self.model.train()
        self.loss = MultiTaskLoss(training.algorithm_weights, training.family_weights).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def step(self, batch: Batch, *, update: bool = True) -> tuple[float, float, float]:
        """One training step; returns (total, algorithm, family) loss. ``update=False`` only
        computes the gradients (worst-case pass)."""
        batch = batch.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.model(batch)
        parts = self.loss(logits, batch.y_algoritmo, batch.y_familia)
        self.scaler.scale(parts.total).backward()
        if update:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        return float(parts.total.detach()), parts.algorithm, parts.family


# --------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------


def worst_case(trainer: Trainer, training: TrainingSet, memory: GpuMemory) -> dict:
    """Forward and backward pass with the largest training graph (memory worst case)."""
    position = int(np.argmax(training.sizes))
    graph = training.dataset[position]
    result = {
        "id": training.dataset.ids[position],
        "n_nodos": int(graph.num_nodes),
        "n_aristas": graph.num_edges,
    }
    memory.reset()
    start = time.perf_counter()
    try:
        trainer.step(Batch.from_data_list([graph]), update=False)
        synchronize(trainer.device)
    except (RuntimeError, torch.OutOfMemoryError) as error:
        if not is_out_of_memory(error):
            raise
        release_memory(trainer.device)
        return {**result, "ok": False, "error": str(error).splitlines()[0]}
    return {**result, "ok": True, "tiempo_s": round(time.perf_counter() - start, 4)}


def timed_loop(
    trainer: Trainer,
    batches: Iterator[Batch],
    memory: GpuMemory,
    monitor: ResourceMonitor,
    *,
    limit: int | None,
    description: str,
) -> dict:
    """Train on ``batches`` (at most ``limit``) timing data wait and computation apart."""
    monitor.reset()
    memory.reset()
    data_time = compute_time = 0.0
    losses: list[tuple[float, float, float]] = []
    nodes = graphs = 0
    start = time.perf_counter()
    progress = tqdm(total=limit, desc=description, leave=False)
    while limit is None or len(losses) < limit:
        before = time.perf_counter()
        batch = next(batches, None)
        loaded = time.perf_counter()
        data_time += loaded - before
        if batch is None:
            break
        losses.append(trainer.step(batch))
        synchronize(trainer.device)
        memory.sample()
        compute_time += time.perf_counter() - loaded
        nodes += batch.num_nodes
        graphs += batch.num_graphs
        progress.update()
    progress.close()
    total = time.perf_counter() - start
    mean = np.mean(losses, axis=0) if losses else [float("nan")] * 3
    return {
        "tiempo_total_s": round(total, 3),
        "tiempo_datos_s": round(data_time, 3),
        "tiempo_computo_s": round(compute_time, 3),
        "fraccion_espera_datos": round(data_time / total, 4) if total else None,
        "lotes": len(losses),
        "grafos": graphs,
        "nodos": nodes,
        "nodos_s": round(nodes / total, 1) if total else None,
        "grafos_s": round(graphs / total, 2) if total else None,
        "perdida_media": round(float(mean[0]), 5),
        "perdida_algoritmo": round(float(mean[1]), 5),
        "perdida_familia": round(float(mean[2]), 5),
        **memory.peak(),
        **monitor.reset().summary(),
    }


def run_epochs(
    trainer: Trainer,
    loader: DataLoader,
    memory: GpuMemory,
    monitor: ResourceMonitor,
    epochs: int,
) -> list[dict]:
    """Measure ``epochs`` epochs (the loader reshuffles at every epoch)."""
    results = []
    for epoch in range(1, epochs + 1):
        # creating the iterator starts the workers: it counts as waiting for data
        started = time.perf_counter()
        batches = iter(loader)
        startup = time.perf_counter() - started
        result = timed_loop(
            trainer, batches, memory, monitor, limit=None, description=f"epoch {epoch}"
        )
        result["tiempo_total_s"] = round(result["tiempo_total_s"] + startup, 3)
        result["tiempo_datos_s"] = round(result["tiempo_datos_s"] + startup, 3)
        logger.info(
            "epoch %d: %.1f s, %.0f nodes/s, loss %.4f",
            epoch,
            result["tiempo_total_s"],
            result["nodos_s"] or 0,
            result["perdida_media"],
        )
        results.append({"epoca": epoch, **result})
    return results


def budget_sweep(
    training: TrainingSet,
    options: MeasurementOptions,
    device: str,
    seed: int,
    monitor: ResourceMonitor,
) -> dict:
    """:data:`SWEEP_BATCHES` batches with every budget of :data:`SWEEP_BUDGETS`."""
    results = []
    for budget in SWEEP_BUDGETS:
        release_memory(device)
        trainer = Trainer(build_model(training, options, seed), training, device, options.amp)
        loader = make_loader(training, budget, options, device, seed, None)
        # small datasets give fewer batches than needed: keep cycling through epochs
        batches = (batch for _ in itertools.count() for batch in loader)
        memory = GpuMemory(device)
        try:
            result = timed_loop(
                trainer,
                batches,
                memory,
                monitor,
                limit=SWEEP_BATCHES,
                description=f"budget {budget}",
            )
            results.append({"presupuesto_nodos": budget, "ok": True, **result})
        except (RuntimeError, torch.OutOfMemoryError) as error:
            if not is_out_of_memory(error):
                raise
            logger.warning("budget %d: out of memory", budget)
            results.append(
                {"presupuesto_nodos": budget, "ok": False, "error": str(error).splitlines()[0]}
            )
        finally:
            del trainer, loader, batches
            release_memory(device)
    working = [r["presupuesto_nodos"] for r in results if r["ok"]]
    return {
        "lotes_por_presupuesto": SWEEP_BATCHES,
        "mayor_presupuesto_ok": max(working) if working else None,
        "presupuestos": results,
    }


def projection(epochs: list[dict], epoch_nodes: int) -> dict | None:
    """Hours of full trainings and Optuna searches from the last epoch's nodes per second."""
    if not epochs or not epochs[-1]["nodos_s"]:
        return None
    last = epochs[-1]
    hours_per_epoch = epoch_nodes / last["nodos_s"] / 3600
    fold_epochs = OPTUNA_FOLDS * OPTUNA_EPOCHS * OPTUNA_FOLD_TRAIN_FRACTION
    return {
        "fuente": f"epoca_{last['epoca']}",
        "cache_caliente": last["epoca"] >= 2,
        "nodos_s": last["nodos_s"],
        "nodos_por_epoca": epoch_nodes,
        "horas_por_epoca": round(hours_per_epoch, 4),
        "entrenamiento_completo_h": {
            str(n): round(n * hours_per_epoch, 2) for n in PROJECTION_EPOCHS
        },
        "optuna_h": {
            str(trials): round(trials * fold_epochs * hours_per_epoch, 1)
            for trials in OPTUNA_TRIALS
        },
        "supuestos": (
            f"solo entrenamiento (sin validación); Optuna = intentos x {OPTUNA_FOLDS} bloques x "
            f"{OPTUNA_EPOCHS} épocas, cada bloque entrena con "
            f"{OPTUNA_FOLD_TRAIN_FRACTION:.0%} de train"
        ),
    }


# --------------------------------------------------------------------------------------------
# Whole command
# --------------------------------------------------------------------------------------------


def output_path(config: DatasetConfig, machine: str, device: str, amp: bool) -> Path:
    """``<resultados>/medicion/<machine>_<device>[_amp].json``."""
    suffix = "_amp" if amp else ""
    return config.rutas.resultados / MEASUREMENT_DIR / f"{machine}_{device}{suffix}.json"


def _mps_fallback_ops(caught: list[warnings.WarningMessage]) -> list[str]:
    """Operators reported as falling back from MPS to the CPU."""
    found = {
        match.group(1) for w in caught if (match := MPS_FALLBACK_PATTERN.search(str(w.message)))
    }
    return sorted(found)


def run_measurement(
    config: DatasetConfig,
    options: MeasurementOptions,
    *,
    device: str,
    mps_fallback: dict | None = None,
    on_model: Callable[[int], None] | None = None,
) -> tuple[Path, dict]:
    """Run the whole measurement and write its JSON; returns (path, result).

    ``device`` must already be resolved (:func:`qcclass.medicion.equipo.resolve_device`).
    ``on_model`` is called with the number of parameters (the CLI prints it).
    """
    if options.amp and device != "cuda":
        raise ValueError("--amp is only available on CUDA")
    machine = options.equipo or machine_name()
    experiment = f"medicion_{machine}_{device}{'_amp' if options.amp else ''}"
    stamp = begin_experiment(config, experiment)  # aborts if the fingerprints differ
    started = time.perf_counter()
    seed = derive_seed(config.semilla_global, "medicion")
    training = load_training_set(config, experiment)
    trainer = Trainer(build_model(training, options, seed), training, device, options.amp)
    parameters = count_parameters(trainer.model)
    if on_model is not None:
        on_model(parameters)
    with (
        warnings.catch_warnings(record=True) as caught,
        ResourceMonitor(gpu_temperature=device == "cuda") as monitor,
    ):
        warnings.simplefilter("always")
        synchronize(device)
        initial_memory = startup_memory(device)  # before the worst case and the first batch
        memory = GpuMemory(device)
        worst = worst_case(trainer, training, memory)
        loader = make_loader(
            training, options.presupuesto_nodos, options, device, seed, options.max_lotes
        )
        epochs = run_epochs(trainer, loader, memory, monitor, options.epocas)
        del loader
        sweep = budget_sweep(training, options, device, seed, monitor) if options.barrido else None
    for warning in caught:
        if not MPS_FALLBACK_PATTERN.search(str(warning.message)):
            logger.debug("warning during the measurement: %s", warning.message)
    result = {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "experimento": experiment,
        **stamp,
        "equipo": machine_info(device, machine),
        "dispositivo": device,
        "opciones": asdict(options),
        "modelo": {"nombre": "referencia_gine", "parametros": parameters},
        "datos": {
            "protocolo": PROTOCOL,
            "conjunto": "train",
            "grafos": len(training.sizes),
            "nodos_por_epoca": int(training.sizes.sum()),
            "max_nodos_grafo": int(training.sizes.max()),
            "grafos_mayores_que_presupuesto": int(
                (training.sizes > options.presupuesto_nodos).sum()
            ),
            "desconocidos_excluidos": training.excluded_unknown,
        },
        "mps_fallback": (
            {**(mps_fallback or {}), "operaciones_cpu": _mps_fallback_ops(caught)}
            if device == "mps"
            else None
        ),
        "memoria_inicial": initial_memory,
        "peor_caso": worst,
        "epocas": epochs,
        "perdida_baja": (
            epochs[-1]["perdida_media"] < epochs[0]["perdida_media"] if len(epochs) > 1 else None
        ),
        "proyeccion": projection(epochs, int(training.sizes.sum())),
        "barrido": sweep,
        "duracion_total_s": round(time.perf_counter() - started, 1),
    }
    path = output_path(config, machine, device, options.amp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("measurement written to %s", path)
    return path, result
