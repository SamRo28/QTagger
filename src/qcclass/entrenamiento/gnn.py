"""Training of the GNN on one cross-validation block, with checkpoints and resumption.

:func:`train_block` trains :class:`~qcclass.modelos.gnn.CircuitGNN` on the training part of a
block (after its cap, with its class weights) and evaluates it on the held-out block:

* AdamW; learning rate with a linear warm-up over ``calentamiento`` of the epochs and a
  cosine decay to 0, updated at every step (fractional epochs);
* batches limited by a node budget (``presupuesto_nodos``), reshuffled every epoch with a
  seed derived from the run seed and the block; ``workers`` loading processes;
* mixed precision (float16 autocast + gradient scaler) only on CUDA;
* gradient accumulation over micro-batches when the whole batch does not fit in the GPU,
  with the exact loss of the whole batch and one optimiser step per batch; on an
  out-of-memory error the batch is retried with micro-batches of half the size
  (``entrenamiento.microlotes``);
* a checkpoint after every epoch in ``directory`` (model, optimiser, scaler, sampler epoch,
  history and every random generator). Launching again with the same ``directory`` resumes
  from the last epoch; a finished run (``resultado.json``) is returned as it is. The
  directory is chosen by :func:`run_directory` from the fingerprint of the hyperparameters
  and the block, so a trial interrupted and launched again continues where it stopped;
* metrics of every epoch (losses, learning rate, time, GPU memory and temperature, and the
  held-out macro F1 every ``evaluar_cada`` epochs and at the end) go to an optional
  :class:`~qcclass.seguimiento.registro.ResilientRun` with a non-blocking flush, and to an
  optional ``on_epoch`` callback (Optuna reports and pruning, phase 12).

MPS is never chosen automatically: in the phase M measurement it was slower than the CPU.
"""

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from qcclass.circuitos import write_text_atomic
from qcclass.entrenamiento.bloques import CrossValidationPool, FoldTraining
from qcclass.entrenamiento.microlotes import AccumulatingStep
from qcclass.entrenamiento.puntos_guardado import CheckpointManager
from qcclass.entrenamiento.reproducibilidad import set_global_seeds
from qcclass.evaluacion.metricas import confusion, scores_from_confusion
from qcclass.evaluacion.predicciones import make_table
from qcclass.formatos.comun import UNKNOWN
from qcclass.formatos.construir import DAG_DIR
from qcclass.formatos.grafo_dag import NODE_FEATURES, IndexedGraphDataset
from qcclass.medicion.monitor import ResourceMonitor
from qcclass.modelos.aristas import EDGE_DIM, EdgeTransform
from qcclass.modelos.gnn import CircuitGNN, GNNConfig, MultiTaskLoss
from qcclass.modelos.lotes import NodeBudgetBatchSampler, graph_sizes
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

RESULT_FILE = "resultado.json"
PREDICTIONS_FILE = "predicciones.parquet"
ARCHITECTURE_FILE = "arquitectura.json"
CHECKPOINT_DIR = "puntos"
DEVICES = ("auto", "cuda", "cpu", "mps")
RAM_INTERVAL = 1.0  # seconds between RAM samples during training
DEFAULT_MIN_MICRO_NODES = 20_000  # smallest micro-batch: above the largest graph (19 937)


@dataclass(frozen=True)
class TrainingSettings:
    """Training hyperparameters (``entrenamiento`` of an experiment configuration)."""

    epocas: int
    tasa_aprendizaje: float
    regularizacion_pesos: float
    lambda_familia: float = 1.0
    calentamiento: float = 0.05
    presupuesto_nodos: int = 100_000
    precision_mixta: bool = True
    variante_aristas: str = "completo"
    evaluar_cada: int = 1

    def __post_init__(self) -> None:
        if self.epocas < 1 or self.presupuesto_nodos < 1 or self.evaluar_cada < 1:
            raise ValueError("epocas, presupuesto_nodos and evaluar_cada must be positive")
        if not 0 <= self.calentamiento < 1:
            raise ValueError("calentamiento must be in [0, 1)")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingSettings":
        """Build from a configuration mapping; unknown keys raise."""
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(values) - known):
            raise ValueError(f"unknown training settings {unknown}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary."""
        return asdict(self)


@dataclass
class BlockResult:
    """Outcome of :func:`train_block`."""

    block: int
    f1_macro_algoritmo: float
    f1_macro_familia: float
    history: list[dict[str, Any]]
    seconds: float
    directory: Path
    resumed_from: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def hyperparameter_fingerprint(params: dict[str, Any]) -> str:
    """Short SHA-256 of the canonical JSON of ``params`` (keys sorted)."""
    text = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_directory(root: Path, experiment: str, params: dict[str, Any], block: int) -> Path:
    """``<root>/<experiment>/<fingerprint of params>/bloque_<block>``."""
    return Path(root) / experiment / hyperparameter_fingerprint(params) / f"bloque_{block}"


def resolve_training_device(requested: str) -> str:
    """``auto`` -> ``cuda`` if available, else ``cpu`` (never MPS; see the docstring).

    Raises:
        ValueError: if the device is unknown or not available.
    """
    if requested not in DEVICES:
        raise ValueError(f"unknown device {requested!r} (valid: {DEVICES})")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine")
    return requested


def learning_rate_factor(progress: float, epochs: int, warmup_fraction: float) -> float:
    """Factor of the base learning rate after ``progress`` epochs (warm-up, then cosine)."""
    warmup = warmup_fraction * epochs
    if warmup > 0 and progress < warmup:
        return (progress + 1e-8) / warmup
    span = max(epochs - warmup, 1e-8)
    return 0.5 * (1 + math.cos(math.pi * min((progress - warmup) / span, 1.0)))


class PoolGraphs(IndexedGraphDataset):
    """F4 graphs of some pool rows with the targets of the pool (not those stored)."""

    def __init__(
        self,
        directory: Path,
        ids: list[str],
        y_algorithm: np.ndarray,
        y_family: np.ndarray,
        transform: EdgeTransform,
    ) -> None:
        super().__init__(directory, ids, transform=None)
        self.edges = transform
        self.y_algorithm = torch.tensor(np.array(y_algorithm), dtype=torch.long)
        self.y_family = torch.tensor(np.array(y_family), dtype=torch.long)

    def get(self, idx: int):  # noqa: ANN201 - torch_geometric Data
        data = self.edges(super().get(idx))
        data.y_algoritmo = self.y_algorithm[idx : idx + 1]
        data.y_familia = self.y_family[idx : idx + 1]
        return data


def _weights(table: dict[str, float] | None, names: tuple[str, ...]) -> torch.Tensor:
    if not table:
        return torch.ones(len(names))
    return torch.tensor([float(table.get(name, 1.0)) for name in names])


def _loader(
    dataset: PoolGraphs,
    sizes: np.ndarray,
    budget: int,
    *,
    shuffle: bool,
    seed: int,
    workers: int,
    device: str,
    max_batches: int | None,
) -> tuple[DataLoader, NodeBudgetBatchSampler]:
    sampler = NodeBudgetBatchSampler(
        sizes, budget, shuffle=shuffle, seed=seed, max_batches=max_batches
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=device == "cuda",
        persistent_workers=workers > 0,
    )
    return loader, sampler


def _resource_readings(device: str, monitor: ResourceMonitor) -> dict[str, float]:
    """Peaks of the epoch: RAM (process, and with the loading workers), GPU memory allocated
    and reserved by PyTorch, maximum GPU temperature."""
    summary = monitor.reset().summary()
    readings = {
        "ram_max_mb": summary["ram_max_mb"],
        "ram_max_con_workers_mb": summary["ram_max_con_workers_mb"],
    }
    if summary["temperatura_gpu_max"] is not None:
        readings["temperatura_gpu"] = summary["temperatura_gpu_max"]
    if device == "cuda":
        readings["memoria_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
        readings["memoria_gpu_reservada_mb"] = torch.cuda.max_memory_reserved() / 2**20
        torch.cuda.reset_peak_memory_stats()
    return readings


@torch.no_grad()
def predict(
    model: CircuitGNN, loader: DataLoader, device: str, amp: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities of both heads for every graph of ``loader`` (in its order)."""
    model.eval()
    algorithm, family = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            output = model(batch)
        algorithm.append(output.algoritmo.float().softmax(dim=1).cpu())
        family.append(output.familia.float().softmax(dim=1).cpu())
    model.train()
    return torch.cat(algorithm).numpy(), torch.cat(family).numpy()


def _macro_f1(y: np.ndarray, probabilities: np.ndarray) -> float:
    keep = y >= 0
    if not keep.any():
        return float("nan")
    matrix = confusion(y[keep], probabilities[keep].argmax(axis=1), probabilities.shape[1])
    return float(scores_from_confusion(matrix)["f1_macro"])


def train_block(
    *,
    pool: CrossValidationPool,
    fold: FoldTraining,
    graphs_dir: Path,
    model_settings: dict[str, Any],
    training: TrainingSettings,
    seed: int,
    device: str,
    directory: Path,
    workers: int = 0,
    run: Any | None = None,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    max_batches: int | None = None,
    metric_prefix: str = "",
    stop_after: int | None = None,
    micro_nodes: int | None = None,
    min_micro_nodes: int = DEFAULT_MIN_MICRO_NODES,
) -> BlockResult:
    """Train on ``fold`` and evaluate on its held-out block (see the module docstring).

    ``stop_after`` ends the call after that epoch (as an interruption would; tests).
    ``micro_nodes`` is the initial micro-batch size (default: the whole batch); it halves on
    an out-of-memory error down to ``min_micro_nodes`` (see ``entrenamiento.microlotes``).

    Raises:
        ValueError: if a training target is unknown (-2) or a checkpoint belongs to other
            hyperparameters.
    """
    directory = Path(directory)
    result_path = directory / RESULT_FILE
    if result_path.is_file():
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        logger.info("block %d already trained (%s)", fold.block, directory)
        return BlockResult(
            block=fold.block,
            f1_macro_algoritmo=stored["f1_macro_algoritmo"],
            f1_macro_familia=stored["f1_macro_familia"],
            history=stored["historial"],
            seconds=stored["segundos"],
            directory=directory,
            extra=stored,
        )
    amp = training.precision_mixta and device == "cuda"
    rows = pool.rows
    train_rows = rows.iloc[fold.train_index]
    held_rows = rows.iloc[fold.held_out_index]
    for name, part in (("training", train_rows), ("held-out", held_rows)):
        if (part[["y_algoritmo", "y_familia"]].to_numpy() == UNKNOWN).any():
            raise ValueError(f"an unknown (-2) circuit is in the {name} rows of the pool")
    graphs = Path(graphs_dir) / DAG_DIR
    sizes = graph_sizes(graphs)
    transform = EdgeTransform(training.variante_aristas)

    def dataset(part: pd.DataFrame) -> PoolGraphs:
        return PoolGraphs(
            graphs,
            list(part["id"]),
            part["y_algoritmo"].to_numpy(),
            part["y_familia"].to_numpy(),
            transform,
        )

    train_loader, sampler = _loader(
        dataset(train_rows),
        sizes.loc[train_rows["id"]].to_numpy(),
        training.presupuesto_nodos,
        shuffle=True,
        seed=derive_seed(seed, "lotes", fold.block),
        workers=workers,
        device=device,
        max_batches=max_batches,
    )
    eval_loader, _ = _loader(
        dataset(held_rows),
        sizes.loc[held_rows["id"]].to_numpy(),
        training.presupuesto_nodos,
        shuffle=False,
        seed=0,
        workers=workers,
        device=device,
        max_batches=None,
    )
    set_global_seeds(derive_seed(seed, "modelo", fold.block))
    config = GNNConfig.from_mapping(
        model_settings,
        node_dim=len(NODE_FEATURES),
        edge_dim=EDGE_DIM,
        num_classes=len(pool.class_names("algoritmo")),
        num_families=len(pool.class_names("familia")),
    )
    model = CircuitGNN(config).to(device)
    loss_fn = MultiTaskLoss(
        _weights(fold.weights.get("algoritmo"), pool.class_names("algoritmo")),
        _weights(fold.weights.get("familia"), pool.class_names("familia")),
        training.lambda_familia,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.tasa_aprendizaje, weight_decay=training.regularizacion_pesos
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoints = CheckpointManager(directory / CHECKPOINT_DIR)
    identity = {
        "modelo": config.as_dict(),
        "entrenamiento": training.as_dict(),
        "semilla": seed,
        "bloque": fold.block,
    }
    history: list[dict[str, Any]] = []
    start, resumed = 0, None
    micro = (
        training.presupuesto_nodos
        if micro_nodes is None
        else min(micro_nodes, training.presupuesto_nodos)
    )
    loaded = checkpoints.load_latest(map_location=device)
    if loaded is not None:
        epoch, states = loaded
        if states["identidad"] != json.loads(json.dumps(identity)):
            raise ValueError(f"the checkpoints in {directory} belong to other hyperparameters")
        model.load_state_dict(states["modelo"])
        optimizer.load_state_dict(states["optimizador"])
        scaler.load_state_dict(states["escalador"])
        history = list(states["historial"])
        start = resumed = epoch
        micro = int(states.get("microlote", micro))
    sampler.epoch = start
    stepper = AccumulatingStep(
        model,
        loss_fn,
        optimizer,
        scaler,
        device=device,
        amp=amp,
        micro_nodes=max(micro, min_micro_nodes),
        min_micro_nodes=min_micro_nodes,
    )
    y_alg = held_rows["y_algoritmo"].to_numpy()
    y_fam = held_rows["y_familia"].to_numpy()
    started = time.perf_counter() - sum(h["segundos_epoca"] for h in history)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(
        gpu_temperature=device == "cuda", ram_interval=RAM_INTERVAL, temperature_interval=10.0
    )
    monitor.__enter__()
    try:
        for epoch in range(start, training.epocas):
            epoch_start = time.perf_counter()
            totals = np.zeros(3)
            graphs_seen = 0
            batches = len(sampler)
            micro_batches = 0
            for step, batch in enumerate(train_loader):
                factor = learning_rate_factor(
                    epoch + step / max(batches, 1), training.epocas, training.calentamiento
                )
                for group in optimizer.param_groups:
                    group["lr"] = training.tasa_aprendizaje * factor
                result = stepper(batch)
                micro_batches += result.microlotes
                totals += batch.num_graphs * np.array(
                    [result.total, result.algoritmo, result.familia]
                )
                graphs_seen += batch.num_graphs
            metrics: dict[str, float] = {
                "perdida": totals[0] / max(graphs_seen, 1),
                "perdida_algoritmo": totals[1] / max(graphs_seen, 1),
                "perdida_familia": totals[2] / max(graphs_seen, 1),
                "tasa_aprendizaje": optimizer.param_groups[0]["lr"],
                "segundos_entrenamiento": time.perf_counter() - epoch_start,
                "microlote_nodos": stepper.micro_nodes,
                "microlotes_por_lote": micro_batches / max(batches, 1),
            }
            last = epoch + 1 == training.epocas
            if last or (epoch + 1) % training.evaluar_cada == 0:
                evaluation_start = time.perf_counter()
                p_alg, p_fam = predict(model, eval_loader, device, amp)
                metrics["f1_macro_algoritmo"] = _macro_f1(y_alg, p_alg)
                metrics["f1_macro_familia"] = _macro_f1(y_fam, p_fam)
                metrics["segundos_evaluacion"] = time.perf_counter() - evaluation_start
            metrics.update(_resource_readings(device, monitor))
            metrics["segundos_epoca"] = time.perf_counter() - epoch_start
            metrics = {k: float(v) for k, v in metrics.items()}  # plain floats: weights_only load
            history.append({"epoca": epoch + 1, **metrics})
            checkpoints.save(
                epoch + 1,
                {
                    "identidad": identity,
                    "modelo": model.state_dict(),
                    "optimizador": optimizer.state_dict(),
                    "escalador": scaler.state_dict(),
                    "historial": history,
                    "microlote": stepper.micro_nodes,
                },
            )
            logger.info(
                "block %d epoch %d/%d: loss %.4f%s (%.1f s)",
                fold.block,
                epoch + 1,
                training.epocas,
                metrics["perdida"],
                f", F1 {metrics['f1_macro_algoritmo']:.4f}"
                if "f1_macro_algoritmo" in metrics
                else "",
                metrics["segundos_epoca"],
            )
            if run is not None:
                run.log_metrics(
                    {f"{metric_prefix}{k}": v for k, v in metrics.items()}, step=epoch + 1
                )
                run.flush(block=False)
            if on_epoch is not None:
                on_epoch(epoch + 1, metrics)
            if stop_after is not None and epoch + 1 >= stop_after and not last:
                return BlockResult(
                    fold.block,
                    float("nan"),
                    float("nan"),
                    history,
                    time.perf_counter() - started,
                    directory,
                    resumed,
                )
    finally:
        monitor.__exit__(None, None, None)

    if start >= training.epocas:  # every epoch was saved, the result was not
        p_alg, p_fam = predict(model, eval_loader, device, amp)
    final = history[-1]
    table = make_table(
        held_rows["id"],
        held_rows["group_id"],
        {"algoritmo": y_alg, "familia": y_fam},
        {"algoritmo": p_alg, "familia": p_fam},
        evaluation_only=held_rows["solo_evaluacion"],
        block=fold.block,
    )
    table.to_parquet(directory / PREDICTIONS_FILE, index=False)
    seconds = time.perf_counter() - started
    stored = {
        "bloque": fold.block,
        "f1_macro_algoritmo": final["f1_macro_algoritmo"],
        "f1_macro_familia": final["f1_macro_familia"],
        "historial": history,
        "segundos": seconds,
        "dispositivo": device,
        "precision_mixta": amp,
        "identidad": identity,
        "huella": hyperparameter_fingerprint(identity),
        "reanudado_desde": resumed,
    }
    write_text_atomic(
        json.dumps(model.describe(), indent=2, ensure_ascii=False), directory / ARCHITECTURE_FILE
    )
    write_text_atomic(json.dumps(stored, indent=2, ensure_ascii=False) + "\n", result_path)
    return BlockResult(
        fold.block,
        final["f1_macro_algoritmo"],
        final["f1_macro_familia"],
        history,
        seconds,
        directory,
        resumed,
        stored,
    )
