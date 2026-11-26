"""The trained GNN seen from the export: load it, run it with traces, explain and embed.

* :func:`load_trained` loads a finished run directory (``resultado.json``, ``arquitectura.json``
  and the last checkpoint) of :func:`qcclass.entrenamiento.gnn.train_block`, read-only.
* :func:`graph_input` builds the model input of a circuit from its gates exactly as the
  dataset does (F4 of :mod:`qcclass.formatos.grafo_dag` and the edge transform of the run).
* :func:`run_traced` gives the traces of one graph as NumPy arrays.
* :func:`integrated_gradients`: attribution of the logit of a class to every node, with the
  input attributes as the variable, a zero baseline and a midpoint Riemann sum of ``steps``
  points; the importance of a gate is the sum of the attributions of its attributes.
  Edge attributes are kept fixed.
* :func:`summaries`: graph summary vectors (input of the heads) and probabilities of many
  graphs, for the unknown detector and the reference map.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from qcclass.entrenamiento.gnn import ARCHITECTURE_FILE, CHECKPOINT_DIR, RESULT_FILE
from qcclass.entrenamiento.puntos_guardado import CheckpointManager
from qcclass.formatos.comun import GateRecord
from qcclass.formatos.grafo_dag import dag_graph
from qcclass.modelos.aristas import EdgeTransform
from qcclass.modelos.gnn import CircuitGNN, GNNConfig


@dataclass
class TrainedModel:
    """A trained model with what describes its run."""

    model: CircuitGNN
    result: dict[str, Any]
    architecture: dict[str, Any]
    directory: Path
    transform: EdgeTransform

    @property
    def config(self) -> GNNConfig:
        """Model configuration."""
        return self.model.config


def load_trained(directory: Path) -> TrainedModel:
    """Load a finished training run (CPU, evaluation mode).

    Raises:
        FileNotFoundError: if the run did not finish (no ``resultado.json``) or has no
            checkpoint.
    """
    directory = Path(directory)
    result_path = directory / RESULT_FILE
    if not result_path.is_file():
        raise FileNotFoundError(f"{result_path} does not exist: the run did not finish")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    architecture = json.loads((directory / ARCHITECTURE_FILE).read_text(encoding="utf-8"))
    loaded = CheckpointManager(directory / CHECKPOINT_DIR).load_latest(map_location="cpu")
    if loaded is None:
        raise FileNotFoundError(f"no checkpoint in {directory / CHECKPOINT_DIR}")
    epoch, states = loaded
    if epoch != result["identidad"]["entrenamiento"]["epocas"]:
        raise ValueError(f"last checkpoint is epoch {epoch}, not the final one")
    model = CircuitGNN(GNNConfig(**architecture["configuracion"]))
    model.load_state_dict(states["modelo"])
    model.eval()
    transform = EdgeTransform(result["identidad"]["entrenamiento"]["variante_aristas"])
    return TrainedModel(model, result, architecture, directory, transform)


def graph_input(gates: list[GateRecord], num_qubits: int, transform: EdgeTransform) -> Data:
    """Model input of a circuit (F4 plus the edge transform)."""
    return transform(dag_graph(gates, num_qubits))


def _numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, list):
        return [_numpy(v) for v in value]
    if isinstance(value, dict):
        return {k: _numpy(v) for k, v in value.items()}
    return value


@torch.no_grad()
def run_traced(trained: TrainedModel, graph: Data) -> dict[str, Any]:
    """Traces of one graph (``CircuitGNN.forward(trazas=True)``) as NumPy arrays."""
    output = trained.model(Batch.from_data_list([graph]), trazas=True)
    return _numpy(output.trazas)


def integrated_gradients(
    trained: TrainedModel, graph: Data, target: int, *, steps: int, chunk: int = 16
) -> tuple[np.ndarray, float]:
    """Per-node importance for class ``target`` and the relative completeness error."""
    model = trained.model
    x = graph.x.float()
    baseline = torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    gradient_sum = torch.zeros_like(x)
    for start in range(0, steps, chunk):
        part = alphas[start : start + chunk]
        copies = []
        for alpha in part:
            copy = graph.clone()
            copy.x = baseline + alpha * (x - baseline)
            copies.append(copy)
        batch = Batch.from_data_list(copies)
        batch.x.requires_grad_(True)
        logits = model(batch).algoritmo[:, target].sum()
        (gradient,) = torch.autograd.grad(logits, batch.x)
        gradient_sum += gradient.view(len(part), *x.shape).sum(dim=0)
    attributions = (x - baseline) * gradient_sum / steps
    with torch.no_grad():
        ends = Batch.from_data_list([graph, Data(**{**graph.to_dict(), "x": baseline})])
        values = model(ends).algoritmo[:, target]
    difference = float(values[0] - values[1])
    total = float(attributions.sum())
    error = abs(total - difference) / max(abs(difference), 1e-8)
    return attributions.sum(dim=1).numpy(), error


@torch.no_grad()
def summaries(
    trained: TrainedModel, graphs: list[Data], *, batch_nodes: int = 100_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summary vectors and probabilities (algorithm, family) of ``graphs``, in order."""
    model = trained.model
    vectors, algorithm, family = [], [], []
    batch: list[Data] = []
    nodes = 0

    def flush() -> None:
        output = model(Batch.from_data_list(batch), trazas=True)
        vectors.append(output.trazas["resumen"].float())
        algorithm.append(output.trazas["prob_algoritmo"])
        family.append(output.trazas["prob_familia"])

    for graph in graphs:
        if batch and nodes + graph.num_nodes > batch_nodes:
            flush()
            batch, nodes = [], 0
        batch.append(graph)
        nodes += graph.num_nodes
    if batch:
        flush()
    return (
        torch.cat(vectors).numpy(),
        torch.cat(algorithm).numpy(),
        torch.cat(family).numpy(),
    )
