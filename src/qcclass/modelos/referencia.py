"""Reference GNN used to measure training cost (not the final architecture).

Input ``Linear(node_dim -> dim)``; ``layers`` GINEConv layers (``nn`` = Linear, ReLU, Linear;
``edge_dim`` = width of ``edge_attr``), each followed by BatchNorm and ReLU and added to its
input (residual); global pooling = concatenation of mean and max (``2 * dim``); two MLP heads
``2*dim -> dim -> classes`` (algorithm) and ``2*dim -> dim -> families`` (family).

Loss: weighted cross-entropy of the algorithm (``NO_LABEL`` = -1 ignored) plus weighted
cross-entropy of the family, with the class weights of ``balanceo_<protocol>.json``.
Examples coded ``UNKNOWN`` (-2) must never reach the loss: it raises if it sees one.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from qcclass.formatos.comun import NO_LABEL, UNKNOWN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Labels:
    """Classes and families of ``etiquetas.json``, in target-index order."""

    classes: tuple[str, ...]
    families: tuple[str, ...]
    unknown: tuple[str, ...]  # algorithms coded -2 in the algorithm head (reserved + unknown)

    @classmethod
    def read(cls, path: Path) -> "Labels":
        """Read ``etiquetas.json``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = (*data["reservados"], *data["desconocidos_sin_familia"])
        return cls(tuple(data["clases"]), tuple(data["familias"]), tuple(unknown))


class ReferenceGNN(nn.Module):
    """GINEConv baseline with an algorithm head and a family head."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_classes: int,
        num_families: int,
        *,
        dim: int = 64,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(node_dim, dim)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)),
                edge_dim=edge_dim,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for _ in range(layers))
        self.algorithm_head = _mlp(2 * dim, dim, num_classes)
        self.family_head = _mlp(2 * dim, dim, num_families)

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Algorithm and family logits of every graph of the batch."""
        h = self.input(batch.x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = h + functional.relu(norm(conv(h, batch.edge_index, batch.edge_attr)))
        # ``size`` keeps one row per graph even for graphs without nodes
        pooled = torch.cat(
            [
                global_mean_pool(h, batch.batch, size=batch.num_graphs),
                global_max_pool(h, batch.batch, size=batch.num_graphs),
            ],
            dim=1,
        )
        return self.algorithm_head(pooled), self.family_head(pooled)


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def class_weight_tensors(balance_file: Path, labels: Labels) -> tuple[Tensor, Tensor]:
    """Algorithm and family weights of ``balanceo_<protocol>.json`` in target-index order.

    A label without a weight gets 1.0 (with a warning).
    """
    weights = json.loads(Path(balance_file).read_text(encoding="utf-8"))["pesos"]

    def ordered(names: tuple[str, ...], table: dict[str, float]) -> Tensor:
        if missing := [name for name in names if name not in table]:
            logger.warning("no balancing weight for %s: using 1.0", missing)
        return torch.tensor([table.get(name, 1.0) for name in names], dtype=torch.float32)

    return ordered(labels.classes, weights["algoritmo"]), ordered(
        labels.families, weights["familia"]
    )


@dataclass(frozen=True)
class LossParts:
    """Total loss and its two terms (detached terms, for logging)."""

    total: Tensor
    algorithm: float
    family: float


class MultiTaskLoss(nn.Module):
    """Weighted cross-entropy of the algorithm (ignoring -1) plus that of the family."""

    def __init__(self, algorithm_weights: Tensor, family_weights: Tensor) -> None:
        super().__init__()
        self.register_buffer("algorithm_weights", algorithm_weights)
        self.register_buffer("family_weights", family_weights)

    def forward(
        self, logits: tuple[Tensor, Tensor], y_algorithm: Tensor, y_family: Tensor
    ) -> LossParts:
        """Loss of a batch.

        Raises:
            ValueError: if a target is ``UNKNOWN`` (-2), which must never be trained on.
        """
        if bool((y_algorithm == UNKNOWN).any()) or bool((y_family == UNKNOWN).any()):
            raise ValueError("an example coded as unknown (-2) reached the training loss")
        algorithm_logits, family_logits = logits
        algorithm = torch.zeros((), device=algorithm_logits.device)
        if bool((y_algorithm != NO_LABEL).any()):  # a batch of family-only graphs has no term
            algorithm = functional.cross_entropy(
                algorithm_logits.float(),
                y_algorithm,
                weight=self.algorithm_weights,
                ignore_index=NO_LABEL,
            )
        family = functional.cross_entropy(
            family_logits.float(), y_family, weight=self.family_weights
        )
        return LossParts(algorithm + family, float(algorithm.detach()), float(family.detach()))
