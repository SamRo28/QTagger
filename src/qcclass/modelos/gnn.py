"""GNN of the phase 11 (``GNNConfig`` + :class:`CircuitGNN`) and its multi-task loss.

Architecture (every layer reads ``edge_attr``):

1. ``entrada``: ``Linear(node_dim -> dim)``.
2. ``capas`` message-passing layers, ``tipo_capa``:
   * ``gine``: ``GINEConv`` with ``nn = Linear, ReLU, Linear`` (``edge_dim`` projected);
   * ``gatv2``: ``GATv2Conv`` with ``cabezas_atencion`` heads of ``dim / heads`` channels,
     concatenated (self loops with the mean edge attribute).
   Each layer: ``m = Dropout(ReLU(BatchNorm(conv(h))))`` and ``h = h + m`` (residual). With
   ``nodo_virtual``, a per-graph vector is added to every node before each layer and updated
   after it: ``vn = vn + MLP(mean_pool(h) + vn)`` (MLP with LayerNorm: a batch may hold one
   graph).
3. Jumping Knowledge (``jumping_knowledge``, on by default): the outputs of every layer are
   concatenated and projected back to ``dim``; without it, the last layer is used.
4. ``agregacion``: ``media_max`` (mean and max pooling concatenated, ``2 * dim``) or
   ``atencion`` (softmax gate per graph over ``Linear(dim -> 1)``, weighted mean, ``dim``).
   No sum pooling: graph sizes vary over orders of magnitude.
5. Two heads ``Linear, ReLU, Dropout, Linear``: algorithm (``num_classes``) and family
   (``num_families``).

``forward(batch, trazas=True)`` also returns the traces used by explainability, the unknown
detection and the visual project: node vectors after the input and after every layer, the
virtual-node states, the Jumping Knowledge output, the graph summary vector, logits and
probabilities of both heads and the attention weights (GATv2 per layer; pooling gate).

Loss: weighted cross-entropy of the algorithm (``-1`` ignored) plus ``lambda_familia`` times
the weighted cross-entropy of the family; a ``-2`` target raises.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GINEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter, softmax

from qcclass.formatos.comun import NO_LABEL, UNKNOWN

LAYER_TYPES = ("gine", "gatv2")
POOLINGS = ("media_max", "atencion")


@dataclass(frozen=True)
class GNNConfig:
    """Hyperparameters of :class:`CircuitGNN` (input sizes come from the data)."""

    node_dim: int
    edge_dim: int
    num_classes: int
    num_families: int
    tipo_capa: str = "gine"
    dim: int = 64
    capas: int = 3
    dropout: float = 0.0
    nodo_virtual: bool = False
    agregacion: str = "media_max"
    jumping_knowledge: bool = True
    cabezas_atencion: int = 4

    def __post_init__(self) -> None:
        if self.tipo_capa not in LAYER_TYPES:
            raise ValueError(f"tipo_capa must be one of {LAYER_TYPES}, got {self.tipo_capa!r}")
        if self.agregacion not in POOLINGS:
            raise ValueError(f"agregacion must be one of {POOLINGS}, got {self.agregacion!r}")
        if self.capas < 1 or self.dim < 1:
            raise ValueError("capas and dim must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.tipo_capa == "gatv2" and self.dim % self.cabezas_atencion:
            raise ValueError("dim must be a multiple of cabezas_atencion for gatv2")

    @classmethod
    def from_mapping(cls, values: dict[str, Any], **sizes: int) -> "GNNConfig":
        """Build from a configuration mapping plus the data sizes; unknown keys raise."""
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(values) - known):
            raise ValueError(f"unknown model settings {unknown}")
        return cls(**{**values, **sizes})

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary (checkpoints, MLflow)."""
        return asdict(self)


@dataclass
class GNNOutput:
    """Logits of both heads and, on request, the traces."""

    algoritmo: Tensor
    familia: Tensor
    trazas: dict[str, Any] | None = None


def _mlp(inputs: int, hidden: int, outputs: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, outputs)
    )


class CircuitGNN(nn.Module):
    """Graph classifier of the circuits (see the module docstring)."""

    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.entrada = nn.Linear(c.node_dim, c.dim)
        self.convs = nn.ModuleList(self._conv() for _ in range(c.capas))
        self.norms = nn.ModuleList(nn.BatchNorm1d(c.dim) for _ in range(c.capas))
        self.dropout = nn.Dropout(c.dropout)
        if c.nodo_virtual:
            self.vn_inicial = nn.Parameter(torch.zeros(c.dim))
            self.vn_mlps = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(c.dim, c.dim), nn.LayerNorm(c.dim), nn.ReLU(), nn.Dropout(c.dropout)
                )
                for _ in range(c.capas - 1)
            )
        if c.jumping_knowledge:
            self.jk = nn.Linear(c.capas * c.dim, c.dim)
        if c.agregacion == "atencion":
            self.puerta = nn.Linear(c.dim, 1)
        pooled = 2 * c.dim if c.agregacion == "media_max" else c.dim
        self.cabeza_algoritmo = _mlp(pooled, c.dim, c.num_classes, c.dropout)
        self.cabeza_familia = _mlp(pooled, c.dim, c.num_families, c.dropout)

    def _conv(self) -> nn.Module:
        c = self.config
        if c.tipo_capa == "gine":
            return GINEConv(
                nn.Sequential(nn.Linear(c.dim, c.dim), nn.ReLU(), nn.Linear(c.dim, c.dim)),
                edge_dim=c.edge_dim,
            )
        return GATv2Conv(
            c.dim,
            c.dim // c.cabezas_atencion,
            heads=c.cabezas_atencion,
            concat=True,
            edge_dim=c.edge_dim,
            add_self_loops=True,
            fill_value="mean",
        )

    def forward(self, batch: Batch, *, trazas: bool = False) -> GNNOutput:
        """Logits of every graph of the batch (and the traces when ``trazas``)."""
        c = self.config
        index, graphs = batch.batch, batch.num_graphs
        h = self.entrada(batch.x)
        traces: dict[str, Any] = {"nodos": [h], "nodo_virtual": None, "atencion_capas": None}
        vn = None
        if c.nodo_virtual:
            vn = self.vn_inicial.expand(graphs, -1)
            traces["nodo_virtual"] = [vn]
        attention = [] if c.tipo_capa == "gatv2" and trazas else None
        outputs = []
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            h_in = h + vn[index] if vn is not None else h
            if attention is not None:
                message, (edges, alpha) = conv(
                    h_in, batch.edge_index, batch.edge_attr, return_attention_weights=True
                )
                attention.append({"edge_index": edges, "pesos": alpha})
            else:
                message = conv(h_in, batch.edge_index, batch.edge_attr)
            h = h_in + self.dropout(functional.relu(norm(message)))
            outputs.append(h)
            traces["nodos"].append(h)
            if vn is not None and layer < c.capas - 1:
                pooled = global_mean_pool(h, index, size=graphs)
                vn = vn + self.vn_mlps[layer](pooled + vn)
                traces["nodo_virtual"].append(vn)
        traces["atencion_capas"] = attention
        nodes = self.jk(torch.cat(outputs, dim=1)) if c.jumping_knowledge else h
        traces["jumping_knowledge"] = nodes if c.jumping_knowledge else None
        summary, gate = self._pool(nodes, index, graphs)
        traces["atencion_agregacion"] = gate
        traces["resumen"] = summary
        output = GNNOutput(self.cabeza_algoritmo(summary), self.cabeza_familia(summary))
        if trazas:
            traces["logits_algoritmo"] = output.algoritmo
            traces["logits_familia"] = output.familia
            traces["prob_algoritmo"] = output.algoritmo.float().softmax(dim=1)
            traces["prob_familia"] = output.familia.float().softmax(dim=1)
            output.trazas = traces
        return output

    def _pool(self, h: Tensor, index: Tensor, graphs: int) -> tuple[Tensor, Tensor | None]:
        # ``size`` keeps one row per graph even for graphs without nodes
        if self.config.agregacion == "media_max":
            pooled = torch.cat(
                [global_mean_pool(h, index, size=graphs), global_max_pool(h, index, size=graphs)],
                dim=1,
            )
            return pooled, None
        weights = softmax(self.puerta(h).squeeze(-1).float(), index, num_nodes=graphs).to(h.dtype)
        pooled = scatter(weights.unsqueeze(-1) * h, index, dim=0, dim_size=graphs, reduce="sum")
        return pooled, weights

    def describe(self) -> dict[str, Any]:
        """Layers with their input/output sizes and parameters, plus totals."""
        c = self.config
        pooled = 2 * c.dim if c.agregacion == "media_max" else c.dim
        rows = [("entrada", "Linear", c.node_dim, c.dim, self.entrada)]
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            rows.append((f"capa_{i + 1}", type(conv).__name__, c.dim, c.dim, conv))
            rows.append((f"norma_{i + 1}", "BatchNorm1d", c.dim, c.dim, norm))
        if c.nodo_virtual:
            for i, mlp in enumerate(self.vn_mlps):
                rows.append((f"nodo_virtual_{i + 1}", "MLP", c.dim, c.dim, mlp))
        if c.jumping_knowledge:
            rows.append(("jumping_knowledge", "Linear", c.capas * c.dim, c.dim, self.jk))
        if c.agregacion == "atencion":
            rows.append(("agregacion", "atencion", c.dim, c.dim, self.puerta))
        rows.append(("cabeza_algoritmo", "MLP", pooled, c.num_classes, self.cabeza_algoritmo))
        rows.append(("cabeza_familia", "MLP", pooled, c.num_families, self.cabeza_familia))
        layers = [
            {
                "nombre": name,
                "tipo": kind,
                "entrada": inputs,
                "salida": outputs,
                "parametros": sum(p.numel() for p in module.parameters()),
            }
            for name, kind, inputs, outputs, module in rows
        ]
        if c.nodo_virtual:
            layers.insert(
                1,
                {
                    "nombre": "nodo_virtual_inicial",
                    "tipo": "Parameter",
                    "entrada": 0,
                    "salida": c.dim,
                    "parametros": c.dim,
                },
            )
        total = sum(p.numel() for p in self.parameters())
        assert total == sum(layer["parametros"] for layer in layers), "describe() misses a layer"
        return {"configuracion": c.as_dict(), "capas": layers, "parametros": total}


@dataclass
class LossSums:
    """Weighted sums of a (micro-)batch: ``sum_i w_{y_i} * CE_i`` and ``sum_i w_{y_i}`` of each
    head (algorithm rows with -1 excluded)."""

    algoritmo: Tensor
    peso_algoritmo: Tensor
    familia: Tensor
    peso_familia: Tensor


class MultiTaskLoss(nn.Module):
    """``CE_alg(ignore -1) + lambda_familia * CE_fam``, both with class weights.

    Each term is a weighted mean, ``sum_i w_{y_i} CE_i / sum_i w_{y_i}`` (as PyTorch's
    ``cross_entropy`` with ``weight``). For gradient accumulation over micro-batches,
    :meth:`sums` gives the numerators and denominators of a micro-batch and
    :meth:`from_sums` divides by the totals of the whole batch, so the accumulated gradient is
    that of the whole batch at once.
    """

    def __init__(
        self, algorithm_weights: Tensor, family_weights: Tensor, lambda_familia: float
    ) -> None:
        super().__init__()
        self.register_buffer("algorithm_weights", algorithm_weights.float())
        self.register_buffer("family_weights", family_weights.float())
        self.lambda_familia = float(lambda_familia)

    @staticmethod
    def check_targets(y_algorithm: Tensor, y_family: Tensor) -> None:
        """Raises ``ValueError`` if a target is ``UNKNOWN`` (-2), never trained on."""
        if bool((y_algorithm == UNKNOWN).any()) or bool((y_family == UNKNOWN).any()):
            raise ValueError("an example coded as unknown (-2) reached the training loss")

    def weight_totals(self, y_algorithm: Tensor, y_family: Tensor) -> tuple[float, float]:
        """``sum_i w_{y_i}`` of each head over the rows given (any device)."""
        self.check_targets(y_algorithm, y_family)
        labelled = y_algorithm[y_algorithm != NO_LABEL]
        algorithm = self.algorithm_weights.to(labelled.device)[labelled].sum()
        family = self.family_weights.to(y_family.device)[y_family].sum()
        return float(algorithm), float(family)

    def sums(self, output: GNNOutput, y_algorithm: Tensor, y_family: Tensor) -> LossSums:
        """Weighted sums of a (micro-)batch."""
        self.check_targets(y_algorithm, y_family)
        labelled = y_algorithm != NO_LABEL
        zero = torch.zeros((), device=output.algoritmo.device)
        algorithm, algorithm_weight = zero, zero
        if bool(labelled.any()):
            algorithm = functional.cross_entropy(
                output.algoritmo.float(),
                y_algorithm,
                weight=self.algorithm_weights,
                ignore_index=NO_LABEL,
                reduction="sum",
            )
            algorithm_weight = self.algorithm_weights[y_algorithm[labelled]].sum()
        family, family_weight = zero, zero
        if self.lambda_familia > 0:
            family = functional.cross_entropy(
                output.familia.float(), y_family, weight=self.family_weights, reduction="sum"
            )
            family_weight = self.family_weights[y_family].sum()
        return LossSums(algorithm, algorithm_weight, family, family_weight)

    def from_sums(self, sums: LossSums, algorithm_total: float, family_total: float) -> Tensor:
        """Share of the whole-batch loss carried by a micro-batch (totals of the whole batch)."""
        loss = sums.algoritmo / algorithm_total if algorithm_total > 0 else sums.algoritmo * 0
        if self.lambda_familia > 0 and family_total > 0:
            loss = loss + self.lambda_familia * sums.familia / family_total
        return loss

    def forward(
        self, output: GNNOutput, y_algorithm: Tensor, y_family: Tensor
    ) -> tuple[Tensor, float, float]:
        """(total, algorithm term, family term) of a whole batch.

        Raises:
            ValueError: if a target is ``UNKNOWN`` (-2), which must never be trained on.
        """
        sums = self.sums(output, y_algorithm, y_family)
        algorithm_total, family_total = float(sums.peso_algoritmo), float(sums.peso_familia)
        total = self.from_sums(sums, algorithm_total, family_total)
        algorithm = float(sums.algoritmo.detach()) / algorithm_total if algorithm_total else 0.0
        family = float(sums.familia.detach()) / family_total if family_total else 0.0
        return total, algorithm, family
