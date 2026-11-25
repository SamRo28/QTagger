"""Gradient accumulation over micro-batches, with an automatic retry when the GPU runs out.

A training batch holds up to ``presupuesto_nodos`` nodes (100 000). When that does not fit in
the GPU, it is split into consecutive micro-batches of at most ``micro_nodes`` nodes and the
gradients are accumulated, so that every optimiser step still sees the whole batch:

* the loss of each micro-batch is a weighted **sum** (class weights; algorithm rows with -1
  excluded) divided by the total weight of the **whole batch** (computed from its targets
  before any forward pass), so the accumulated gradient equals that of the whole batch at
  once (up to rounding; BatchNorm statistics are the only part computed per micro-batch);
* with mixed precision, every micro-batch is scaled and back-propagated, and the optimiser
  and the gradient scaler take **one** step per whole batch;
* if a micro-batch raises ``torch.OutOfMemoryError``, the gradients of the batch are
  discarded, the cache is emptied and the **same batch** is retried with micro-batches of
  half the size, down to ``min_micro_nodes`` (20 000, above the largest graph). The smaller
  size is kept for the rest of the run. If even that does not fit, the error is raised and
  the run (or the Optuna trial) fails.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Batch

from qcclass.modelos.gnn import CircuitGNN, MultiTaskLoss

logger = logging.getLogger(__name__)


def split_batch(batch: Batch, max_nodes: int) -> list[Batch]:
    """Consecutive groups of graphs of ``batch`` with at most ``max_nodes`` nodes each (a graph
    larger than that forms a group by itself); the batch itself when it already fits."""
    if batch.num_nodes <= max_nodes:
        return [batch]
    sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    groups, current, total = [], [], 0
    for index, size in enumerate(sizes):
        if current and total + size > max_nodes:
            groups.append(current)
            current, total = [], 0
        current.append(index)
        total += size
    if current:
        groups.append(current)
    return [Batch.from_data_list(batch.index_select(group)) for group in groups]


@dataclass
class StepResult:
    """Losses of a whole batch (means over its weights) and the micro-batches used."""

    total: float
    algoritmo: float
    familia: float
    microlotes: int


class AccumulatingStep:
    """One optimiser step per whole batch, accumulating over micro-batches (module docstring)."""

    def __init__(
        self,
        model: CircuitGNN,
        loss: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        *,
        device: str,
        amp: bool,
        micro_nodes: int,
        min_micro_nodes: int,
    ) -> None:
        if micro_nodes < min_micro_nodes:
            raise ValueError("micro_nodes must be at least min_micro_nodes")
        self.model, self.loss, self.optimizer, self.scaler = model, loss, optimizer, scaler
        self.device, self.amp = device, amp
        self.micro_nodes, self.min_micro_nodes = int(micro_nodes), int(min_micro_nodes)

    def _accumulate(self, batch: Batch) -> StepResult:
        algorithm_total, family_total = self.loss.weight_totals(batch.y_algoritmo, batch.y_familia)
        sums = np.zeros(2)
        parts = split_batch(batch, self.micro_nodes)
        for part in parts:
            part = part.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                output = self.model(part)
            loss_sums = self.loss.sums(output, part.y_algoritmo, part.y_familia)
            share = self.loss.from_sums(loss_sums, algorithm_total, family_total)
            self.scaler.scale(share).backward()
            sums += [float(loss_sums.algoritmo.detach()), float(loss_sums.familia.detach())]
        algorithm = sums[0] / algorithm_total if algorithm_total > 0 else 0.0
        family = sums[1] / family_total if family_total > 0 else 0.0
        total = algorithm + (self.loss.lambda_familia * family if family_total > 0 else 0.0)
        return StepResult(total, algorithm, family, len(parts))

    def __call__(self, batch: Batch) -> StepResult:
        """Train on ``batch`` (on the CPU; moved micro-batch by micro-batch) and step."""
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                result = self._accumulate(batch)
            except torch.OutOfMemoryError:
                self.optimizer.zero_grad(set_to_none=True)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if self.micro_nodes <= self.min_micro_nodes:
                    logger.error(
                        "out of GPU memory with micro-batches of %d nodes", self.micro_nodes
                    )
                    raise
                self.micro_nodes = max(self.micro_nodes // 2, self.min_micro_nodes)
                logger.warning(
                    "out of GPU memory: retrying with micro-batches of %d nodes", self.micro_nodes
                )
                continue
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return result
