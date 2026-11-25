"""Batches limited by a node budget, built from the graph sizes in ``indice.csv``.

:class:`NodeBudgetBatchSampler` is equivalent to PyG's ``DynamicBatchSampler`` in ``"node"``
mode (with ``skip_too_big=False``): graphs are taken in a shuffled order and added to the
current batch while the batch stays within the budget; a graph that alone exceeds the
budget forms a batch by itself. Unlike PyG's sampler, it never loads a graph to learn its
size: sizes come from the ``n_nodos`` column of ``<graphs>/dag/indice.csv``.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from qcclass.circuitos import write_text_atomic
from qcclass.formatos.grafo_dag import INDEX_FILE, load_graph

logger = logging.getLogger(__name__)

NODES_COLUMN = "n_nodos"


def graph_sizes(directory: Path) -> pd.Series:
    """Number of nodes of every graph of ``directory``, indexed by ``id``.

    Read from the ``n_nodos`` column of ``indice.csv``. If an older index lacks it, every
    graph is loaded once, the column is added to ``indice.csv`` and later calls read it.

    Raises:
        ValueError: if ``indice.csv`` does not exist (graphs not copied to this machine).
    """
    path = Path(directory) / INDEX_FILE
    if not path.is_file():
        raise ValueError(
            f"{path} does not exist: the graphs are missing on this machine (copy data/graphs "
            "with rsync, see docs/EQUIPOS.md; qcclass verificar does not check them)"
        )
    index = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if NODES_COLUMN not in index.columns:
        logger.warning("%s has no %s column: computing it once", path, NODES_COLUMN)
        index[NODES_COLUMN] = [
            load_graph(Path(directory) / f"{circuit_id}.pt").num_nodes
            for circuit_id in tqdm(index["id"], desc="n_nodos")
        ]
        write_text_atomic(index.to_csv(index=False), path)
    return index.set_index("id")[NODES_COLUMN].astype(int)


def budget_batches(sizes: Sequence[int], order: Sequence[int], max_nodes: int) -> list[list[int]]:
    """Greedy batches of ``order`` (positions into ``sizes``) within ``max_nodes`` nodes."""
    batches: list[list[int]] = []
    current: list[int] = []
    total = 0
    for position in order:
        size = int(sizes[position])
        if current and total + size > max_nodes:
            batches.append(current)
            current, total = [], 0
        current.append(int(position))
        total += size
    if current:
        batches.append(current)
    return batches


class NodeBudgetBatchSampler(Sampler[list[int]]):
    """Batch sampler with at most ``max_nodes`` nodes per batch (see the module docstring).

    The order is reshuffled every epoch: each ``__iter__`` advances the epoch, and the
    permutation of epoch ``e`` depends only on ``seed`` and ``e`` (reproducible).
    ``max_batches`` truncates every epoch (quick measurements).
    """

    def __init__(
        self,
        sizes: Sequence[int],
        max_nodes: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        max_batches: int | None = None,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError("the node budget must be positive")
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.max_nodes = max_nodes
        self.shuffle = shuffle
        self.seed = seed
        self.max_batches = max_batches
        self.epoch = 0
        oversized = int((self.sizes > max_nodes).sum())
        if oversized:
            logger.warning("%d graphs exceed the budget of %d nodes alone", oversized, max_nodes)

    def batches(self, epoch: int) -> list[list[int]]:
        """Batches of a given epoch."""
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(self.sizes), generator=generator).tolist()
        else:
            order = list(range(len(self.sizes)))
        batches = budget_batches(self.sizes, order, self.max_nodes)
        return batches[: self.max_batches] if self.max_batches is not None else batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.batches(self.epoch)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches(self.epoch))
