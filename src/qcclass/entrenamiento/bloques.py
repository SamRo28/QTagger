"""Cross-validation over train + val of ``aleatorio`` with the blocks of ``aleatorio_bloques.csv``.

:func:`load_pool` returns one row per circuit of train + val (evaluation-only variants left
out) with its labels, its targets (``y_algoritmo``, ``y_familia``, codes of
``etiquetas.json``) and its block. Reading goes through :func:`qcclass.versionado.split_ids`,
so the dataset fingerprints are checked and the test set is never touched.

:func:`fold_training` applies, inside the training part of each block, the same balancing as
``balanceo_aleatorio.json`` (``splits.yaml: balanceo``): with ``muestreo`` or ``combinado``
the train groups of every label are capped at ``factor_mediana`` x the median (seeded
sample); with ``ponderacion`` or ``combinado`` the loss weights are inversely proportional to
the remaining train groups of each class (mean 1). The held-out block is always kept whole.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qcclass.config import DatasetConfig, SplitsConfig
from qcclass.data.particiones import (
    CV_PROTOCOL,
    TRAIN,
    blocks_path,
    candidate_rows,
    cap_train_groups,
    class_weights,
    resolve_cap,
)
from qcclass.formatos.comun import LabelVocabulary
from qcclass.metadatos import METADATA_COLUMNS, read_table
from qcclass.versionado import LABEL_VOCABULARY_FILE, split_ids

HEAD_COLUMNS = {"algoritmo": "clase", "familia": "familia"}  # label column of each head


@dataclass(frozen=True)
class CrossValidationPool:
    """Circuits of train + val with targets and blocks, and the label vocabulary."""

    rows: pd.DataFrame
    vocabulary: LabelVocabulary
    n_blocks: int

    def class_names(self, head: str) -> tuple[str, ...]:
        """Class names of a head, in target-index order."""
        return self.vocabulary.classes if head == "algoritmo" else self.vocabulary.families


@dataclass(frozen=True)
class FoldTraining:
    """Training rows (after the cap) and class weights of one block."""

    block: int
    train_index: np.ndarray
    held_out_index: np.ndarray
    weights: dict[str, dict[str, float]]


def load_pool(config: DatasetConfig, splits: SplitsConfig, experiment: str) -> CrossValidationPool:
    """Train + val of :data:`CV_PROTOCOL` with targets and blocks (see the module docstring).

    Raises:
        ValueError: if a circuit of the pool has no block.
    """
    table = split_ids(config, CV_PROTOCOL, ["train", "val"], experiment=experiment)
    table = table[~table["solo_evaluacion"]]
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    rows = candidate_rows(config, splits, metadata)
    rows = rows[rows["id"].isin(table["id"])]
    blocks = pd.read_csv(blocks_path(config), dtype={"group_id": str, "bloque": int})
    rows = rows.merge(blocks, on="group_id", how="left", validate="many_to_one")
    if rows["bloque"].isna().any():
        missing = rows.loc[rows["bloque"].isna(), "group_id"].unique()[:5].tolist()
        raise ValueError(f"groups of the pool without a block: {missing}")
    vocabulary = LabelVocabulary.from_json(
        json.loads((config.rutas.graphs / LABEL_VOCABULARY_FILE).read_text(encoding="utf-8"))
    )
    targets = np.array([vocabulary.targets(row) for row in rows.to_dict("records")], dtype=int)
    rows = rows.assign(
        bloque=rows["bloque"].astype(int),
        y_algoritmo=targets[:, 0],
        y_familia=targets[:, 1],
    )
    rows = rows.sort_values("id").reset_index(drop=True)
    return CrossValidationPool(rows, vocabulary, splits.bloques_cv)


def fold_training(pool: CrossValidationPool, block: int, splits: SplitsConfig) -> FoldTraining:
    """Training rows and class weights of ``block`` (see the module docstring)."""
    rows = pool.rows
    in_train = (rows["bloque"] != block).to_numpy()
    sets = pd.Series(np.where(in_train, TRAIN, None), index=rows.index, dtype=object)
    mode = splits.balanceo.modo
    if mode in ("muestreo", "combinado"):
        cap = resolve_cap(splits, rows, sets)
        sets = cap_train_groups(rows, sets, cap, splits, f"{CV_PROTOCOL}_bloque{block}")
    weights = class_weights(rows, sets) if mode in ("ponderacion", "combinado") else {}
    return FoldTraining(
        block=block,
        train_index=np.flatnonzero((sets == TRAIN).to_numpy()),
        held_out_index=np.flatnonzero(~in_train),
        weights=weights,
    )


def head_index(pool: CrossValidationPool, index: np.ndarray, head: str) -> np.ndarray:
    """Positions of ``index`` whose ``head`` target is a real class (>= 0)."""
    targets = pool.rows[f"y_{head}"].to_numpy()
    return index[targets[index] >= 0]


def sample_weights(
    pool: CrossValidationPool, index: np.ndarray, head: str, fold: FoldTraining
) -> np.ndarray | None:
    """Per-row loss weights of ``head`` for the rows ``index`` (``None`` without weights)."""
    table = fold.weights.get(head)
    if not table:
        return None
    labels = pool.rows[HEAD_COLUMNS[head]].to_numpy()[index]
    return np.array([table.get(label, 1.0) for label in labels], dtype=float)
