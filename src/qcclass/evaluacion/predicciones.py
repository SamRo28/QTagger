"""Prediction tables, exclusion rules and out-of-fold aggregation.

A prediction table is a DataFrame with one row per evaluated circuit and the columns
``id``, ``group_id`` and ``solo_evaluacion``, plus, for every predicted head (``algoritmo``,
``familia`` or both), ``y_<head>`` and ``pred_<head>``; optionally ``bloque``
(cross-validation block) and the probabilities ``p_<head>_<i>`` (one column per class, in
label-index order).

Exclusion rules (:func:`head_rows`):

* ``-2`` (unknown) never counts in the metrics of known circuits, in either head;
* ``-1`` (family only) counts in the family head only;
* ``solo_evaluacion`` variants are left out of the main metrics.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

HEADS = ("algoritmo", "familia")
REQUIRED_COLUMNS = ("id", "group_id", "solo_evaluacion")
BLOCK_COLUMN = "bloque"


def table_heads(table: pd.DataFrame) -> tuple[str, ...]:
    """Heads predicted in a table (those with a ``pred_<head>`` column)."""
    return tuple(head for head in HEADS if f"pred_{head}" in table.columns)


def check_table(table: pd.DataFrame) -> None:
    """Check the required columns and that every id appears once.

    Raises:
        ValueError: if a column is missing, no head is predicted or an id is repeated.
    """
    heads = table_heads(table)
    if not heads:
        raise ValueError("prediction table without any pred_<head> column")
    required = [*REQUIRED_COLUMNS, *(f"y_{head}" for head in heads)]
    if missing := [c for c in required if c not in table.columns]:
        raise ValueError(f"prediction table without columns {missing}")
    if not table["id"].is_unique:
        repeated = table.loc[table["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(f"repeated ids in the prediction table: {repeated}")


def probability_columns(head: str, n_classes: int) -> list[str]:
    """Names of the probability columns of a head."""
    return [f"p_{head}_{index}" for index in range(n_classes)]


def head_rows(table: pd.DataFrame, head: str) -> pd.DataFrame:
    """Rows that count in the main metrics of ``head`` (see the module docstring)."""
    if head not in HEADS:
        raise ValueError(f"unknown head '{head}' ({HEADS})")
    target = table[f"y_{head}"].astype(int)
    regular = ~table["solo_evaluacion"].astype(bool)
    return table[regular & (target >= 0)]  # drops -2 always and -1 in the algorithm head


def head_arrays(
    table: pd.DataFrame, head: str, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """(true labels, predictions, probabilities or ``None``, groups) of the rows of a head."""
    rows = head_rows(table, head)
    y_true = rows[f"y_{head}"].to_numpy(dtype=int)
    y_pred = rows[f"pred_{head}"].to_numpy(dtype=int)
    columns = probability_columns(head, n_classes)
    probabilities = (
        rows[columns].to_numpy(dtype=float) if all(c in rows.columns for c in columns) else None
    )
    return y_true, y_pred, probabilities, rows["group_id"].to_numpy()


def make_table(
    ids: Sequence[str],
    groups: Sequence[str],
    targets: dict[str, Sequence[int]],
    probabilities: dict[str, np.ndarray],
    *,
    evaluation_only: Sequence[bool] | None = None,
    block: int | None = None,
) -> pd.DataFrame:
    """Prediction table from targets and probabilities per head (``{head: (n, classes)}``),
    predicting the argmax.

    Rows whose target is -1 or -2 keep the argmax too; the exclusion rules decide later
    whether they count.
    """
    table = pd.DataFrame(
        {
            "id": list(ids),
            "group_id": list(groups),
            "solo_evaluacion": (
                list(evaluation_only) if evaluation_only is not None else [False] * len(ids)
            ),
        }
    )
    for head in HEADS:
        if head not in probabilities:
            continue
        table[f"y_{head}"] = np.asarray(targets[head], dtype=int)
        matrix = np.asarray(probabilities[head], dtype=float)
        table[f"pred_{head}"] = matrix.argmax(axis=1)
        table[probability_columns(head, matrix.shape[1])] = matrix
    if block is not None:
        table[BLOCK_COLUMN] = block
    check_table(table)
    return table


def aggregate_out_of_fold(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the held-out predictions of every block into one table.

    Every table must carry its ``bloque``; a circuit may appear in a single block only.

    Raises:
        ValueError: if a block column is missing or a circuit is in two blocks.
    """
    for table in tables:
        if BLOCK_COLUMN not in table.columns:
            raise ValueError("every out-of-fold table needs its 'bloque' column")
    combined = pd.concat(list(tables), ignore_index=True)
    check_table(combined)
    leaked = combined.groupby("group_id")[BLOCK_COLUMN].nunique()
    if (leaked > 1).any():
        raise ValueError(f"groups in several blocks: {leaked[leaked > 1].index[:5].tolist()}")
    return combined


def attach_metadata(table: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Add the breakdown columns (``nivel``, ``n_logico``, ``familia``, ``algoritmo``,
    ``construccion_mcx``, ``tipo_oraculo_dj``) from ``metadata.csv``."""
    from qcclass.data.particiones import logical_sizes

    rows = metadata[metadata["id"].isin(table["id"])].copy()
    rows["n_logico"] = logical_sizes(metadata, rows)
    columns = ["nivel", "n_logico", "familia", "algoritmo", "construccion_mcx", "tipo_oraculo_dj"]
    return table.merge(rows[["id", *columns]], on="id", how="left", validate="one_to_one")
