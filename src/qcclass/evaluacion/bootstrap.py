"""Group bootstrap: confidence intervals, paired differences and Holm's correction.

The unit resampled is the base circuit (``group_id``): a resample draws as many groups as
there are, with replacement, and keeps every row of each drawn group as many times as it was
drawn. Variants of one circuit are therefore never treated as independent observations.

Implementation: the confusion counts (and top-k hits) of every group are computed once; a
resample is a vector of multinomial group counts, so its confusion matrix is a weighted sum
of the group matrices. All metrics then come from :func:`metricas.scores_from_confusion`.
Intervals are percentile intervals. The paired bootstrap applies the same resampled groups to
two models evaluated on the same rows; its two-sided p-value is
``min(1, 2 * min(P(d <= 0), P(d >= 0)))`` over the resampled differences ``d``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qcclass.evaluacion.metricas import METRICS, scores_from_confusion

CHUNK = 500  # resamples per matrix product


@dataclass(frozen=True)
class GroupCounts:
    """Per-group confusion counts (``G x K*K``) and top-k hits/rows (``G``)."""

    groups: np.ndarray
    confusion: np.ndarray
    hits: np.ndarray | None
    rows: np.ndarray
    n_classes: int


def group_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    top_hits: np.ndarray | None = None,
) -> GroupCounts:
    """Aggregate the rows of every group (groups sorted by name)."""
    names, index = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    cell = np.asarray(y_true, dtype=int) * n_classes + np.asarray(y_pred, dtype=int)
    confusion = np.zeros((len(names), n_classes * n_classes))
    np.add.at(confusion, (index, cell), 1)
    rows = np.bincount(index, minlength=len(names)).astype(float)
    hits = None
    if top_hits is not None:
        hits = np.bincount(index, weights=np.asarray(top_hits, float), minlength=len(names))
    return GroupCounts(names, confusion, hits, rows, n_classes)


def _resample_weights(n_groups: int, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=size).astype(float)


def _scores(counts: GroupCounts, weights: np.ndarray, top_k_name: str | None) -> dict:
    k = counts.n_classes
    matrices = (weights @ counts.confusion).reshape(len(weights), k, k)
    scores = scores_from_confusion(matrices)
    if counts.hits is not None and top_k_name:
        scores[top_k_name] = (weights @ counts.hits) / (weights @ counts.rows)
    return scores


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_intervals(
    counts: GroupCounts,
    *,
    resamples: int,
    level: float,
    seed: int,
    top_k_name: str | None = None,
) -> dict[str, dict[str, float]]:
    """Point estimate and percentile interval of every metric: ``{metric: {valor, ic_inf,
    ic_sup}}``."""
    rng = np.random.default_rng(seed)
    point = _scores(counts, np.ones((1, len(counts.groups))), top_k_name)
    samples: dict[str, list[np.ndarray]] = {name: [] for name in point}
    for start in range(0, resamples, CHUNK):
        weights = _resample_weights(len(counts.groups), min(CHUNK, resamples - start), rng)
        for name, values in _scores(counts, weights, top_k_name).items():
            samples[name].append(values)
    result = {}
    for name, values in samples.items():
        low, high = _interval(np.concatenate(values), level)
        result[name] = {"valor": float(point[name][0]), "ic_inf": low, "ic_sup": high}
    return result


def paired_bootstrap(
    first: GroupCounts,
    second: GroupCounts,
    *,
    metric: str = "f1_macro",
    resamples: int,
    level: float,
    seed: int,
) -> dict[str, float]:
    """Difference ``first - second`` of ``metric`` with a paired group bootstrap.

    Raises:
        ValueError: if the two models were not evaluated on the same groups and rows.
    """
    if metric not in METRICS:
        raise ValueError(f"paired bootstrap supports {METRICS}, not '{metric}'")
    if not np.array_equal(first.groups, second.groups) or not np.array_equal(
        first.rows, second.rows
    ):
        raise ValueError("a paired bootstrap needs both models on the same groups and rows")
    rng = np.random.default_rng(seed)
    ones = np.ones((1, len(first.groups)))
    point = _scores(first, ones, None)[metric][0] - _scores(second, ones, None)[metric][0]
    differences = []
    for start in range(0, resamples, CHUNK):
        weights = _resample_weights(len(first.groups), min(CHUNK, resamples - start), rng)
        differences.append(
            _scores(first, weights, None)[metric] - _scores(second, weights, None)[metric]
        )
    values = np.concatenate(differences)
    low, high = _interval(values, level)
    p_value = min(1.0, 2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
    return {"diferencia": float(point), "ic_inf": low, "ic_sup": high, "p_valor": p_value}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()
