"""Classification metrics computed from confusion matrices.

Every metric is derived from the confusion matrix (rows = true class, columns = predicted
class), vectorised over a batch of matrices so that the same code gives the point estimate
and the bootstrap resamples. Macro averages (F1 and balanced accuracy) are taken over the
classes present in the evaluated rows (support > 0): a class absent from a subset or a
resample neither helps nor penalises it. Predictions of absent classes still count as
errors of the true class.

* exactitud: correct / total.
* exactitud_balanceada: mean recall over the present classes.
* f1_macro: mean over the present classes of ``2 TP / (2 TP + FP + FN)`` (0 when TP = 0).
* mcc: multiclass Matthews correlation (Gorodkin), 0 when undefined, as scikit-learn.
* top-k: fraction of rows whose true class is among the ``k`` most probable.
"""

import numpy as np

METRICS = ("exactitud", "exactitud_balanceada", "f1_macro", "mcc")
PRIMARY_METRIC = "f1_macro"


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``n_classes x n_classes`` confusion matrix (rows = true, columns = predicted)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    if len(y_true) and (
        min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= n_classes
    ):
        raise ValueError(f"labels must be in [0, {n_classes})")
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def scores_from_confusion(matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics of a batch of confusion matrices ``(B, K, K)`` (or one ``(K, K)``).

    Returns arrays of shape ``(B,)`` (or scalars for a single matrix).
    """
    single = matrices.ndim == 2
    cm = np.asarray(matrices, dtype=np.float64)
    if single:
        cm = cm[None]
    true_counts = cm.sum(axis=2)
    predicted = cm.sum(axis=1)
    hits = np.diagonal(cm, axis1=1, axis2=2)
    total = true_counts.sum(axis=1)
    present = true_counts > 0
    n_present = present.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(present, hits / true_counts, 0.0)
        f1 = np.where(present, 2 * hits / (true_counts + predicted), 0.0)
        accuracy = np.where(total > 0, hits.sum(axis=1) / total, np.nan)
        balanced = np.where(n_present > 0, recall.sum(axis=1) / n_present, np.nan)
        f1_macro = np.where(n_present > 0, f1.sum(axis=1) / n_present, np.nan)
        correct = hits.sum(axis=1)
        numerator = correct * total - (predicted * true_counts).sum(axis=1)
        denominator = np.sqrt(
            (total**2 - (predicted**2).sum(axis=1)) * (total**2 - (true_counts**2).sum(axis=1))
        )
        mcc = np.where(denominator > 0, numerator / denominator, 0.0)
    scores = {
        "exactitud": accuracy,
        "exactitud_balanceada": balanced,
        "f1_macro": f1_macro,
        "mcc": mcc,
    }
    return {name: float(v[0]) for name, v in scores.items()} if single else scores


def per_class(matrix: np.ndarray, class_names: tuple[str, ...]) -> list[dict]:
    """Precision, recall, F1 and support of every class (0 where undefined)."""
    cm = np.asarray(matrix, dtype=np.float64)
    true_counts, predicted, hits = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    rows = []
    for index, name in enumerate(class_names):
        precision = hits[index] / predicted[index] if predicted[index] else 0.0
        recall = hits[index] / true_counts[index] if true_counts[index] else 0.0
        denominator = true_counts[index] + predicted[index]
        rows.append(
            {
                "clase": name,
                "precision": float(precision),
                "exhaustividad": float(recall),
                "f1": float(2 * hits[index] / denominator) if denominator else 0.0,
                "soporte": int(true_counts[index]),
            }
        )
    return rows


def top_k_hits(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> np.ndarray:
    """Boolean array: is the true class among the ``k`` most probable classes of each row?"""
    probabilities = np.asarray(probabilities)
    k = min(k, probabilities.shape[1])
    top = np.argpartition(-probabilities, k - 1, axis=1)[:, :k]
    return (top == np.asarray(y_true, dtype=int)[:, None]).any(axis=1)
