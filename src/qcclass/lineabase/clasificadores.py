"""Classifiers of the baseline, input preparation and per-block fit/predict.

* ``regresion_logistica``: optional ``log1p``, standardisation (without centring for sparse
  WL input) and multinomial logistic regression with L2 (``lbfgs``).
* ``random_forest``: scikit-learn's random forest (all cores).
* ``hist_gradient_boosting``: histogram gradient boosting without early stopping. It needs
  dense input: on WL only the ``columnas_wl`` columns most frequent in the block's training
  part are kept.

WL input keeps only the columns present in at least ``min_circuitos`` circuits of the block's
training part (a label seen in a single circuit cannot generalise). Every column choice looks
at the training part only, never at the held-out block.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logger = logging.getLogger(__name__)

RUNG_PARAMS = ("h",)  # hyperparameters consumed by the features, not by the classifier
PREPARATION_PARAMS = ("transformacion", "columnas_wl", "min_circuitos")


@dataclass(frozen=True)
class Prepared:
    """Inputs of one block after the column choice, and the columns kept."""

    train: np.ndarray | sp.csr_matrix
    held_out: np.ndarray | sp.csr_matrix
    columns: np.ndarray | None  # kept WL columns (indices into the full matrix)


def _log1p(values: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
    return values.log1p() if sp.issparse(values) else np.log1p(values)


def prepare(
    features: np.ndarray | sp.csr_matrix,
    train_index: np.ndarray,
    held_out_index: np.ndarray,
    classifier: str,
    params: dict[str, Any],
) -> Prepared:
    """Select the rows of a block and, for sparse WL input, the columns (training part only)."""
    train, held_out = features[train_index], features[held_out_index]
    if not sp.issparse(features):
        return Prepared(np.asarray(train, float), np.asarray(held_out, float), None)
    frequency = train.getnnz(axis=0)
    columns = np.flatnonzero(frequency >= max(1, int(params.get("min_circuitos", 1))))
    if classifier == "hist_gradient_boosting":
        keep = int(params.get("columnas_wl", len(columns)))
        order = np.lexsort((columns, -frequency[columns]))  # most frequent first, stable
        columns = np.sort(columns[order[:keep]])
        return Prepared(train[:, columns].toarray(), held_out[:, columns].toarray(), columns)
    return Prepared(train[:, columns].tocsr(), held_out[:, columns].tocsr(), columns)


def build(classifier: str, params: dict[str, Any], seed: int, sparse: bool) -> Any:
    """Unfitted estimator for a configuration (preparation parameters left out)."""
    model_params = {k: v for k, v in params.items() if k not in (*RUNG_PARAMS, *PREPARATION_PARAMS)}
    if classifier == "regresion_logistica":
        steps = []
        if params.get("transformacion") == "log1p":
            steps.append(("log1p", FunctionTransformer(_log1p, accept_sparse=True)))
        steps.append(("escala", StandardScaler(with_mean=not sparse)))
        steps.append(("modelo", LogisticRegression(random_state=seed, **model_params)))
        return Pipeline(steps)
    if classifier == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **model_params)
    if classifier == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=seed, early_stopping=False, **model_params
        )
    raise ValueError(f"unknown classifier '{classifier}'")


def fit(model: Any, x: Any, y: np.ndarray, weights: np.ndarray | None) -> Any:
    """Fit with per-row weights (logistic regression convergence warnings are logged)."""
    target = model.steps[-1][0] if isinstance(model, Pipeline) else None
    keyword = f"{target}__sample_weight" if target else "sample_weight"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, **({keyword: weights} if weights is not None else {}))
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        logger.debug("logistic regression did not fully converge")
    return model


def full_probabilities(model: Any, x: Any, n_classes: int) -> np.ndarray:
    """Probabilities over every class (classes absent from training get 0)."""
    partial = model.predict_proba(x)
    probabilities = np.zeros((partial.shape[0], n_classes))
    probabilities[:, np.asarray(model.classes_, dtype=int)] = partial
    return probabilities
