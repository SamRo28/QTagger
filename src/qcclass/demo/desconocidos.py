"""Provisional unknown detector: Mahalanobis distance on the graph summary vectors.

Fitted on the training part of one cross-validation block: one mean per trainable algorithm
class and a shared covariance (Ledoit-Wolf shrinkage, stable with 128 dimensions and few
circuits in some classes). The score of a circuit is its distance to the nearest class; the
threshold is the ``acceptance`` quantile of the scores of the known circuits of the held-out
block, so that share of them is accepted as known. Phase 16 sets the final threshold.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class Mahalanobis:
    """Class means, precision matrix and threshold."""

    classes: tuple[str, ...]
    means: np.ndarray
    precision: np.ndarray
    threshold: float
    acceptance: float

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        labels: np.ndarray,
        classes: tuple[str, ...],
        known_validation: np.ndarray,
        acceptance: float,
    ) -> "Mahalanobis":
        """Fit on ``vectors`` with class indices ``labels`` (rows with -1 are skipped) and set
        the threshold on ``known_validation``."""
        keep = labels >= 0
        vectors, labels = vectors[keep], labels[keep]
        present = sorted(set(labels.tolist()))
        means = np.stack([vectors[labels == c].mean(axis=0) for c in present])
        centred = vectors - means[np.searchsorted(present, labels)]
        precision = LedoitWolf().fit(centred).precision_
        detector = cls(tuple(classes[c] for c in present), means, precision, 0.0, acceptance)
        threshold = float(np.quantile(detector.scores(known_validation), acceptance))
        return cls(detector.classes, means, precision, threshold, acceptance)

    def distances(self, vectors: np.ndarray) -> np.ndarray:
        """Distance of every row to every class, ``[n, classes]``."""
        difference = vectors[:, None, :] - self.means[None, :, :]
        squared = np.einsum("ncd,de,nce->nc", difference, self.precision, difference)
        return np.sqrt(np.maximum(squared, 0))

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        """Distance to the nearest class."""
        return self.distances(vectors).min(axis=1)
