"""Time model of the GNN on one machine: predicted hours of a search trial.

Fitted on the ``estimaciones`` of ``<resultados>/medicion/calibracion_gnn_<machine>.json``
(``qcclass calibrar-gnn``): ``log(s per training epoch) = a + b log(dim) + c log(capas) +
d [gatv2]`` by least squares; the evaluation of the held-out block costs the mean ratio
``s per evaluation / s per epoch`` of the calibration. A trial of ``epocas`` epochs on
``bloques`` blocks with one evaluation every ``evaluar_cada`` epochs takes
``bloques * (epocas + ceil(epocas / evaluar_cada) * ratio) * s_epoch * sobrecoste`` seconds;
``sobrecoste`` covers what the calibration leaves out (data loading, checkpoints).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPORT_FILE = "calibracion_gnn_{machine}.json"


def _features(model: dict[str, Any]) -> list[float]:
    return [
        1.0,
        math.log(float(model.get("dim", 64))),
        math.log(float(model.get("capas", 3))),
        float(model.get("tipo_capa", "gine") == "gatv2"),
    ]


@dataclass(frozen=True)
class TimeModel:
    """Fitted time model (see the module docstring)."""

    coefficients: tuple[float, ...]
    evaluation_ratio: float

    @classmethod
    def fit(cls, estimates: list[dict[str, Any]]) -> "TimeModel":
        """Fit on calibration estimates (entries without an error)."""
        rows = [e for e in estimates if "segundos_entrenamiento_epoca" in e]
        if len(rows) < 4:
            raise ValueError("the time model needs at least 4 calibrated configurations")
        x = np.array([_features(e["modelo"]) for e in rows])
        y = np.log([e["segundos_entrenamiento_epoca"] for e in rows])
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        ratio = float(
            np.mean([e["segundos_evaluacion"] / e["segundos_entrenamiento_epoca"] for e in rows])
        )
        return cls(tuple(float(c) for c in coefficients), ratio)

    @classmethod
    def load(cls, results: Path, machine: str) -> "TimeModel | None":
        """Model of ``machine`` from its calibration report, or ``None`` if there is none."""
        path = Path(results) / "medicion" / REPORT_FILE.format(machine=machine)
        if not path.is_file():
            return None
        return cls.fit(json.loads(path.read_text(encoding="utf-8"))["estimaciones"])

    def epoch_seconds(self, model: dict[str, Any]) -> float:
        """Predicted training seconds per epoch."""
        return float(math.exp(np.dot(self.coefficients, _features(model))))

    def trial_hours(
        self,
        model: dict[str, Any],
        *,
        epochs: int,
        blocks: int,
        evaluate_every: int,
        overhead: float,
    ) -> float:
        """Predicted hours of a trial (see the module docstring)."""
        evaluations = math.ceil(epochs / evaluate_every)
        seconds = blocks * (epochs + evaluations * self.evaluation_ratio)
        return seconds * self.epoch_seconds(model) * overhead / 3600
