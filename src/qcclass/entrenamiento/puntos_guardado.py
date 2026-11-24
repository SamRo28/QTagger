"""Checkpoints at every epoch and resumption after an interruption.

A checkpoint is ``<directory>/epoca_<NNNN>.pt``: a dictionary with the epoch, the states the
caller passes (model, optimiser, scheduler, gradient scaler, sampler epoch...) and the state
of every random generator. Files are written atomically (temporary file + rename), so a cut
during a save never leaves a corrupt latest checkpoint; only the last ``keep`` are kept.
Loading uses ``weights_only=True``.
"""

import logging
import re
from pathlib import Path
from typing import Any

import torch

from qcclass.entrenamiento.reproducibilidad import restore_rng_state, rng_state

logger = logging.getLogger(__name__)

PATTERN = re.compile(r"^epoca_(\d{4,})\.pt$")


class CheckpointManager:
    """Save, list and load the checkpoints of one training run."""

    def __init__(self, directory: Path, *, keep: int = 2) -> None:
        if keep < 1:
            raise ValueError("keep at least one checkpoint")
        self.directory = Path(directory)
        self.keep = keep

    def path(self, epoch: int) -> Path:
        """``epoca_<NNNN>.pt``."""
        return self.directory / f"epoca_{epoch:04d}.pt"

    def epochs(self) -> list[int]:
        """Saved epochs, in increasing order."""
        if not self.directory.is_dir():
            return []
        found = (PATTERN.match(p.name) for p in self.directory.iterdir())
        return sorted(int(m.group(1)) for m in found if m)

    def save(self, epoch: int, states: dict[str, Any]) -> Path:
        """Save the states after ``epoch`` (plus the generator states) and prune old ones."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(epoch)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save({"epoca": epoch, "estados": states, "rng": rng_state()}, temporary)
        temporary.replace(path)
        for old in self.epochs()[: -self.keep]:
            self.path(old).unlink(missing_ok=True)
        return path

    def load_latest(
        self, map_location: str | torch.device = "cpu", *, restore_rng: bool = True
    ) -> tuple[int, dict[str, Any]] | None:
        """(epoch, states) of the latest checkpoint, or ``None``; restores the generators."""
        epochs = self.epochs()
        if not epochs:
            return None
        content = torch.load(self.path(epochs[-1]), map_location=map_location, weights_only=True)
        if restore_rng:
            restore_rng_state(content["rng"])
        logger.info("resuming from epoch %d (%s)", content["epoca"], self.directory)
        return content["epoca"], content["estados"]
