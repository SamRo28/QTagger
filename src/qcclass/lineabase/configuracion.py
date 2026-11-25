"""Configuration of the baseline (``configs/experimentos/linea_base.yaml``) and search spaces.

A search space maps hyperparameter names to specifications: ``{tipo: float, min, max, log}``,
``{tipo: int, min, max, paso}``, ``{tipo: categorica, valores}`` or ``{tipo: fijo, valor}``.
:func:`suggest` draws a configuration from a space with an Optuna trial.

``revisiones`` repeats single studies with a changed space (the phase 10 range review): the
key is ``<rung>.<classifier>.<target>``, ``revision`` (>= 2) is appended to the study name
(``.r2``) and ``espacio`` replaces or adds hyperparameters on top of the study space. The
original study is kept; the summary uses the highest revision of every study.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from qcclass.config import ConfigError, load_experiment_config
from qcclass.evaluacion.predicciones import HEADS
from qcclass.lineabase.caracteristicas import WLSettings

CLASSIFIERS = ("regresion_logistica", "random_forest", "hist_gradient_boosting")
SOURCES = ("f3", "wl")
SPEC_TYPES = ("float", "int", "categorica", "fijo")
REQUIRED = (
    "nombre",
    "experimento_mlflow",
    "intentos",
    "semilla",
    "poda",
    "objetivos",
    "peldanos",
    "clasificadores",
    "importancia",
)


@dataclass(frozen=True)
class Rung:
    """A feature rung: F3 columns (``None`` = all) or WL with its settings and space."""

    name: str
    source: str
    columns: tuple[str, ...] | None
    wl: WLSettings | None
    space: dict[str, dict]


@dataclass(frozen=True)
class BaselineConfig:
    """Parsed ``linea_base.yaml``."""

    name: str
    mlflow_experiment: str
    trials: int
    seed: int
    pruning_startup: int
    targets: tuple[str, ...]
    rungs: dict[str, Rung]
    spaces: dict[str, dict[str, dict]]
    rung_overrides: dict[str, dict[str, dict[str, dict]]]
    revisions: dict[str, tuple[int, dict[str, dict]]]
    importance_repeats: int
    importance_wl_columns: int
    raw: dict[str, Any]

    def revision(self, rung: str, classifier: str, target: str) -> int:
        """Revision of a study (1 unless listed in ``revisiones``)."""
        return self.revisions.get(f"{rung}.{classifier}.{target}", (1, {}))[0]

    def space(self, rung: str, classifier: str, target: str | None = None) -> dict[str, dict]:
        """Search space of a study: rung parameters, classifier space, rung overrides and the
        changes of the study revision (when ``target`` is given)."""
        revised = self.revisions.get(f"{rung}.{classifier}.{target}", (1, {}))[1]
        return {
            **self.rungs[rung].space,
            **self.spaces[classifier],
            **self.rung_overrides.get(rung, {}).get(classifier, {}),
            **revised,
        }

    def study_name(self, rung: str, classifier: str, target: str) -> str:
        """``<nombre>.<rung>.<classifier>.<target>``, plus ``.r<n>`` from revision 2."""
        revision = self.revision(rung, classifier, target)
        suffix = f".r{revision}" if revision > 1 else ""
        return f"{self.name}.{rung}.{classifier}.{target}{suffix}"


def _check_space(space: dict, ctx: str) -> dict[str, dict]:
    for name, spec in space.items():
        if not isinstance(spec, dict) or spec.get("tipo") not in SPEC_TYPES:
            raise ConfigError(f"{ctx}[{name}]: tipo must be one of {SPEC_TYPES}")
        required = {
            "float": ("min", "max"),
            "int": ("min", "max"),
            "categorica": ("valores",),
            "fijo": ("valor",),
        }[spec["tipo"]]
        if missing := [key for key in required if key not in spec]:
            raise ConfigError(f"{ctx}[{name}]: missing {missing}")
    return dict(space)


def _rung(name: str, data: dict) -> Rung:
    ctx = f"peldanos[{name}]"
    source = data.get("fuente")
    if source not in SOURCES:
        raise ConfigError(f"{ctx}[fuente]: must be one of {SOURCES}")
    space = _check_space(data.get("espacio") or {}, f"{ctx}[espacio]")
    if source == "f3":
        columns = data.get("columnas")
        return Rung(name, source, tuple(columns) if columns else None, None, space)
    try:
        settings = WLSettings(
            h_max=int(data["h_max"]),
            bits_hash=int(data["bits_hash"]),
            divisiones_pi=int(data["divisiones_pi"]),
            tolerancia_angulo=float(data["tolerancia_angulo"]),
        )
    except KeyError as exc:
        raise ConfigError(f"{ctx}: missing {exc}") from exc
    return Rung(name, source, None, settings, space)


def load_baseline_config(path: Path | str) -> BaselineConfig:
    """Load and validate ``linea_base.yaml``.

    Raises:
        ConfigError: if a key, a target, a classifier or a space is invalid.
    """
    data = load_experiment_config(path, REQUIRED)
    targets = tuple(data["objetivos"])
    if bad := [t for t in targets if t not in HEADS]:
        raise ConfigError(f"objetivos: unknown {bad} (valid: {HEADS})")
    if bad := [c for c in data["clasificadores"] if c not in CLASSIFIERS]:
        raise ConfigError(f"clasificadores: unknown {bad} (valid: {CLASSIFIERS})")
    spaces = {
        name: _check_space(space or {}, f"clasificadores[{name}]")
        for name, space in data["clasificadores"].items()
    }
    rungs = {name: _rung(name, spec) for name, spec in data["peldanos"].items()}
    overrides = data.get("ajustes_por_peldano") or {}
    for rung, by_classifier in overrides.items():
        if rung not in rungs:
            raise ConfigError(f"ajustes_por_peldano: unknown rung {rung!r}")
        for classifier, space in by_classifier.items():
            _check_space(space, f"ajustes_por_peldano[{rung}][{classifier}]")
    revisions = {}
    for key, entry in (data.get("revisiones") or {}).items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] not in rungs or parts[1] not in spaces:
            raise ConfigError(f"revisiones[{key}]: expected <rung>.<classifier>.<target>")
        if parts[2] not in targets:
            raise ConfigError(f"revisiones[{key}]: unknown target {parts[2]!r}")
        revision = int(entry.get("revision", 0))
        if revision < 2:
            raise ConfigError(f"revisiones[{key}][revision]: must be >= 2")
        revisions[key] = (revision, _check_space(entry.get("espacio") or {}, f"revisiones[{key}]"))
    return BaselineConfig(
        name=data["nombre"],
        mlflow_experiment=data["experimento_mlflow"],
        trials=int(data["intentos"]),
        seed=int(data["semilla"]),
        pruning_startup=int(data["poda"]["intentos_iniciales"]),
        targets=targets,
        rungs=rungs,
        spaces=spaces,
        rung_overrides=overrides,
        revisions=revisions,
        importance_repeats=int(data["importancia"]["repeticiones"]),
        importance_wl_columns=int(data["importancia"]["max_columnas_wl"]),
        raw=data,
    )


def suggest(trial: optuna.Trial, space: dict[str, dict]) -> dict[str, Any]:
    """Draw a configuration from ``space`` (fixed values are returned as they are)."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["tipo"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["min"]), float(spec["max"]), log=bool(spec.get("log", False))
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["min"]), int(spec["max"]), step=int(spec.get("paso", 1))
            )
        elif kind == "categorica":
            params[name] = trial.suggest_categorical(name, list(spec["valores"]))
        else:
            params[name] = spec["valor"]
    return params


def fixed_params(space: dict[str, dict], chosen: dict[str, Any]) -> dict[str, Any]:
    """Complete the parameters of a finished trial with the fixed values of the space."""
    return {**{n: s["valor"] for n, s in space.items() if s["tipo"] == "fijo"}, **chosen}
