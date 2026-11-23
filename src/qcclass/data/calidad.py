"""Dataset quality control and report generation.

For now (Phase 5) this module builds the deduplication report: unique canonical circuits
per class and level, unique base circuits per class, conflicts between labels, and a
*proposal* (never applied automatically) of classes to merge or to keep only with their
family label.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig, SplitsConfig

DEDUP_REPORT_NAME = "informe_deduplicacion.md"
HIGH_OVERLAP = 0.2  # share of a class's circuits in exact conflicts that suggests a merge
LEVELS = ("alg", "indep", "nativegates", "mapped")


@dataclass(frozen=True)
class ClassProposal:
    """Proposed treatment of one class."""

    algorithm: str
    family: str
    bases: int
    exact_overlap: float
    partner: str
    proposal: str


def unique_canonical(metadata: pd.DataFrame) -> pd.DataFrame:
    """Canonical circuits that are not duplicates of another one."""
    return metadata[(metadata["ruta_canonical"] != "") & (metadata["duplicado_de"] == "")]


def unique_per_class_and_level(unique: pd.DataFrame) -> pd.DataFrame:
    """Table of unique canonical circuits per algorithm (rows) and level (columns)."""
    table = pd.crosstab(unique["algoritmo"], unique["nivel"])
    table = table.reindex(columns=[level for level in LEVELS if level in table.columns])
    table["total"] = table.sum(axis=1)
    return table


def unique_bases(unique: pd.DataFrame) -> pd.Series:
    """Base circuits (``group_id``) with at least one unique canonical circuit, per algorithm."""
    return unique.groupby("algoritmo")["group_id"].nunique()


def _partners(conflicts: pd.DataFrame, kind: str) -> dict[str, pd.Series]:
    """For each algorithm, how many conflict groups of ``kind`` it shares with each other one."""
    shared: dict[str, dict[str, int]] = {}
    for _, group in conflicts[conflicts["tipo"] == kind].groupby("huella"):
        algorithms = sorted(set(group["algoritmo"]))
        for a in algorithms:
            for b in algorithms:
                if a != b:
                    shared.setdefault(a, {}).setdefault(b, 0)
                    shared[a][b] += 1
    return {a: pd.Series(counts).sort_values(ascending=False) for a, counts in shared.items()}


def _best_partner(algorithm: str, family: str, config: DatasetConfig, partners: dict) -> str:
    """Class of the same family sharing most conflict groups with ``algorithm``, if any."""
    candidates = partners.get(algorithm, pd.Series(dtype=int))
    same_family = [name for name in candidates.index if name in config.familias.get(family, ())]
    return same_family[0] if same_family else ""


def propose_classes(
    config: DatasetConfig, splits: SplitsConfig, unique: pd.DataFrame, conflicts: pd.DataFrame
) -> list[ClassProposal]:
    """Propose keeping, merging or family-only labelling for every non-reserved class.

    A class needs about ``minimo_por_clase_train / aleatorio.train`` unique bases to reach
    the training minimum. Below that, it is proposed for a merge with the class of its family
    it shares most conflict groups with, or else for a family-only label. Classes with many
    circuits identical to another class (exact conflicts) are proposed for review.
    """
    needed = math.ceil(splits.minimo_por_clase_train / splits.aleatorio.train)
    bases = unique_bases(unique)
    exact_ids = set(conflicts.loc[conflicts["tipo"] == "exacto", "id"])
    partners = {**_partners(conflicts, "estructural"), **_partners(conflicts, "exacto")}
    proposals = []
    for algorithm in config.all_algorithms:
        if algorithm in splits.algoritmos_reservados:
            continue
        family = config.family_of(algorithm)
        ids = unique.loc[unique["algoritmo"] == algorithm, "id"]
        overlap = ids.isin(exact_ids).mean() if len(ids) else 0.0
        partner = _best_partner(algorithm, family, config, partners)
        count = int(bases.get(algorithm, 0))
        if count < needed:
            proposal = f"fusionar con {partner}" if partner else "solo etiqueta de familia"
        elif overlap >= HIGH_OVERLAP:
            proposal = f"revisar: {overlap:.0%} idénticos a otra clase" + (
                f" (posible fusión con {partner})" if partner else ""
            )
        else:
            proposal = "mantener"
        proposals.append(ClassProposal(algorithm, family, count, overlap, partner, proposal))
    return proposals


def _markdown(table: pd.DataFrame) -> str:
    """Minimal Markdown rendering of a DataFrame (no optional dependencies)."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule, *rows])


def write_deduplication_report(
    config: DatasetConfig, splits: SplitsConfig, metadata: pd.DataFrame, conflicts: pd.DataFrame
) -> Path:
    """Write ``results/informe_deduplicacion.md`` and return its path."""
    canonical = metadata[metadata["ruta_canonical"] != ""]
    unique = unique_canonical(metadata)
    per_level = unique_per_class_and_level(unique).reset_index()
    proposals = pd.DataFrame(
        [
            {
                "clase": p.algorithm,
                "familia": p.family,
                "bases únicas": p.bases,
                "idénticos a otra clase": f"{p.exact_overlap:.0%}",
                "propuesta": p.proposal,
            }
            for p in propose_classes(config, splits, unique, conflicts)
        ]
    )
    needed = math.ceil(splits.minimo_por_clase_train / splits.aleatorio.train)
    by_kind = conflicts.groupby("tipo")["huella"].nunique() if len(conflicts) else pd.Series()
    text = "\n\n".join(
        [
            "# Informe de deduplicación (Fase 5)",
            f"Circuitos canónicos: {len(canonical)}; duplicados exactos marcados: "
            f"{int((canonical['duplicado_de'] != '').sum())}; únicos: {len(unique)}.",
            f"Grupos conflictivos (etiquetas distintas): exactos {int(by_kind.get('exacto', 0))}, "
            f"estructurales {int(by_kind.get('estructural', 0))} "
            "(ver `data/logs/duplicados_conflictivos.csv`).",
            "## Circuitos únicos por clase y nivel",
            _markdown(per_level),
            "## Propuesta de clases (no aplicada)",
            f"Una clase necesita unas {needed} bases únicas para llegar a "
            f"`minimo_por_clase_train = {splits.minimo_por_clase_train}` con "
            f"`train = {splits.aleatorio.train}`. Los algoritmos reservados no se evalúan. "
            "La pareja propuesta es la clase de la misma familia con la que comparte más "
            "grupos de circuitos idénticos o estructuralmente iguales.",
            _markdown(proposals),
        ]
    )
    path = config.rutas.resultados / DEDUP_REPORT_NAME
    write_text_atomic(text + "\n", path)
    return path
