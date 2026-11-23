"""Dataset quality control and report generation.

For now (Phase 5) this module builds the deduplication report: unique canonical circuits
per training label and level, the label conflicts excluded from the dataset, the decision
on classes that need a minimum of base circuits, and a *proposal* (never applied
automatically) of classes to merge or to keep only with their family label.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig, SplitsConfig
from qcclass.data.deduplicar import (
    FAMILY_LABEL_PREFIX,
    LABELLED_VARIANT,
    family_label,
    merged_labels,
)

DEDUP_REPORT_NAME = "informe_deduplicacion.md"
MERGE_SHARE = 0.5  # propose a merge above this share of identical bases (smaller class)
LEVELS = ("alg", "indep", "nativegates", "mapped")


@dataclass(frozen=True)
class ClassProposal:
    """Proposed treatment of one training label."""

    label: str
    family: str
    bases: int
    partner: str
    identical_share: float
    proposal: str


def labels(canonical: pd.DataFrame) -> pd.Series:
    """Training label of each canonical row: ``clase`` or ``familia:<family>``."""
    return canonical["clase"].where(
        canonical["clase"] != "", canonical["familia"].map(family_label)
    )


def usable(canonical: pd.DataFrame) -> pd.DataFrame:
    """Canonical circuits that are neither duplicates nor label conflicts."""
    rows = canonical[(canonical["ruta_canonical"] != "") & (canonical["exclusion"] == "")]
    return rows.assign(etiqueta=labels(rows))


def usable_per_label_and_level(rows: pd.DataFrame) -> pd.DataFrame:
    """Table of usable canonical circuits per label (rows) and level (columns)."""
    table = pd.crosstab(rows["etiqueta"], rows["nivel"])
    table = table.reindex(columns=[level for level in LEVELS if level in table.columns])
    table["total"] = table.sum(axis=1)
    table["bases"] = rows.groupby("etiqueta")["group_id"].nunique()
    return table


def identical_base_shares(canonical: pd.DataFrame) -> pd.DataFrame:
    """For every ordered pair of labels (A, B): share of A's bases with a circuit identical
    (same SHA) to some circuit of B. Computed on every canonical circuit, excluded included.
    """
    rows = canonical[canonical["ruta_canonical"] != ""].assign(etiqueta=labels(canonical))
    labels_by_sha = rows.groupby("huella_sha")["etiqueta"].agg(set)
    shared = rows.assign(others=rows["huella_sha"].map(labels_by_sha))
    totals = rows.groupby("etiqueta")["group_id"].nunique()
    records = []
    for (label, group_id), group in shared.groupby(["etiqueta", "group_id"]):
        for other in set().union(*group["others"]) - {label}:
            records.append((label, other, group_id))
    if not records:
        return pd.DataFrame(
            columns=["a", "b", "bases_identicas", "bases_a", "bases_b", "proporcion"]
        )
    pairs = pd.DataFrame(records, columns=["a", "b", "group_id"])
    counts = pairs.groupby(["a", "b"])["group_id"].nunique().rename("bases_identicas")
    table = counts.reset_index()
    table["bases_a"] = table["a"].map(totals)
    table["bases_b"] = table["b"].map(totals)
    table["proporcion"] = table["bases_identicas"] / table["bases_a"]
    return table


def propose_classes(
    config: DatasetConfig, splits: SplitsConfig, canonical: pd.DataFrame
) -> list[ClassProposal]:
    """Propose keeping, merging or family-only labelling for every non-reserved class label.

    A merge is proposed only when more than half of the bases of the smaller class have a
    circuit identical to one of the other class. Otherwise, a class with fewer than about
    ``minimo_por_clase_train / aleatorio.train`` usable bases is proposed for a family-only
    label.
    """
    needed = math.ceil(splits.minimo_por_clase_train / splits.aleatorio.train)
    rows = usable(canonical)
    bases = rows.groupby("etiqueta")["group_id"].nunique()
    shares = identical_base_shares(canonical)
    families = rows.drop_duplicates("etiqueta").set_index("etiqueta")["familia"]
    reserved = set(merged_labels(config, pd.Series(list(splits.algoritmos_reservados))))
    class_labels = set(merged_labels(config, pd.Series(list(config.all_algorithms))))
    proposals = []
    for label in sorted(bases.index):
        if label in reserved or label not in class_labels:
            continue
        candidates = shares[(shares["a"] == label) & (shares["bases_a"] <= shares["bases_b"])]
        best = candidates.sort_values("proporcion", ascending=False).head(1)
        partner = best["b"].iloc[0] if len(best) else ""
        share = float(best["proporcion"].iloc[0]) if len(best) else 0.0
        count = int(bases[label])
        if share > MERGE_SHARE:
            proposal = f"fusionar con {partner}"
        elif count < needed:
            proposal = "solo etiqueta de familia"
        else:
            proposal = "mantener"
        proposals.append(
            ClassProposal(label, families.get(label, ""), count, partner, share, proposal)
        )
    return proposals


def _markdown(table: pd.DataFrame) -> str:
    """Minimal Markdown rendering of a DataFrame (no optional dependencies)."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule, *rows])


def _conflict_summary(conflicts: pd.DataFrame) -> pd.DataFrame:
    """Excluded exact conflicts: label combination, groups and circuits removed."""
    exact = conflicts[conflicts["tipo"] == "exacto"]
    if not len(exact):
        return pd.DataFrame(columns=["etiquetas", "grupos", "circuitos excluidos"])
    combos = exact.groupby("huella")["etiqueta"].agg(lambda s: " + ".join(sorted(set(s))))
    exact = exact.assign(etiquetas=exact["huella"].map(combos))
    summary = exact.groupby("etiquetas").agg(
        grupos=("huella", "nunique"), circuitos=("id", "count")
    )
    return summary.rename(columns={"circuitos": "circuitos excluidos"}).reset_index()


def _minimum_decisions(config: DatasetConfig, canonical: pd.DataFrame) -> pd.DataFrame:
    """Outcome of ``clase_propia_con_minimo`` for each listed algorithm."""
    minimum = config.etiquetas.minimo_bases_clase_propia
    rows = canonical[(canonical["ruta_canonical"] != "") & (canonical["exclusion"] == "")]
    bases = rows.groupby("algoritmo")["group_id"].nunique()
    return pd.DataFrame(
        [
            {
                "algoritmo": name,
                "bases únicas": int(bases.get(name, 0)),
                "mínimo": minimum,
                "decisión": "clase propia"
                if bases.get(name, 0) >= minimum
                else "solo etiqueta de familia",
            }
            for name in config.etiquetas.clase_propia_con_minimo
        ]
    )


def write_deduplication_report(
    config: DatasetConfig, splits: SplitsConfig, metadata: pd.DataFrame, conflicts: pd.DataFrame
) -> Path:
    """Write ``results/informe_deduplicacion.md`` and return its path."""
    canonical = metadata[
        (metadata["ruta_canonical"] != "") & (metadata["variante_comp"] != LABELLED_VARIANT)
    ]
    rows = usable(canonical)
    excluded = canonical["exclusion"].value_counts()
    needed = math.ceil(splits.minimo_por_clase_train / splits.aleatorio.train)
    proposals = pd.DataFrame(
        [
            {
                "etiqueta": p.label,
                "familia": p.family,
                "bases útiles": p.bases,
                "más parecida": p.partner or "—",
                "bases idénticas": f"{p.identical_share:.0%}",
                "propuesta": p.proposal,
            }
            for p in propose_classes(config, splits, canonical)
        ]
    )
    merges = ", ".join(f"`{a}` → `{b}`" for a, b in config.etiquetas.fusiones.items())
    text = "\n\n".join(
        [
            "# Informe de deduplicación (Fase 5)",
            f"Circuitos canónicos: {len(canonical)}; duplicados exactos: "
            f"{int(excluded.get('duplicado', 0))}; excluidos por conflicto de etiquetas: "
            f"{int(excluded.get('conflicto_etiquetas', 0))}; útiles: {len(rows)}.",
            f"Fusiones aplicadas: {merges or 'ninguna'}. Las etiquetas "
            f"`{FAMILY_LABEL_PREFIX}<familia>` son clases que solo conservan la familia.",
            "## Circuitos útiles por etiqueta y nivel",
            _markdown(usable_per_label_and_level(rows).reset_index()),
            "## Conflictos de etiquetas excluidos",
            "Circuitos canónicos idénticos (mismo SHA) con etiquetas distintas: se eliminan de "
            "todas las clases implicadas (lista completa en "
            "`data/logs/duplicados_conflictivos.csv`, `tipo = exacto`).",
            _markdown(_conflict_summary(conflicts)),
            "## Clases propias condicionadas a un mínimo",
            _markdown(_minimum_decisions(config, canonical)),
            "## Propuesta de clases (no aplicada)",
            f"Una clase necesita unas {needed} bases útiles para llegar a "
            f"`minimo_por_clase_train = {splits.minimo_por_clase_train}` con "
            f"`train = {splits.aleatorio.train}`. Solo se propone una fusión si más del "
            f"{MERGE_SHARE:.0%} de las bases de la clase más pequeña tienen un circuito idéntico "
            'en la otra ("bases idénticas", calculado antes de excluir conflictos). Los '
            "algoritmos reservados no se evalúan.",
            _markdown(proposals),
        ]
    )
    path = config.rutas.resultados / DEDUP_REPORT_NAME
    write_text_atomic(text + "\n", path)
    return path
