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


# --------------------------------------------------------------------------------------------
# Dataset quality report (Phase 8)
# --------------------------------------------------------------------------------------------

DATASET_REPORT_NAME = "informe_dataset.md"
FIGURE_NAME = "distribucion_cubits_profundidad.png"
MAX_EXAMPLES = 5


@dataclass(frozen=True)
class CoherenceCheck:
    """Result of one consistency check between the metadata and the files on disk."""

    name: str
    problems: list[str]

    @property
    def ok(self) -> bool:
        """True when nothing is wrong."""
        return not self.problems


def _relative_files(config: DatasetConfig, folder: Path, pattern: str) -> set[str]:
    """Files under ``folder`` matching ``pattern``, as paths relative to ``rutas.raiz``."""
    if not folder.is_dir():
        return set()
    return {path.relative_to(config.rutas.raiz).as_posix() for path in folder.rglob(pattern)}


def _missing(config: DatasetConfig, paths: pd.Series) -> list[str]:
    """Paths (relative to ``raiz``) whose QASM file or QPY twin does not exist."""
    missing = []
    for relative in paths:
        qasm_file = config.rutas.raiz / relative
        if not qasm_file.is_file() or not qasm_file.with_suffix(".qpy").is_file():
            missing.append(relative)
    return missing


def _ids_check(name: str, expected: set[str], found: set[str]) -> CoherenceCheck:
    """Compare two sets of identifiers."""
    problems = [f"falta {i}" for i in sorted(expected - found)]
    problems += [f"sobra {i}" for i in sorted(found - expected)]
    return CoherenceCheck(name, problems)


def coherence_checks(config: DatasetConfig, metadata: pd.DataFrame) -> list[CoherenceCheck]:
    """Check that the metadata and the files on disk describe the same dataset."""
    from qcclass.formatos.construir import DAG_DIR, FEATURES_FILE, QUBIT_DIR, TOKENS_FILE
    from qcclass.formatos.grafo_dag import INDEX_FILE

    raw = metadata[metadata["variante_comp"] == ""]
    canonical = metadata[metadata["ruta_canonical"] != ""]
    with_formats = canonical[canonical["exclusion"].isin(["", LABELLED_VARIANT])]
    expected_ids = set(with_formats["id"])
    raw_files = _relative_files(config, config.rutas.raw, "*.qasm")
    canonical_files = _relative_files(config, config.rutas.canonical, "*.qasm")
    checks = [
        CoherenceCheck("raw: filas sin QASM o QPY", _missing(config, raw["ruta_raw"])),
        CoherenceCheck(
            "canonical: filas sin QASM o QPY", _missing(config, canonical["ruta_canonical"])
        ),
        CoherenceCheck("raw: archivos sin fila", sorted(raw_files - set(raw["ruta_raw"]))),
        CoherenceCheck(
            "canonical: archivos sin fila",
            sorted(canonical_files - set(canonical["ruta_canonical"])),
        ),
    ]
    for directory in (DAG_DIR, QUBIT_DIR):
        folder = config.rutas.graphs / directory
        index = folder / INDEX_FILE
        listed = set(pd.read_csv(index, dtype=str)["id"]) if index.is_file() else set()
        on_disk = {path.stem for path in folder.glob("*.pt")}
        checks.append(_ids_check(f"{directory}: índice frente a metadatos", expected_ids, listed))
        checks.append(_ids_check(f"{directory}: archivos .pt frente a índice", listed, on_disk))
    features_file = config.rutas.features / FEATURES_FILE
    features = (
        set(pd.read_parquet(features_file, columns=["id"])["id"])
        if features_file.is_file()
        else set()
    )
    checks.append(_ids_check("F3: filas frente a metadatos", expected_ids, features))
    tokens_file = config.rutas.tokens / TOKENS_FILE
    tokens = set()
    if tokens_file.is_file():
        tokens = (
            set(pd.read_json(tokens_file, lines=True)["id"])
            if tokens_file.stat().st_size
            else set()
        )
    checks.append(_ids_check("F6: secuencias frente a metadatos", expected_ids, tokens))
    labelled = set(canonical.loc[canonical["variante_comp"] == LABELLED_VARIANT, "id"])
    gate_labels = {path.stem for path in config.rutas.gate_labels.glob("*.json")}
    checks.append(_ids_check("F7: etiquetas frente a variantes etiquetadas", labelled, gate_labels))
    groups = metadata.set_index("id")["group_id"]
    for split_file in sorted(config.rutas.splits.glob("*.csv")):
        table = pd.read_csv(split_file, dtype=str)
        checks.append(
            CoherenceCheck(
                f"particiones/{split_file.name}: ids sin formatos",
                sorted(set(table["id"]) - expected_ids),
            )
        )
        sets = table.assign(group_id=table["id"].map(groups)).groupby("group_id")["conjunto"]
        leaked = sets.nunique()
        checks.append(
            CoherenceCheck(
                f"particiones/{split_file.name}: grupos en más de un conjunto",
                sorted(leaked[leaked > 1].index),
            )
        )
    return checks


def counts_before_after(canonical: pd.DataFrame, column: str) -> pd.DataFrame:
    """Canonical circuits before deduplication and usable ones after, grouped by ``column``."""
    canonical = canonical.assign(etiqueta=labels(canonical))
    before = canonical.groupby(column).size().rename("antes")
    after = canonical[canonical["exclusion"] == ""].groupby(column).size().rename("después")
    table = pd.concat([before, after], axis=1).fillna(0).astype(int)
    table["% conservado"] = (100 * table["después"] / table["antes"]).round(1)
    return table.rename_axis(column).reset_index()


def size_depth_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Qubits and depth per label of the usable circuits (min / median / max)."""
    numeric = rows.assign(
        n_qubits=rows["n_qubits"].astype(int), profundidad=rows["profundidad"].astype(int)
    )
    stats = numeric.groupby("etiqueta").agg(
        circuitos=("id", "count"),
        cubits_min=("n_qubits", "min"),
        cubits_mediana=("n_qubits", "median"),
        cubits_max=("n_qubits", "max"),
        profundidad_min=("profundidad", "min"),
        profundidad_mediana=("profundidad", "median"),
        profundidad_max=("profundidad", "max"),
    )
    return stats.reset_index()


def size_depth_figure(rows: pd.DataFrame, path: Path) -> Path:
    """Box plots of qubits and depth per label (usable circuits)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_order = sorted(rows["etiqueta"].unique())
    qubits = [rows.loc[rows["etiqueta"] == lab, "n_qubits"].astype(int) for lab in labels_order]
    depth = [rows.loc[rows["etiqueta"] == lab, "profundidad"].astype(int) for lab in labels_order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.32 * len(labels_order) + 1.5), sharey=True)
    axes[0].boxplot(qubits, orientation="horizontal", tick_labels=labels_order)
    axes[0].set_xlabel("cúbits")
    axes[1].boxplot(depth, orientation="horizontal", tick_labels=labels_order)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("profundidad (escala log)")
    fig.suptitle("Circuitos útiles por etiqueta")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _log_table(path: Path, rows: str, columns: str) -> pd.DataFrame | None:
    """Crosstab of a CSV log, or None if the log does not exist or is empty."""
    if not path.is_file():
        return None
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not len(log):
        return pd.DataFrame()
    return pd.crosstab(log[rows], log[columns]).reset_index()


@dataclass(frozen=True)
class DatasetReport:
    """Outcome of the quality step."""

    path: Path
    figure: Path
    checks: list[CoherenceCheck]
    below_minimum: dict[str, list[str]]

    @property
    def coherent(self) -> bool:
        """True if every coherence check passed."""
        return all(check.ok for check in self.checks)


def _below_minimum(
    config: DatasetConfig, splits: SplitsConfig, metadata: pd.DataFrame
) -> dict[str, list[str]]:
    """Labels below ``minimo_por_clase_train`` in each existing split protocol."""
    from qcclass.data.particiones import PROTOCOLS, below_minimum, candidate_rows, split_path

    rows = candidate_rows(config, splits, metadata)
    result = {}
    for protocol in PROTOCOLS:
        path = split_path(config, protocol)
        if path.is_file() and protocol != "oraculos_no_vistos":
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            result[protocol] = below_minimum(table, rows, splits.minimo_por_clase_train)
    return result


def write_dataset_report(config: DatasetConfig, splits: SplitsConfig) -> DatasetReport:
    """Write ``results/informe_dataset.md`` and its figure."""
    from qcclass.data.deduplicar import conflicts_path
    from qcclass.data.generar_mqt import errors_path as generation_log
    from qcclass.data.normalizar import log_path as normalization_log
    from qcclass.data.oraculos import errors_path as oracle_log
    from qcclass.metadatos import METADATA_COLUMNS, read_table

    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    canonical = metadata[
        (metadata["ruta_canonical"] != "") & (metadata["variante_comp"] != LABELLED_VARIANT)
    ]
    rows = usable(canonical)
    figure = size_depth_figure(rows, config.rutas.resultados / "figuras" / FIGURE_NAME)
    checks = coherence_checks(config, metadata)
    low = _below_minimum(config, splits, metadata)
    sections = [
        "# Informe de calidad del dataset (Fase 8)",
        f"Circuitos raw: {int((metadata['variante_comp'] == '').sum())}; canónicos: "
        f"{len(canonical)}; útiles tras deduplicar: {len(rows)}; variantes etiquetadas (F7): "
        f"{int((metadata['variante_comp'] == LABELLED_VARIANT).sum())}. Detalle de la "
        f"deduplicación en `{DEDUP_REPORT_NAME}` y de las particiones en "
        "`informe_particiones.md`.",
        "## Circuitos por etiqueta, antes y después de deduplicar",
        _markdown(counts_before_after(canonical, "etiqueta")),
        "## Por familia",
        _markdown(counts_before_after(canonical, "familia")),
        "## Por nivel",
        _markdown(counts_before_after(canonical, "nivel")),
        "## Cúbits y profundidad por etiqueta (circuitos útiles)",
        f"![distribución](figuras/{FIGURE_NAME})",
        _markdown(size_depth_table(rows)),
        "## Etiquetas por debajo del mínimo de train",
        "\n".join(f"- `{p}`: {', '.join(labs) if labs else 'ninguna'}" for p, labs in low.items())
        or "Sin particiones creadas.",
    ]
    logs = [
        ("Errores de generación (MQT Bench)", generation_log(config), "algoritmo", "tipo"),
        ("Errores de los oráculos propios", oracle_log(config), "algoritmo", "tipo"),
        ("Descartes de la normalización", normalization_log(config), "algoritmo", "tipo"),
    ]
    sections.append("## Errores y descartes")
    for title, path, row_key, column_key in logs:
        table = _log_table(path, row_key, column_key)
        sections.append(f"### {title}")
        if table is None:
            sections.append("Sin registro.")
        else:
            sections.append(_markdown(table) if len(table) else "Ninguno.")
    sections.append("### Duplicados conflictivos (excluidos de todas las clases implicadas)")
    conflicts = conflicts_path(config)
    if conflicts.is_file():
        table = pd.read_csv(conflicts, dtype=str, keep_default_na=False)
        sections.append(_markdown(_conflict_summary(table)) if len(table) else "Ninguno.")
    else:
        sections.append("Sin registro.")
    sections += ["## Coherencia entre `metadata.csv` y los archivos"]
    coherence = pd.DataFrame(
        [
            {
                "comprobación": check.name,
                "estado": "ok" if check.ok else f"{len(check.problems)} problemas",
                "ejemplos": ", ".join(check.problems[:MAX_EXAMPLES]) or "—",
            }
            for check in checks
        ]
    )
    sections.append(_markdown(coherence))
    path = config.rutas.resultados / DATASET_REPORT_NAME
    write_text_atomic("\n\n".join(sections) + "\n", path)
    return DatasetReport(path, figure, checks, low)
