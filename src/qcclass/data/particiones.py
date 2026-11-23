"""Create the train/val/test split protocols.

One CSV per protocol in ``<splits>/`` with the columns ``id``, ``conjunto`` (``train``,
``val``, ``test`` or ``desconocido``) and ``solo_evaluacion``. Every protocol assigns whole
base circuits (``group_id``) to a set, so no group ever appears in two sets.

Rows considered: usable canonical circuits (``exclusion`` empty) and labelled F7 variants
(``exclusion = etiquetada``). A labelled variant follows its group: if the group is in
val/test it is added there with ``solo_evaluacion = True``; if it is in train it is left
out. Labels are the training labels (``clase`` or ``familia:<family>``). The negative
control and every reserved algorithm are "unknown": they only appear in ``abierto``, as
``desconocido``.

Protocols:

* ``aleatorio``: 70/15/15 of the groups of every label (stratified by label) with
  scikit-learn's ``GroupShuffleSplit`` in two stages.
* ``por_tamano``: by number of qubits, using the ranges of ``splits.yaml``.
* ``oraculos_no_vistos``: oracle family only; the split unit is ``oraculo_id`` (the group
  for MQT circuits), so no test oracle is seen in training.
* ``por_nivel``: groups split as in ``aleatorio``; train keeps the train levels of its
  groups (ALG + INDEP), val the train levels of its groups and test the test levels
  (NATIVEGATES + MAPPED) of other groups.
* ``abierto``: ``aleatorio`` plus the unknown circuits as ``desconocido``.

Balancing (``balanceo``): in ``muestreo`` mode, the train groups of each label are sampled
down to ``max_bases_por_clase`` (val/test are kept whole; nothing is deleted from disk); in
``ponderacion`` mode every group is kept and ``pesos_<protocolo>.json`` holds per-class loss
weights. Splits are created once and only rebuilt with ``force``.
"""

import json
import logging
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig, SplitsConfig
from qcclass.data.deduplicar import family_label
from qcclass.formatos.etiquetas_puerta import LABELLED_VARIANT
from qcclass.metadatos import METADATA_COLUMNS, read_table, write_table
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

PROTOCOLS = ("aleatorio", "por_tamano", "oraculos_no_vistos", "por_nivel", "abierto")
TRAIN, VAL, TEST, UNKNOWN = "train", "val", "test", "desconocido"
SPLIT_COLUMNS = ("id", "conjunto", "solo_evaluacion")
ORACLE_FAMILY = "oraculos"
REPORT_NAME = "informe_particiones.md"


# --------------------------------------------------------------------------------------------
# Candidate rows
# --------------------------------------------------------------------------------------------


def candidate_rows(
    config: DatasetConfig, splits: SplitsConfig, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Rows that can enter a split, with ``etiqueta``, ``desconocido`` and ``solo_evaluacion``."""
    canonical = metadata[metadata["ruta_canonical"] != ""]
    rows = canonical[canonical["exclusion"].isin(["", LABELLED_VARIANT])].copy()
    unknown = {*config.control_negativo, *config.reservados_sin_familia}
    unknown |= set(splits.algoritmos_reservados)
    rows["etiqueta"] = rows["clase"].where(rows["clase"] != "", rows["familia"].map(family_label))
    rows["desconocido"] = rows["algoritmo"].isin(sorted(unknown))
    rows["solo_evaluacion"] = rows["exclusion"] == LABELLED_VARIANT
    rows["n_qubits"] = rows["n_qubits"].astype(int)
    return rows


# --------------------------------------------------------------------------------------------
# Group assignment
# --------------------------------------------------------------------------------------------


def _three_way(units: Sequence[str], splits: SplitsConfig, seed: int) -> dict[str, str]:
    """Assign split units to train/val/test with ``GroupShuffleSplit`` in two stages.

    With fewer than three units, the first goes to train and the second to test.
    """
    units = sorted(set(units))
    if len(units) < 3:
        shuffled = units[:]
        random.Random(seed).shuffle(shuffled)
        return dict(zip(shuffled, [TRAIN, TEST], strict=False))
    fractions = splits.aleatorio
    held_out = fractions.val + fractions.test
    array = np.array(units)
    first = GroupShuffleSplit(n_splits=1, test_size=held_out, random_state=seed)
    train_idx, rest_idx = next(first.split(array, groups=array))
    assignment = dict.fromkeys(array[train_idx], TRAIN)
    rest = array[rest_idx]
    if len(rest) == 1 or fractions.val == 0:
        assignment.update(dict.fromkeys(rest, TEST))
        return assignment
    second = GroupShuffleSplit(n_splits=1, test_size=fractions.test / held_out, random_state=seed)
    val_idx, test_idx = next(second.split(rest, groups=rest))
    assignment.update(dict.fromkeys(rest[val_idx], VAL))
    assignment.update(dict.fromkeys(rest[test_idx], TEST))
    return assignment


def stratified_assignment(
    rows: pd.DataFrame, splits: SplitsConfig, protocol: str, unit: str = "group_id"
) -> pd.Series:
    """Set of every row, splitting the ``unit`` values of each label separately."""
    assignment: dict[str, str] = {}
    for label, group in rows.groupby("etiqueta", sort=True):
        seed = derive_seed(splits.semilla, protocol, label)
        assignment.update(_three_way(list(group[unit]), splits, seed))
    return rows[unit].map(assignment)


# --------------------------------------------------------------------------------------------
# Balancing
# --------------------------------------------------------------------------------------------


def resolve_cap(config: DatasetConfig, splits: SplitsConfig, rows: pd.DataFrame) -> int:
    """``max_bases_por_clase``, or the median of base circuits per label over the labels whose
    algorithms are not variational with parameters."""
    if splits.balanceo.max_bases_por_clase is not None:
        return splits.balanceo.max_bases_por_clase
    trainable = rows[~rows["desconocido"] & ~rows["solo_evaluacion"]]
    variational = set(config.variacionales_con_parametros)
    reference = trainable[~trainable["algoritmo"].isin(sorted(variational))]
    return round(statistics.median(reference.groupby("etiqueta")["group_id"].nunique()))


def cap_train_groups(
    rows: pd.DataFrame, sets: pd.Series, cap: int, splits: SplitsConfig, protocol: str
) -> pd.Series:
    """Drop (set to NaN) the train rows of groups outside a seeded sample of ``cap`` groups."""
    sets = sets.copy()
    in_train = sets == TRAIN
    for label, group in rows[in_train].groupby("etiqueta", sort=True):
        groups = sorted(set(group["group_id"]))
        if len(groups) <= cap:
            continue
        rng = random.Random(derive_seed(splits.semilla, protocol, "balanceo", label))
        kept = set(rng.sample(groups, cap))
        dropped = group.index[~group["group_id"].isin(sorted(kept))]
        sets[dropped] = np.nan
    return sets


def class_weights(rows: pd.DataFrame, sets: pd.Series) -> dict[str, dict[str, float]]:
    """Inverse-frequency loss weights (mean 1) from the train circuits of each head."""
    train = rows[sets == TRAIN]
    weights = {}
    for head, column in (("algoritmo", "clase"), ("familia", "familia")):
        counts = train.loc[train[column] != "", column].value_counts()
        total, classes = counts.sum(), len(counts)
        weights[head] = {name: total / (classes * count) for name, count in counts.items()}
    return weights


# --------------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------------


def _random(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    known = rows[~rows["desconocido"]]
    return stratified_assignment(known, splits, "aleatorio").reindex(rows.index)


def _by_size(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    sets = pd.Series(np.nan, index=rows.index, dtype=object)
    known = ~rows["desconocido"]
    for name in (TRAIN, VAL, TEST):
        size = getattr(splits.por_tamano, name)
        sets[known & rows["n_qubits"].between(size.min, size.max)] = name
    return sets


def _unseen_oracles(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    oracles = rows[~rows["desconocido"] & (rows["familia"] == ORACLE_FAMILY)]
    unit = oracles["oraculo_id"].where(oracles["oraculo_id"] != "", "mqt:" + oracles["group_id"])
    oracles = oracles.assign(unidad=unit)
    return stratified_assignment(oracles, splits, "oraculos_no_vistos", "unidad").reindex(
        rows.index
    )


def _by_level(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    known = rows[~rows["desconocido"]]
    sets = stratified_assignment(known, splits, "por_nivel").reindex(rows.index)
    train_levels, test_levels = splits.por_nivel.train, splits.por_nivel.test
    keep = (
        (sets.isin([TRAIN, VAL]) & rows["nivel"].isin(train_levels))
        | ((sets == TEST) & rows["nivel"].isin(test_levels))
        | rows["solo_evaluacion"]
    )
    return sets.where(keep)


def _open(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    sets = _random(rows, config, splits)
    sets[rows["desconocido"]] = UNKNOWN
    return sets


BUILDERS: dict[str, Callable[[pd.DataFrame, DatasetConfig, SplitsConfig], pd.Series]] = {
    "aleatorio": _random,
    "por_tamano": _by_size,
    "oraculos_no_vistos": _unseen_oracles,
    "por_nivel": _by_level,
    "abierto": _open,
}


def build_protocol(
    rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig, protocol: str, cap: int | None
) -> pd.DataFrame:
    """Split table of one protocol (rows not assigned to any set are left out)."""
    sets = BUILDERS[protocol](rows, config, splits)
    labelled_in_train = rows["solo_evaluacion"] & (sets == TRAIN)
    sets = sets.where(~labelled_in_train)
    if cap is not None:
        # ``abierto`` must keep exactly the known part of ``aleatorio``: same sampling seed
        cap_key = "aleatorio" if protocol == "abierto" else protocol
        sets = cap_train_groups(rows, sets, cap, splits, cap_key)
    table = rows.assign(conjunto=sets).dropna(subset=["conjunto"])
    return table[list(SPLIT_COLUMNS)].sort_values("id").reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Whole step
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitsReport:
    """Outcome of the split step."""

    tables: dict[str, pd.DataFrame]
    created: tuple[str, ...]
    cap: int | None
    below_minimum: dict[str, list[str]]
    report_path: Path


def split_path(config: DatasetConfig, protocol: str) -> Path:
    """``<splits>/<protocol>.csv``."""
    return config.rutas.splits / f"{protocol}.csv"


def _read_split(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.assign(solo_evaluacion=table["solo_evaluacion"] == "True")


def below_minimum(table: pd.DataFrame, rows: pd.DataFrame, minimum: int) -> list[str]:
    """Labels with fewer than ``minimum`` unique train groups."""
    train = rows.loc[rows["id"].isin(table.loc[table["conjunto"] == TRAIN, "id"])]
    counts = train.groupby("etiqueta")["group_id"].nunique()
    known = rows.loc[~rows["desconocido"] & ~rows["solo_evaluacion"], "etiqueta"].unique()
    return sorted(label for label in known if counts.get(label, 0) < minimum)


def create_splits(
    config: DatasetConfig,
    splits: SplitsConfig,
    *,
    protocols: Sequence[str] = PROTOCOLS,
    force: bool = False,
) -> SplitsReport:
    """Create the missing (or, with ``force``, all) protocol files and the split report."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    rows = candidate_rows(config, splits, metadata)
    sampling = splits.balanceo.modo == "muestreo"
    cap = resolve_cap(config, splits, rows) if sampling else None
    tables, created = {}, []
    for protocol in protocols:
        path = split_path(config, protocol)
        if path.is_file() and not force:
            tables[protocol] = _read_split(path)
            continue
        table = build_protocol(rows, config, splits, protocol, cap)
        write_table(table, path)
        if not sampling:
            sets = rows["id"].map(table.set_index("id")["conjunto"])
            weights = class_weights(rows, sets)
            write_text_atomic(
                json.dumps(weights, indent=2), config.rutas.splits / f"pesos_{protocol}.json"
            )
        tables[protocol] = table
        created.append(protocol)
    minimum = splits.minimo_por_clase_train
    low = {
        p: below_minimum(t, rows, minimum) for p, t in tables.items() if p != "oraculos_no_vistos"
    }
    for protocol, labels in low.items():
        if labels:
            logger.warning(
                "%s: labels below minimo_por_clase_train=%d: %s", protocol, minimum, labels
            )
    report = write_split_report(config, splits, rows, tables, cap, low)
    return SplitsReport(tables, tuple(created), cap, low, report)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule, *body])


def _set_summary(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Circuits, groups and labelled (evaluation-only) circuits per set."""
    merged = table.merge(rows[["id", "group_id"]], on="id")
    summary = merged.groupby("conjunto").agg(
        circuitos=("id", "count"),
        grupos=("group_id", "nunique"),
        solo_evaluacion=("solo_evaluacion", "sum"),
    )
    return summary.reindex([s for s in (TRAIN, VAL, TEST, UNKNOWN) if s in summary.index])


def _labelled_in_test(table: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Labelled F7 circuits in test per label."""
    test_ids = table.loc[(table["conjunto"] == TEST) & table["solo_evaluacion"], "id"]
    return rows[rows["id"].isin(test_ids)].groupby("etiqueta").size()


def write_split_report(
    config: DatasetConfig,
    splits: SplitsConfig,
    rows: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    cap: int | None,
    low: dict[str, list[str]],
) -> Path:
    """Write ``results/informe_particiones.md``."""
    by_label = rows[~rows["desconocido"] & ~rows["solo_evaluacion"]]
    labels = sorted(by_label["etiqueta"].unique())
    train_groups = pd.DataFrame(index=labels)
    labelled = pd.DataFrame(index=labels)
    for protocol, table in tables.items():
        train_ids = table.loc[table["conjunto"] == TRAIN, "id"]
        train = rows[rows["id"].isin(train_ids)]
        train_groups[protocol] = train.groupby("etiqueta")["group_id"].nunique()
        labelled[protocol] = _labelled_in_test(table, rows)
    train_groups = train_groups.fillna(0).astype(int).rename_axis("etiqueta").reset_index()
    labelled = labelled.fillna(0).astype(int).rename_axis("etiqueta").reset_index()
    mode = splits.balanceo.modo
    balance = (
        f"muestreo: como máximo {cap} grupos de train por etiqueta (val y test completos)"
        if cap is not None
        else "ponderación: todos los grupos; pesos en `pesos_<protocolo>.json`"
    )
    sections = [
        "# Informe de particiones (Fase 7)",
        f"Semilla {splits.semilla}; balanceo `{mode}` ({balance}); "
        f"mínimo de grupos de train por etiqueta: {splits.minimo_por_clase_train}.",
    ]
    for protocol, table in tables.items():
        sections += [f"## {protocol}", _markdown(_set_summary(table, rows).reset_index())]
        if low.get(protocol):
            sections.append(f"Etiquetas por debajo del mínimo: {', '.join(low[protocol])}.")
    sections += [
        "## Grupos de train por etiqueta",
        _markdown(train_groups),
        "## Circuitos con F7 (variante etiquetada) en test por etiqueta",
        _markdown(labelled),
    ]
    path = config.rutas.resultados / REPORT_NAME
    write_text_atomic("\n\n".join(sections) + "\n", path)
    return path
