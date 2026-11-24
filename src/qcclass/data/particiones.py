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
* ``por_tamano``: relative cuts per label on the logical size of each group (canonical widths
  of routed circuits can be larger): the smallest 70 % of its valid sizes go to train,
  the next 15 % to val and the largest 15 % to test (at least one size each); labels with
  fewer than three sizes are left out and logged.
* ``oraculos_no_vistos``: oracle family only; the split unit is ``oraculo_id`` (the group
  for MQT circuits), so no test oracle is seen in training.
* ``por_nivel``: groups split as in ``aleatorio``; train keeps the train levels of its
  groups (ALG + INDEP), val the train levels of its groups and test the test levels
  (NATIVEGATES + MAPPED) of other groups.
* ``abierto``: ``aleatorio`` plus the unknown circuits as ``desconocido``.

Cross-validation blocks: ``aleatorio_bloques.csv`` (``group_id``, ``bloque``) assigns every
group of train + val of ``aleatorio`` to one of ``bloques_cv`` blocks with scikit-learn's
``StratifiedGroupKFold`` (stratified by label, seeded). Evaluation-only variants are not used
to build the blocks; they inherit the block of their group through ``group_id``.

Balancing (``balanceo``, val and test always kept whole, nothing deleted from disk):
``muestreo`` samples the train groups of each label down to ``max_bases_por_clase`` (by
default ``factor_mediana`` x the median of train groups per label); ``ponderacion`` keeps
every group and computes loss weights inversely proportional to the train groups of each
class; ``combinado`` does both. ``balanceo_<protocolo>.json`` stores the cap and the
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig, SplitsConfig
from qcclass.data.deduplicar import family_label
from qcclass.formatos.etiquetas_puerta import LABELLED_VARIANT
from qcclass.metadatos import METADATA_COLUMNS, read_table, write_table
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

PROTOCOLS = ("aleatorio", "por_tamano", "oraculos_no_vistos", "por_nivel", "abierto")
TRAIN, VAL, TEST, UNKNOWN = "train", "val", "test", "desconocido"
CV_PROTOCOL = "aleatorio"
BLOCKS_FILE = "aleatorio_bloques.csv"
BLOCK_COLUMNS = ("group_id", "bloque")
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
    rows["n_logico"] = logical_sizes(metadata, rows)
    return rows


def logical_sizes(metadata: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Logical size of the group of every row (``n_qubits`` of its raw rows).

    Canonical widths can differ within a group (routed ``mapped`` circuits activate extra
    physical qubits), so protocols and size breakdowns use the logical size of the group.
    Rows whose group has no raw row keep their own ``n_qubits``.
    """
    raw = metadata[metadata["variante_comp"] == ""]
    logical = raw.groupby("group_id")["n_qubits"].first().astype(int)
    own = rows["n_qubits"].astype(int)
    return rows["group_id"].map(logical).fillna(own).astype(int)


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


def _train_groups(rows: pd.DataFrame, sets: pd.Series) -> pd.Series:
    """Train groups per label (known circuits used for training only)."""
    train = rows[(sets == TRAIN) & ~rows["desconocido"] & ~rows["solo_evaluacion"]]
    return train.groupby("etiqueta")["group_id"].nunique()


def resolve_cap(splits: SplitsConfig, rows: pd.DataFrame, sets: pd.Series) -> int:
    """``max_bases_por_clase``, or ``factor_mediana`` x the median of train groups per label."""
    if splits.balanceo.max_bases_por_clase is not None:
        return splits.balanceo.max_bases_por_clase
    median = statistics.median(_train_groups(rows, sets))
    return max(1, round(splits.balanceo.factor_mediana * median))


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
    """Loss weights inversely proportional to the train groups of each class (mean 1).

    ``algoritmo``: trainable class labels (family-only and unknown circuits excluded);
    ``familia``: families of every known training circuit.
    """
    train = rows[(sets == TRAIN) & ~rows["desconocido"] & ~rows["solo_evaluacion"]]
    weights = {}
    for head, column in (("algoritmo", "clase"), ("familia", "familia")):
        groups = train[train[column] != ""].groupby(column)["group_id"].nunique()
        inverse = 1.0 / groups
        weights[head] = {name: float(w) for name, w in (inverse / inverse.mean()).items()}
    return weights


# --------------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------------


def _random(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    known = rows[~rows["desconocido"]]
    return stratified_assignment(known, splits, "aleatorio").reindex(rows.index)


def size_cuts(sizes: Sequence[int], splits: SplitsConfig) -> dict[int, str] | None:
    """Relative cuts: smallest sizes to train, next to val, largest to test.

    With ``k`` distinct sizes, val and test get ``max(1, round(k * fraction))`` sizes each
    and train the rest. ``None`` when there are fewer than three sizes.
    """
    ordered = sorted(set(sizes))
    if len(ordered) < 3:
        return None
    n_val = max(1, round(len(ordered) * splits.aleatorio.val))
    n_test = max(1, round(len(ordered) * splits.aleatorio.test))
    n_train = len(ordered) - n_val - n_test
    names = [TRAIN] * n_train + [VAL] * n_val + [TEST] * n_test
    return dict(zip(ordered, names, strict=True))


def size_exclusions(rows: pd.DataFrame, splits: SplitsConfig) -> dict[str, list[int]]:
    """Labels left out of ``por_tamano`` (fewer than three sizes) and their sizes."""
    known = rows[~rows["desconocido"] & ~rows["solo_evaluacion"]]
    sizes = known.groupby("etiqueta")["n_logico"].agg(lambda s: sorted(set(s)))
    return {label: values for label, values in sizes.items() if size_cuts(values, splits) is None}


def _by_size(rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig) -> pd.Series:
    sets = pd.Series(np.nan, index=rows.index, dtype=object)
    known = rows[~rows["desconocido"]]
    for _, group in known.groupby("etiqueta", sort=True):
        training = group[~group["solo_evaluacion"]]
        cuts = size_cuts(list(training["n_logico"]), splits)
        if cuts is None:
            continue
        sets[group.index] = group["n_logico"].map(cuts)
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


@dataclass(frozen=True)
class ProtocolSplit:
    """Split table of one protocol and its balancing."""

    table: pd.DataFrame
    cap: int | None
    weights: dict[str, dict[str, float]] | None


def build_protocol(
    rows: pd.DataFrame, config: DatasetConfig, splits: SplitsConfig, protocol: str
) -> ProtocolSplit:
    """Build one protocol: assign sets, drop labelled variants in train, balance."""
    sets = BUILDERS[protocol](rows, config, splits)
    labelled_in_train = rows["solo_evaluacion"] & (sets == TRAIN)
    sets = sets.where(~labelled_in_train)
    mode = splits.balanceo.modo
    cap = None
    if mode in ("muestreo", "combinado"):
        cap = resolve_cap(splits, rows, sets)
        # ``abierto`` must keep exactly the known part of ``aleatorio``: same sampling seed
        cap_key = "aleatorio" if protocol == "abierto" else protocol
        sets = cap_train_groups(rows, sets, cap, splits, cap_key)
    weights = class_weights(rows, sets) if mode in ("ponderacion", "combinado") else None
    table = rows.assign(conjunto=sets).dropna(subset=["conjunto"])
    table = table[list(SPLIT_COLUMNS)].sort_values("id").reset_index(drop=True)
    return ProtocolSplit(table, cap, weights)


# --------------------------------------------------------------------------------------------
# Whole step
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitsReport:
    """Outcome of the split step."""

    tables: dict[str, pd.DataFrame]
    created: tuple[str, ...]
    caps: dict[str, int | None]
    below_minimum: dict[str, list[str]]
    size_exclusions: dict[str, list[int]]
    report_path: Path
    blocks: pd.DataFrame | None = None
    blocks_created: bool = False


def split_path(config: DatasetConfig, protocol: str) -> Path:
    """``<splits>/<protocol>.csv``."""
    return config.rutas.splits / f"{protocol}.csv"


def balance_path(config: DatasetConfig, protocol: str) -> Path:
    """``<splits>/balanceo_<protocol>.json``: cap used and loss weights."""
    return config.rutas.splits / f"balanceo_{protocol}.json"


def blocks_path(config: DatasetConfig) -> Path:
    """``<splits>/aleatorio_bloques.csv``: cross-validation block of every group."""
    return config.rutas.splits / BLOCKS_FILE


def cv_blocks(rows: pd.DataFrame, table: pd.DataFrame, splits: SplitsConfig) -> pd.DataFrame:
    """Block (0 .. ``bloques_cv`` - 1) of every group of train + val of ``aleatorio``.

    ``StratifiedGroupKFold`` over the circuits of train + val (evaluation-only variants
    left out), stratified by label and grouped by ``group_id``, with a seed derived from the
    split seed. Returns the columns ``group_id`` and ``bloque``, sorted by ``group_id``.
    """
    pool_ids = table.loc[table["conjunto"].isin([TRAIN, VAL]) & ~table["solo_evaluacion"], "id"]
    pool = rows[rows["id"].isin(pool_ids)].sort_values("id")
    folds = StratifiedGroupKFold(
        n_splits=splits.bloques_cv,
        shuffle=True,
        random_state=derive_seed(splits.semilla, CV_PROTOCOL, "bloques"),
    )
    block_of: dict[str, int] = {}
    for block, (_, held_out) in enumerate(
        folds.split(pool, pool["etiqueta"], groups=pool["group_id"])
    ):
        block_of.update(dict.fromkeys(pool["group_id"].iloc[held_out], block))
    variants = rows[rows["id"].isin(table.loc[table["solo_evaluacion"], "id"])]
    orphans = sorted(set(variants["group_id"]) - set(block_of))
    if orphans:
        logger.warning("%d evaluation-only groups have no block: %s", len(orphans), orphans[:5])
    blocks = pd.DataFrame(sorted(block_of.items()), columns=list(BLOCK_COLUMNS))
    return blocks.astype({"bloque": int})


def create_blocks(
    config: DatasetConfig,
    splits: SplitsConfig,
    rows: pd.DataFrame,
    table: pd.DataFrame,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Write ``aleatorio_bloques.csv`` if missing (or with ``force``); return (blocks, created)."""
    path = blocks_path(config)
    if path.is_file() and not force:
        return pd.read_csv(path, dtype={"group_id": str, "bloque": int}), False
    blocks = cv_blocks(rows, table, splits)
    write_table(blocks, path)
    logger.info("%d groups assigned to %d blocks", len(blocks), splits.bloques_cv)
    return blocks, True


def block_summary(blocks: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Groups per label and block (labels of the regular, known circuits)."""
    labels = rows[~rows["solo_evaluacion"]].drop_duplicates("group_id")
    joined = blocks.merge(labels[["group_id", "etiqueta"]], on="group_id", how="left")
    summary = joined.pivot_table(
        index="etiqueta", columns="bloque", values="group_id", aggfunc="count", fill_value=0
    )
    summary.columns = [f"bloque {c}" for c in summary.columns]
    return summary.rename_axis("etiqueta").reset_index()


def _read_split(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.assign(solo_evaluacion=table["solo_evaluacion"] == "True")


def below_minimum(table: pd.DataFrame, rows: pd.DataFrame, minimum: int) -> list[str]:
    """Labels of the protocol with fewer than ``minimum`` unique train groups."""
    joined = rows[rows["id"].isin(table["id"])]
    known = joined.loc[~joined["desconocido"] & ~joined["solo_evaluacion"], "etiqueta"].unique()
    train = rows[rows["id"].isin(table.loc[table["conjunto"] == TRAIN, "id"])]
    counts = train.groupby("etiqueta")["group_id"].nunique()
    return sorted(label for label in known if counts.get(label, 0) < minimum)


def _write_protocol(config: DatasetConfig, protocol: str, split: ProtocolSplit) -> None:
    write_table(split.table, split_path(config, protocol))
    balance = {"max_bases_por_clase": split.cap, "pesos": split.weights}
    write_text_atomic(json.dumps(balance, indent=2), balance_path(config, protocol))


def _read_cap(config: DatasetConfig, protocol: str) -> int | None:
    path = balance_path(config, protocol)
    return (
        json.loads(path.read_text(encoding="utf-8"))["max_bases_por_clase"]
        if path.is_file()
        else None
    )


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
    tables, caps, created = {}, {}, []
    for protocol in protocols:
        path = split_path(config, protocol)
        if path.is_file() and not force:
            tables[protocol], caps[protocol] = _read_split(path), _read_cap(config, protocol)
            continue
        split = build_protocol(rows, config, splits, protocol)
        _write_protocol(config, protocol, split)
        tables[protocol], caps[protocol] = split.table, split.cap
        created.append(protocol)
    excluded = size_exclusions(rows, splits) if "por_tamano" in protocols else {}
    for label, sizes in excluded.items():
        logger.warning("por_tamano: '%s' left out, only %d sizes %s", label, len(sizes), sizes)
    minimum = splits.minimo_por_clase_train
    low = {
        p: below_minimum(t, rows, minimum) for p, t in tables.items() if p != "oraculos_no_vistos"
    }
    for protocol, labels in low.items():
        if labels:
            logger.warning(
                "%s: labels below minimo_por_clase_train=%d: %s", protocol, minimum, labels
            )
    blocks, blocks_created = None, False
    if CV_PROTOCOL in tables:
        blocks, blocks_created = create_blocks(
            config, splits, rows, tables[CV_PROTOCOL], force=force
        )
    report = write_split_report(config, splits, rows, tables, caps, low, excluded, blocks)
    return SplitsReport(tables, tuple(created), caps, low, excluded, report, blocks, blocks_created)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule, *body])


def set_summary(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Real distribution per set: circuits and groups (with shares over train/val/test)."""
    merged = table.merge(rows[["id", "group_id"]], on="id")
    summary = merged.groupby("conjunto").agg(
        circuitos=("id", "count"),
        grupos=("group_id", "nunique"),
        solo_evaluacion=("solo_evaluacion", "sum"),
    )
    summary = summary.reindex([s for s in (TRAIN, VAL, TEST, UNKNOWN) if s in summary.index])
    known = summary.drop(index=UNKNOWN, errors="ignore")
    summary["% circuitos"] = (100 * known["circuitos"] / known["circuitos"].sum()).round(1)
    summary["% grupos"] = (100 * known["grupos"] / known["grupos"].sum()).round(1)
    return summary.astype(object).fillna("—")


def _labelled_in_test(table: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Labelled F7 circuits in test per label."""
    test_ids = table.loc[(table["conjunto"] == TEST) & table["solo_evaluacion"], "id"]
    return rows[rows["id"].isin(test_ids)].groupby("etiqueta").size()


def write_split_report(
    config: DatasetConfig,
    splits: SplitsConfig,
    rows: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    caps: dict[str, int | None],
    low: dict[str, list[str]],
    excluded: dict[str, list[int]],
    blocks: pd.DataFrame | None = None,
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
    balance = splits.balanceo
    cap_rule = (
        "fijo"
        if balance.max_bases_por_clase
        else f"{balance.factor_mediana:g} × mediana de grupos de train"
    )
    sections = [
        "# Informe de particiones (Fase 7)",
        f"Semilla {splits.semilla}. Balanceo `{balance.modo}`: como máximo "
        f"`max_bases_por_clase` grupos de train por etiqueta "
        f"({cap_rule}), y pesos de pérdida inversamente proporcionales a los grupos de "
        "train restantes "
        "(`balanceo_<protocolo>.json`). Val y test se conservan completos. Mínimo de grupos de "
        f"train por etiqueta: {splits.minimo_por_clase_train}.",
    ]
    for protocol, table in tables.items():
        cap = caps.get(protocol)
        title = f"## {protocol}" + (f" (tope de train: {cap} grupos)" if cap else "")
        sections += [title, _markdown(set_summary(table, rows).reset_index())]
        if low.get(protocol):
            sections.append(f"Etiquetas por debajo del mínimo: {', '.join(low[protocol])}.")
        if protocol == "por_tamano" and excluded:
            listed = "; ".join(f"`{k}` ({len(v)} tamaños: {v})" for k, v in excluded.items())
            sections.append(f"Fuera del protocolo por tener menos de 3 tamaños: {listed}.")
    sections += [
        "## Grupos de train por etiqueta",
        _markdown(train_groups),
        "## Circuitos con F7 (variante etiquetada) en test por etiqueta",
        _markdown(labelled),
    ]
    if blocks is not None:
        sections += [
            f"## Bloques de validación cruzada ({CV_PROTOCOL}, train + val)",
            f"`{BLOCKS_FILE}`: `StratifiedGroupKFold` con {splits.bloques_cv} bloques, "
            "estratificado por etiqueta. Grupos por etiqueta y bloque:",
            _markdown(block_summary(blocks, rows)),
        ]
    path = config.rutas.resultados / REPORT_NAME
    write_text_atomic("\n\n".join(sections) + "\n", path)
    return path
