"""Tests for the split protocols (Phase 7)."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qcclass.config import (
    BalancingConfig,
    DatasetConfig,
    SplitsConfig,
    load_dataset_config,
    load_splits_config,
)
from qcclass.data.deduplicar import deduplicate
from qcclass.data.generar_mqt import build_selection, generate
from qcclass.data.normalizar import normalize
from qcclass.data.oraculos import generate_oracles
from qcclass.data.particiones import (
    PROTOCOLS,
    TRAIN,
    balance_path,
    blocks_path,
    build_protocol,
    candidate_rows,
    create_splits,
    size_cuts,
    size_exclusions,
    split_path,
)
from qcclass.formatos.construir import build_labelled_variants
from qcclass.metadatos import METADATA_COLUMNS, read_table

ALGORITHMS = ["ghz", "qft", "wstate", "hhl", "qwalk", "randomcircuit", "half_adder"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[DatasetConfig, SplitsConfig]:
    """A small dataset with classes, family-only labels, unknowns, oracles and F7 variants."""
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
    data["oraculos"] = {"por_tamano": 3}
    folder = tmp_path_factory.mktemp("splits") / "configs"
    folder.mkdir()
    (folder / "dataset.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_dataset_config(folder / "dataset.yaml")
    generate(config, build_selection(config, ALGORITHMS, 3, 22, levels=["alg", "nativegates"]))
    generate_oracles(config, ["dj", "bv"], range(3, 8))
    normalize(config, processes=4)
    deduplicate(config)
    build_labelled_variants(config, processes=4)
    splits = load_splits_config(root / "configs" / "splits.yaml", dataset=config)
    return config, splits


@pytest.fixture(scope="module")
def created(dataset: tuple[DatasetConfig, SplitsConfig]) -> dict[str, pd.DataFrame]:
    config, splits = dataset
    return create_splits(config, splits, force=True).tables


@pytest.fixture(scope="module")
def rows(dataset: tuple[DatasetConfig, SplitsConfig]) -> pd.DataFrame:
    config, splits = dataset
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    return candidate_rows(config, splits, metadata).set_index("id")


def _joined(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    return table.merge(rows, left_on="id", right_index=True, suffixes=("", "_fila"))


def test_every_protocol_is_written(
    dataset: tuple[DatasetConfig, SplitsConfig], created: dict[str, pd.DataFrame]
) -> None:
    config, _ = dataset
    assert set(created) == set(PROTOCOLS)
    for protocol, table in created.items():
        assert split_path(config, protocol).is_file()
        assert list(table.columns) == ["id", "conjunto", "solo_evaluacion"]
        assert len(table) > 0 and table["id"].is_unique


def test_no_group_in_two_sets(created: dict[str, pd.DataFrame], rows: pd.DataFrame) -> None:
    for protocol, table in created.items():
        sets_per_group = _joined(table, rows).groupby("group_id")["conjunto"].nunique()
        assert (sets_per_group == 1).all(), protocol


def test_reserved_and_unknown_never_in_train(
    dataset: tuple[DatasetConfig, SplitsConfig],
    created: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
) -> None:
    config, splits = dataset
    unknown = {*splits.algoritmos_reservados, *config.control_negativo}
    for protocol, table in created.items():
        joined = _joined(table, rows)
        assert not (joined["conjunto"] == TRAIN)[joined["algoritmo"].isin(unknown)].any(), protocol
    open_protocol = _joined(created["abierto"], rows)
    unknown_rows = open_protocol[open_protocol["algoritmo"].isin(unknown)]
    assert len(unknown_rows) and set(unknown_rows["conjunto"]) == {"desconocido"}
    assert "desconocido" not in set(created["aleatorio"]["conjunto"])


def test_labelled_variants_follow_their_group_and_never_train(
    created: dict[str, pd.DataFrame], rows: pd.DataFrame
) -> None:
    for protocol, table in created.items():
        joined = _joined(table, rows)
        labelled = joined[joined["variante_comp"] == "etiquetada"]
        assert labelled["solo_evaluacion"].all()
        assert (joined.loc[~joined["solo_evaluacion"], "variante_comp"] != "etiquetada").all()
        assert not (labelled["conjunto"] == TRAIN).any(), protocol
    joined = _joined(created["aleatorio"], rows)
    assert joined["solo_evaluacion"].any()


def test_size_cuts_are_relative_per_label() -> None:
    splits = load_splits_config(Path(__file__).resolve().parents[1] / "configs" / "splits.yaml")
    assert size_cuts([5, 3], splits) is None
    assert size_cuts([4, 2, 3], splits) == {2: "train", 3: "val", 4: "test"}
    cuts = size_cuts(list(range(2, 22)), splits)  # 20 sizes: 14 / 3 / 3
    assert [cuts[n] for n in range(2, 22)] == ["train"] * 14 + ["val"] * 3 + ["test"] * 3


def test_size_and_level_protocols(
    dataset: tuple[DatasetConfig, SplitsConfig],
    created: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
) -> None:
    _, splits = dataset
    by_size = _joined(created["por_tamano"], rows)
    regular = by_size[~by_size["solo_evaluacion"]]
    for label, group in regular.groupby("etiqueta"):
        sizes = group.groupby("conjunto")["n_qubits"].agg(["min", "max"])
        assert set(sizes.index) == {"train", "val", "test"}, label
        assert sizes.loc["train", "max"] < sizes.loc["val", "min"], label
        assert sizes.loc["val", "max"] < sizes.loc["test", "min"], label
    excluded = size_exclusions(rows.reset_index(), splits)
    assert "ghz" not in excluded
    assert not set(excluded) & set(regular["etiqueta"])
    by_level = _joined(created["por_nivel"], rows)
    regular = by_level[~by_level["solo_evaluacion"]]
    assert set(regular.loc[regular["conjunto"] == "train", "nivel"]) <= {"alg", "indep"}
    assert set(regular.loc[regular["conjunto"] == "test", "nivel"]) <= {"nativegates", "mapped"}


def test_unseen_oracles(created: dict[str, pd.DataFrame], rows: pd.DataFrame) -> None:
    joined = _joined(created["oraculos_no_vistos"], rows)
    assert set(joined["familia"]) == {"oraculos"}
    oracle = joined["oraculo_id"].where(joined["oraculo_id"] != "", joined["group_id"])
    train = set(oracle[joined["conjunto"] == "train"])
    assert not train & set(oracle[joined["conjunto"] == "test"])


def test_combined_balancing(
    dataset: tuple[DatasetConfig, SplitsConfig],
    created: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
) -> None:
    config, _ = dataset
    balance = json.loads(balance_path(config, "aleatorio").read_text(encoding="utf-8"))
    cap = balance["max_bases_por_clase"]
    joined = _joined(created["aleatorio"], rows)
    train = joined[(joined["conjunto"] == "train") & ~joined["solo_evaluacion"]]
    assert train.groupby("etiqueta")["group_id"].nunique().max() <= cap
    # weights are inversely proportional to the remaining train groups (mean 1)
    weights = balance["pesos"]["algoritmo"]
    class_groups = train[train["clase"] != ""].groupby("clase")["group_id"].nunique()
    assert set(weights) == set(class_groups.index)
    products = {weights[name] * class_groups[name] for name in weights}
    assert max(products) == pytest.approx(min(products))
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)
    assert set(balance["pesos"]["familia"]) <= set(config.familias)


def test_splits_are_created_once(
    dataset: tuple[DatasetConfig, SplitsConfig], created: dict[str, pd.DataFrame]
) -> None:
    config, splits = dataset
    path = split_path(config, "aleatorio")
    before = path.read_text(encoding="utf-8")
    path.write_text(before.replace(",val,", ",VAL,"), encoding="utf-8")  # mark the file
    report = create_splits(config, splits)
    assert report.created == ()
    assert ",VAL," in path.read_text(encoding="utf-8")  # not regenerated
    report = create_splits(config, splits, protocols=["aleatorio"], force=True)
    assert path.read_text(encoding="utf-8") == before  # same seed, same split


def test_minimum_warning(
    dataset: tuple[DatasetConfig, SplitsConfig], caplog: pytest.LogCaptureFixture
) -> None:
    config, splits = dataset
    strict = replace(splits, minimo_por_clase_train=10_000)
    with caplog.at_level(logging.WARNING):
        report = create_splits(config, strict, protocols=["aleatorio"], force=True)
    assert report.below_minimum["aleatorio"]
    assert "below minimo_por_clase_train" in caplog.text
    create_splits(config, splits, protocols=["aleatorio"], force=True)  # restore


def test_weighting_mode_writes_weights(dataset: tuple[DatasetConfig, SplitsConfig]) -> None:
    config, splits = dataset
    weighted = replace(
        splits,
        balanceo=BalancingConfig(modo="ponderacion", max_bases_por_clase=None, factor_mediana=3),
    )
    report = create_splits(config, weighted, protocols=["aleatorio"], force=True)
    assert report.caps["aleatorio"] is None
    balance = json.loads(balance_path(config, "aleatorio").read_text(encoding="utf-8"))
    assert balance["max_bases_por_clase"] is None
    assert set(balance["pesos"]) == {"algoritmo", "familia"}
    create_splits(config, splits, protocols=["aleatorio"], force=True)  # restore


def test_open_protocol_matches_random_on_known_circuits(created: dict[str, pd.DataFrame]) -> None:
    random_split = created["aleatorio"].set_index("id")["conjunto"]
    open_split = created["abierto"].set_index("id")["conjunto"]
    known = open_split[open_split != "desconocido"]
    assert known.sort_index().equals(random_split.sort_index())


def test_labelled_variants_keep_their_exclusion_after_deduplication(
    dataset: tuple[DatasetConfig, SplitsConfig],
) -> None:
    config, _ = dataset
    deduplicate(config)  # a second run must not touch the F7 variants
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    labelled = metadata[metadata["variante_comp"] == "etiquetada"]
    assert len(labelled) and set(labelled["exclusion"]) == {"etiquetada"}


def test_size_protocol_uses_the_logical_size_of_the_group(
    dataset: tuple[DatasetConfig, SplitsConfig],
) -> None:
    """A routed mapped circuit wider than its group must not leave its group's set."""
    config, splits = dataset
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    rows = candidate_rows(config, splits, metadata)
    target = rows[(rows["algoritmo"] == "ghz") & (rows["nivel"] == "nativegates")].iloc[0]
    rows.loc[target.name, "n_qubits"] = 99  # as if routing had activated extra qubits
    assert rows.loc[target.name, "n_logico"] != 99
    split = build_protocol(rows, config, splits, "por_tamano")
    joined = split.table.merge(rows[["id", "group_id"]], on="id")
    assert (joined.groupby("group_id")["conjunto"].nunique() == 1).all()


def test_cross_validation_blocks(
    dataset: tuple[DatasetConfig, SplitsConfig],
    created: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
) -> None:
    config, splits = dataset
    blocks = pd.read_csv(blocks_path(config), dtype={"group_id": str})
    assert list(blocks.columns) == ["group_id", "bloque"] and blocks["group_id"].is_unique
    assert set(blocks["bloque"]) == set(range(splits.bloques_cv))
    joined = _joined(created["aleatorio"], rows)
    pool = joined[joined["conjunto"].isin(["train", "val"])]
    assert set(blocks["group_id"]) == set(pool.loc[~pool["solo_evaluacion"], "group_id"])
    # evaluation-only variants inherit the block of their group; test groups have none
    assert set(pool.loc[pool["solo_evaluacion"], "group_id"]) <= set(blocks["group_id"])
    assert not set(joined.loc[joined["conjunto"] == "test", "group_id"]) & set(blocks["group_id"])
    # stratified: a label with at least bloques_cv groups appears in every block
    labelled = blocks.merge(pool.drop_duplicates("group_id")[["group_id", "etiqueta"]])
    per_label = labelled.groupby("etiqueta")["bloque"].nunique()
    sizes = labelled.groupby("etiqueta")["group_id"].nunique()
    assert (per_label[sizes >= splits.bloques_cv] == splits.bloques_cv).all()


def test_blocks_are_created_once(
    dataset: tuple[DatasetConfig, SplitsConfig], created: dict[str, pd.DataFrame]
) -> None:
    config, splits = dataset
    before = blocks_path(config).read_text(encoding="utf-8")
    report = create_splits(config, splits)
    assert not report.blocks_created
    assert blocks_path(config).read_text(encoding="utf-8") == before
    report = create_splits(config, splits, protocols=["aleatorio"], force=True)
    assert report.blocks_created and blocks_path(config).read_text(encoding="utf-8") == before
