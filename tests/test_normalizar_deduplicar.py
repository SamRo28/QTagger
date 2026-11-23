"""Tests for F2 (normalisation) and deduplication."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qcclass.circuitos import CANONICAL_BASIS, qpy_path, read_qpy
from qcclass.config import DatasetConfig, load_splits_config
from qcclass.data.calidad import propose_classes, write_deduplication_report
from qcclass.data.deduplicar import (
    _label_and_mark,
    deduplicate,
    mark_duplicates,
    normalized_qasm,
    structural_fingerprint,
)
from qcclass.data.generar_mqt import build_selection, generate
from qcclass.data.normalizar import (
    log_path,
    normalize,
    remove_idle_qubits,
    strip_non_unitary,
    to_canonical,
)
from qcclass.metadatos import METADATA_COLUMNS, read_table

ConfigFactory = Callable[..., DatasetConfig]  # the ``make_config`` fixture


def _metadata(config: DatasetConfig) -> pd.DataFrame:
    return read_table(config.rutas.metadata, METADATA_COLUMNS)


@pytest.mark.parametrize("optimization_level", [0, 1, 2])
def test_hadamard_becomes_canonical_and_equivalent(optimization_level: int) -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(2)
    canonical = to_canonical(circuit, optimization_level, seed=1)
    assert set(canonical.count_ops()) <= set(CANONICAL_BASIS)
    assert "h" not in canonical.count_ops()
    assert Operator(circuit).equiv(Operator.from_circuit(canonical))


def test_strip_non_unitary_and_idle_qubits() -> None:
    circuit = QuantumCircuit(5, 2)
    circuit.h(1)
    circuit.barrier()
    circuit.cx(1, 3)
    circuit.measure([1, 3], [0, 1])
    body = strip_non_unitary(circuit, drop_idle_qubits=True)
    assert body.num_qubits == 2
    assert set(body.count_ops()) == {"h", "cx"}
    assert remove_idle_qubits(QuantumCircuit(3)).num_qubits == 0


def test_normalization_variants_and_equivalence(make_config: ConfigFactory) -> None:
    config = make_config()
    generate(config, build_selection(config, ["qft", "ghz"], 3, 4))
    report = normalize(config)

    canonical = _metadata(config).query("variante_comp != ''")
    assert len(report.rows) == len(canonical) == 2 * 2 * (3 + 3 + 1 + 1)
    variants = canonical.groupby("nivel")["variante_comp"].unique().map(sorted)
    assert variants["alg"] == variants["indep"] == ["c0", "c1", "c2"]
    assert variants["nativegates"] == variants["mapped"] == ["c0"]
    assert set(canonical["opt_level_comp"]) == {"0", "1", "2"}
    assert (canonical.groupby("group_id")["id"].count() == 8).all()

    raw = _metadata(config).query("variante_comp == ''").set_index("id")
    for row in canonical.itertuples():
        circuit = read_qpy(qpy_path(config.rutas.raiz / row.ruta_canonical))
        assert set(circuit.count_ops()) <= set(CANONICAL_BASIS)
        assert circuit.num_qubits == int(row.n_qubits)  # mapped: idle qubits removed
        assert int(row.n_puertas) == circuit.size()
        text = (config.rutas.raiz / row.ruta_canonical).read_text(encoding="utf-8")
        assert "\ngate " not in text
        if row.nivel in ("alg", "indep"):
            raw_id = row.id.rsplit("_c", 1)[0]
            source = read_qpy(qpy_path(config.rutas.raiz / raw.loc[raw_id, "ruta_raw"]))
            body = strip_non_unitary(source, drop_idle_qubits=False)
            assert Operator(body).equiv(Operator.from_circuit(circuit))


def test_normalization_is_incremental_and_discards_large(make_config: ConfigFactory) -> None:
    generator = make_config()
    generate(generator, build_selection(generator, ["qft"], 2, 5, levels=["alg"]))
    config = make_config(max_nodos=40)  # same directories, tighter limit for F2
    first = normalize(config)
    assert len(first.log) > 0
    assert set(first.log["tipo"]) == {"max_nodos"}
    again = normalize(config)
    assert len(again.rows) == 0 and again.skipped == len(first.rows)
    log = read_table(log_path(config), ["id_raw"])
    assert len(log) == len(first.log)  # replaced, not appended


def test_qasm_normalisation_drops_comments_and_spaces() -> None:
    text = "OPENQASM 3.0; // header\n/* block\ncomment */ qubit[2]   q;\n\n h q[0];"
    assert normalized_qasm(text) == "OPENQASM 3.0; qubit[2] q; h q[0];"


def test_structural_fingerprint_ignores_angles_and_qubit_labels() -> None:
    a = QuantumCircuit(2)
    a.rz(0.1, 0)
    a.cx(0, 1)
    b = QuantumCircuit(2)
    b.rz(0.7, 1)
    b.cx(1, 0)
    c = QuantumCircuit(2)
    c.cx(0, 1)
    c.rz(0.1, 0)
    assert structural_fingerprint(a) == structural_fingerprint(b)
    assert structural_fingerprint(a) != structural_fingerprint(c)


def _fingerprinted(rows: list[tuple[str, ...]]) -> pd.DataFrame:
    columns = [
        "id",
        "algoritmo",
        "familia",
        "nivel",
        "n_qubits",
        "group_id",
        "huella_sha",
        "huella_wl",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.assign(ruta_canonical="x.qasm")


def test_mark_duplicates_policy() -> None:
    rows = _fingerprinted(
        [
            # same text, same label -> duplicates of the alg representative
            ("indep_x_c0", "x", "f", "indep", "3", "x3", "s1", "w1"),
            ("alg_x_c0", "x", "f", "alg", "3", "x3", "s1", "w1"),
            ("alg_x_c1", "x", "f", "alg", "3", "x3", "s1", "w1"),
            # same text, different labels -> exact conflict, nothing marked
            ("alg_y_c0", "y", "f", "alg", "2", "y2", "s2", "w2"),
            ("alg_z_c0", "z", "g", "alg", "2", "z2", "s2", "w2"),
            # different text, same structure, different labels -> structural conflict only
            ("alg_y_c1", "y", "f", "alg", "4", "y4", "s3", "w3"),
            ("alg_x_c2", "x", "f", "alg", "4", "x4", "s4", "w3"),
        ]
    )
    duplicate_of, conflicts = mark_duplicates(rows.assign(etiqueta=rows["algoritmo"]))
    assert duplicate_of["indep_x_c0"] == duplicate_of["alg_x_c1"] == "alg_x_c0"
    assert duplicate_of[["alg_x_c0", "alg_y_c0", "alg_z_c0", "alg_y_c1", "alg_x_c2"]].eq("").all()
    exact = set(conflicts.loc[conflicts["tipo"] == "exacto", "id"])
    structural = set(conflicts.loc[conflicts["tipo"] == "estructural", "id"])
    assert exact == {"alg_y_c0", "alg_z_c0"}
    assert {"alg_y_c1", "alg_x_c2"} <= structural


def test_labels_merges_conflicts_and_family_only(make_config: ConfigFactory) -> None:
    config = make_config()
    rows = _fingerprinted(
        [
            # merged classes: identical circuits become plain duplicates
            ("alg_full_c0", "full_adder", "aritmetica", "alg", "4", "fa4", "s1", "w1"),
            ("alg_cdkm_c0", "cdkm_ripple_carry_adder", "aritmetica", "alg", "4", "cd4", "s1", "w1"),
            # different labels: excluded from both
            ("alg_dj_c0", "dj", "oraculos", "alg", "3", "dj3", "s2", "w2"),
            ("alg_bv_c0", "bv", "oraculos", "alg", "3", "bv3", "s2", "w2"),
            # vbe with fewer than 22 bases -> family label only
            ("alg_vbe_c0", "vbe_ripple_carry_adder", "aritmetica", "alg", "4", "vb4", "s3", "w3"),
        ]
    )
    labelled, conflicts = _label_and_mark(config, rows)
    by_id = labelled.set_index("id")
    assert (
        by_id.loc["alg_full_c0", "clase"]
        == by_id.loc["alg_cdkm_c0", "clase"]
        == ("cdkm_ripple_carry_adder")
    )
    assert {by_id.loc["alg_full_c0", "exclusion"], by_id.loc["alg_cdkm_c0", "exclusion"]} == {
        "",
        "duplicado",
    }
    assert (
        by_id.loc["alg_dj_c0", "exclusion"]
        == by_id.loc["alg_bv_c0", "exclusion"]
        == ("conflicto_etiquetas")
    )
    assert by_id.loc["alg_vbe_c0", "clase"] == ""
    assert set(conflicts.loc[conflicts["tipo"] == "exacto", "id"]) == {"alg_dj_c0", "alg_bv_c0"}


def test_deduplication_end_to_end(
    make_config: ConfigFactory, splits_yaml: Path, tmp_path: Path
) -> None:
    config = make_config()
    generate(config, build_selection(config, ["ghz", "bell"], 2, 4))
    normalize(config)
    report = deduplicate(config)

    canonical = report.canonical
    assert (canonical["huella_sha"] != "").all() and (canonical["huella_wl"] != "").all()
    # bell is merged into ghz: their identical n=2 circuits are duplicates, not conflicts
    assert set(canonical["clase"]) == {"ghz"}
    assert (canonical["exclusion"] != "conflicto_etiquetas").all()
    bell_n2 = canonical[(canonical["algoritmo"] == "bell") & (canonical["n_qubits"] == "2")]
    assert (bell_n2["exclusion"] == "duplicado").any()
    by_id = canonical.set_index("id")
    for row in canonical[canonical["duplicado_de"] != ""].itertuples():
        target = by_id.loc[row.duplicado_de]
        assert (target.huella_sha, target.clase) == (row.huella_sha, row.clase)
        assert target.duplicado_de == ""
        assert row.exclusion == "duplicado"

    splits = load_splits_config(splits_yaml, dataset=config)
    path = write_deduplication_report(config, splits, _metadata(config), report.conflicts)
    assert path == tmp_path / "results" / "informe_deduplicacion.md"
    text = path.read_text(encoding="utf-8")
    assert "Propuesta de clases" in text and "Conflictos de etiquetas excluidos" in text


def test_merge_proposed_only_above_half_of_smaller_class(
    make_config: ConfigFactory, splits_yaml: Path
) -> None:
    config = make_config()
    splits = load_splits_config(splits_yaml, dataset=config)
    rows = []
    for i in range(30):  # qft: 30 bases
        rows.append((f"qft{i}", "qft", "qft_fase", "alg", "5", f"qft_g{i}", f"q{i}", "w"))
    for i in range(30):  # ae: 30 bases, 18 identical to qft (60 %) -> merge
        sha = f"q{i}" if i < 18 else f"a{i}"
        rows.append((f"ae{i}", "ae", "qft_fase", "alg", "5", f"ae_g{i}", sha, "w"))
    for i in range(25):  # hhl reserved; qpeexact: 25 bases, 5 identical to qft (20 %)
        sha = f"q{i}" if i < 5 else f"p{i}"
        rows.append((f"qp{i}", "qpeexact", "qft_fase", "alg", "5", f"qp_g{i}", sha, "w"))
    for i in range(3):  # wstate: few bases, nothing identical -> family label
        rows.append((f"w{i}", "wstate", "preparacion_estados", "alg", "5", f"w_g{i}", f"w{i}", "w"))
    canonical = _fingerprinted(rows).assign(
        clase=lambda f: f["algoritmo"], exclusion="", duplicado_de=""
    )
    proposals = {p.label: p for p in propose_classes(config, splits, canonical)}
    assert proposals["ae"].proposal == "fusionar con qft"
    assert proposals["qpeexact"].proposal == "mantener"
    assert proposals["wstate"].proposal == "solo etiqueta de familia"
