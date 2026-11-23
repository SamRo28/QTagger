"""Single command-line entry point for the dataset pipeline (``qcclass``).

Heavy dependencies (Qiskit, MQT Bench) are imported inside each command so that
``qcclass --help`` stays fast.
"""

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from qcclass.config import DatasetConfig, SplitsConfig
    from qcclass.data.generar_mqt import GenerationReport

DEFAULT_DATASET_CONFIG = Path("configs/dataset.yaml")
DEFAULT_SPLITS_CONFIG = Path("configs/splits.yaml")

app = typer.Typer(
    name="qcclass",
    help="Build the quantum-circuit classification dataset.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[Path, typer.Option("--config", help="Dataset configuration file.")]
SplitsOption = Annotated[Path, typer.Option("--splits", help="Split configuration file.")]
# Every command that evaluates models must take this option and pass it to
# ``qcclass.versionado.split_ids``: without it, the test sets cannot be read.
EvaluarTestOption = Annotated[
    bool,
    typer.Option(
        "--evaluar-test",
        help="Allow reading the test sets (logged in results/accesos_test.log).",
    ),
]


def _split_list(value: str | None) -> list[str] | None:
    """Split a comma-separated option value; ``None`` when the option was not given."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_dataset_config(path: Path) -> "DatasetConfig":
    """Load the dataset configuration, turning validation errors into a clean exit."""
    from qcclass.config import ConfigError, load_dataset_config

    try:
        return load_dataset_config(path)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _load_splits_config(path: Path, dataset: "DatasetConfig") -> "SplitsConfig":
    """Load the split configuration, turning validation errors into a clean exit."""
    from qcclass.config import ConfigError, load_splits_config

    try:
        return load_splits_config(path, dataset=dataset)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _print_generation_report(report: "GenerationReport", levels: tuple[str, ...]) -> None:
    """Print circuits generated per algorithm and level, and errors per algorithm and type."""
    import pandas as pd

    typer.echo(f"\nCircuits written: {len(report.rows)} (already present: {report.skipped})")
    if len(report.rows):
        table = pd.crosstab(report.rows["algoritmo"], report.rows["nivel"])
        table = table.reindex(columns=[level for level in levels if level in table.columns])
        table.loc["total"] = table.sum()
        typer.echo(table.to_string())
    typer.echo(f"\nErrors: {len(report.errors)}")
    if len(report.errors):
        typer.echo(pd.crosstab(report.errors["algoritmo"], report.errors["tipo"]).to_string())


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Configure logging before running any subcommand.

    Third-party libraries (Qiskit logs every transpiler pass at INFO) only show warnings;
    the ``qcclass`` loggers show INFO, or DEBUG with ``--verbose``.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("qcclass").setLevel(logging.DEBUG if verbose else logging.INFO)
    # torch_geometric triggers this at import time with torch 2.15; it is not actionable here.
    warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")


@app.command()
def generar(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    algoritmos: Annotated[
        str | None, typer.Option(help="Comma-separated algorithms (default: all configured).")
    ] = None,
    min_qubits: Annotated[
        int | None, typer.Option("--min", help="Smallest size (default: tamanos.min).")
    ] = None,
    max_qubits: Annotated[
        int | None, typer.Option("--max", help="Largest size (default: tamanos.max).")
    ] = None,
    niveles: Annotated[
        str | None, typer.Option(help="Comma-separated levels (default: all configured).")
    ] = None,
    forzar: Annotated[bool, typer.Option("--forzar", help="Regenerate existing files.")] = False,
    procesos: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
) -> None:
    """F1: generate raw circuits from MQT Bench."""
    from qcclass.data.generar_mqt import build_selection, generate

    dataset = _load_dataset_config(config)
    try:
        selection = build_selection(
            dataset, _split_list(algoritmos), min_qubits, max_qubits, _split_list(niveles)
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = generate(dataset, selection, force=forzar, processes=procesos)
    _print_generation_report(report, selection.levels)


@app.command()
def oraculos(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    algoritmos: Annotated[
        str | None, typer.Option(help="Comma-separated subset of dj,bv,simon,grover.")
    ] = None,
    min_qubits: Annotated[
        int | None, typer.Option("--min", help="Smallest size (default: tamanos.min).")
    ] = None,
    max_qubits: Annotated[
        int | None, typer.Option("--max", help="Largest size (default: tamanos.max).")
    ] = None,
    forzar: Annotated[bool, typer.Option("--forzar", help="Regenerate existing files.")] = False,
    procesos: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
) -> None:
    """Generate the in-house oracle circuits (dj, bv, simon, grover)."""
    import pandas as pd

    from qcclass.data.oraculos import generate_oracles, select_oracle_algorithms

    dataset = _load_dataset_config(config)
    low = dataset.tamanos.min if min_qubits is None else min_qubits
    high = dataset.tamanos.max if max_qubits is None else max_qubits
    try:
        if low < 1 or low > high:
            raise ValueError(f"invalid size range [{low}, {high}]")
        algorithms = select_oracle_algorithms(dataset, _split_list(algoritmos))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = generate_oracles(
        dataset, algorithms, range(low, high + 1), force=forzar, processes=procesos
    )
    typer.echo(f"\nOracle circuits written: {len(report.rows)} (already present: {report.skipped})")
    if len(report.rows):
        sizes = report.rows.groupby("algoritmo")["n_qubits"]
        summary = pd.DataFrame(
            {"circuits": sizes.size(), "sizes": sizes.nunique(), "max_n": sizes.max()}
        )
        typer.echo(summary.to_string())
    typer.echo(f"\nErrors: {len(report.errors)}")
    if len(report.errors):
        typer.echo(pd.crosstab(report.errors["algoritmo"], report.errors["tipo"]).to_string())


@app.command()
def normalizar(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    niveles: Annotated[
        str | None, typer.Option(help="Comma-separated levels (default: all).")
    ] = None,
    forzar: Annotated[bool, typer.Option("--forzar", help="Regenerate existing files.")] = False,
    procesos: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
) -> None:
    """F2: flatten and normalise raw circuits to the canonical gate set."""
    import pandas as pd

    from qcclass.data.normalizar import normalize

    dataset = _load_dataset_config(config)
    report = normalize(dataset, levels=_split_list(niveles), force=forzar, processes=procesos)
    typer.echo(
        f"\nCanonical circuits written: {len(report.rows)} (already present: {report.skipped})"
    )
    typer.echo(f"Discarded: {len(report.log)}")
    if len(report.log):
        typer.echo(pd.crosstab(report.log["algoritmo"], report.log["tipo"]).to_string())


@app.command()
def deduplicar(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    forzar: Annotated[bool, typer.Option("--forzar", help="Recompute every fingerprint.")] = False,
    procesos: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
) -> None:
    """Detect duplicated canonical circuits and write the deduplication report."""
    from qcclass.data.calidad import (
        usable,
        usable_per_label_and_level,
        write_deduplication_report,
    )
    from qcclass.data.deduplicar import deduplicate

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    report = deduplicate(dataset, force=forzar, processes=procesos)
    rows = usable(report.canonical)
    typer.echo(f"\nUsable canonical circuits: {len(rows)} of {len(report.canonical)}")
    typer.echo(usable_per_label_and_level(rows).to_string())
    path = write_deduplication_report(dataset, split_config, report.canonical, report.conflicts)
    typer.echo(f"\nReport: {path}")


@app.command()
def formatos(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    procesos: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
) -> None:
    """F3-F7: derive features, graphs, tokens and gate labels."""
    from qcclass.formatos.construir import build_formats

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    report = build_formats(dataset, split_config.algoritmos_reservados, processes=procesos)
    typer.echo(
        f"\nCircuits with formats: {report.circuits} (labelled F7 variants: {report.labelled})"
    )
    typer.echo(f"Trainable class labels ({len(report.classes)}): {', '.join(report.classes)}")
    typer.echo(f"Families ({len(report.families)}): {', '.join(report.families)}")
    typer.echo("\nConsistency check (F2 gates = F4 nodes = F6 tokens = F7 labels):")
    for row in report.consistency:
        labels = "-" if row.labels_f7 is None else row.labels_f7
        status = "ok" if row.consistent else "MISMATCH"
        typer.echo(
            f"  {row.id}: {row.gates_f2} / {row.nodes_f4} / {row.tokens_f6} / {labels}  {status}"
        )
    if not all(row.consistent for row in report.consistency):
        raise typer.Exit(code=1)


@app.command()
def particiones(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    forzar: Annotated[bool, typer.Option("--forzar", help="Rebuild existing splits.")] = False,
) -> None:
    """Create the train/val/test split protocols."""
    from qcclass.data.particiones import create_splits

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    report = create_splits(dataset, split_config, force=forzar)
    created = ", ".join(report.created) or "none (use --forzar to rebuild)"
    typer.echo(f"\nProtocols created: {created}")
    for protocol, table in report.tables.items():
        counts = table["conjunto"].value_counts().to_dict()
        cap = report.caps.get(protocol)
        typer.echo(f"  {protocol}: {counts}" + (f" (train cap {cap} groups)" if cap else ""))
        if report.below_minimum.get(protocol):
            typer.echo(f"    below minimum: {', '.join(report.below_minimum[protocol])}")
    for label, sizes in report.size_exclusions.items():
        typer.echo(f"  por_tamano excludes {label}: only {len(sizes)} sizes {sizes}")
    typer.echo(f"\nReport: {report.report_path}")


@app.command()
def calidad(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
) -> None:
    """Write the dataset quality report and check metadata/files coherence."""
    from qcclass.data.calidad import write_dataset_report

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    report = write_dataset_report(dataset, split_config)
    failed = [check for check in report.checks if not check.ok]
    typer.echo(f"\nQuality report: {report.path}")
    typer.echo(f"Coherence checks: {len(report.checks) - len(failed)}/{len(report.checks)} ok")
    for check in failed:
        typer.echo(f"  {check.name}: {len(check.problems)} problems, e.g. {check.problems[:3]}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def todo(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    procesos: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
) -> None:
    """Run the whole pipeline in order.

    generar -> oraculos -> normalizar -> deduplicar -> formatos -> particiones -> calidad.
    Every step is incremental; existing splits are kept (use ``particiones --forzar``).
    """
    steps = [
        (
            "generar",
            lambda: generar(
                config=config,
                procesos=procesos,
                algoritmos=None,
                min_qubits=None,
                max_qubits=None,
                niveles=None,
                forzar=False,
            ),
        ),
        (
            "oraculos",
            lambda: oraculos(
                config=config,
                procesos=procesos,
                algoritmos=None,
                min_qubits=None,
                max_qubits=None,
                forzar=False,
            ),
        ),
        (
            "normalizar",
            lambda: normalizar(config=config, procesos=procesos, niveles=None, forzar=False),
        ),
        (
            "deduplicar",
            lambda: deduplicar(config=config, splits=splits, procesos=procesos, forzar=False),
        ),
        ("formatos", lambda: formatos(config=config, splits=splits, procesos=procesos)),
        ("particiones", lambda: particiones(config=config, splits=splits, forzar=False)),
        ("calidad", lambda: calidad(config=config, splits=splits)),
    ]
    for number, (name, step) in enumerate(steps, start=1):
        typer.echo(f"\n=== [{number}/{len(steps)}] {name} ===")
        step()


@app.command()
def versionar(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    nombre: Annotated[
        str | None, typer.Option(help="Version name to record, e.g. dataset-v1.")
    ] = None,
    comprobar: Annotated[
        bool, typer.Option("--comprobar", help="Only check the files against VERSION.json.")
    ] = False,
) -> None:
    """Record (``--nombre``) or check (``--comprobar``) the dataset version."""
    from qcclass.versionado import VersionError, read_version, version_differences, write_version

    dataset = _load_dataset_config(config)
    if comprobar == (nombre is not None):
        raise typer.BadParameter("use exactly one of --nombre or --comprobar")
    try:
        if nombre is not None:
            version = write_version(dataset, nombre)
            typer.echo(
                f"Version {version.nombre}: {version.huella} ({len(version.archivos)} files)"
            )
            return
        version = read_version(dataset)
        differences = version_differences(dataset)
    except VersionError as exc:
        typer.echo(f"Version error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if differences:
        typer.echo(f"Dataset does NOT match {version.nombre}:")
        for difference in differences:
            typer.echo(f"  {difference}")
        raise typer.Exit(code=1)
    typer.echo(f"Dataset matches {version.nombre} ({version.huella[:12]})")
