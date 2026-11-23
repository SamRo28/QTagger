"""Single command-line entry point for the dataset pipeline (``qcclass``).

Heavy dependencies (Qiskit, MQT Bench) are imported inside each command so that
``qcclass --help`` stays fast.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from qcclass.config import DatasetConfig
    from qcclass.data.generar_mqt import GenerationReport

NOT_IMPLEMENTED_MESSAGE = "no implementado"
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


def _not_implemented(command: str) -> None:
    """Print the placeholder message for a pipeline step that is not implemented yet."""
    typer.echo(f"{command}: {NOT_IMPLEMENTED_MESSAGE}")


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
    from qcclass.config import ConfigError, load_splits_config
    from qcclass.data.calidad import (
        unique_canonical,
        unique_per_class_and_level,
        write_deduplication_report,
    )
    from qcclass.data.deduplicar import deduplicate

    dataset = _load_dataset_config(config)
    try:
        split_config = load_splits_config(splits, dataset=dataset)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    report = deduplicate(dataset, force=forzar, processes=procesos)
    unique = unique_canonical(report.canonical)
    typer.echo(f"\nUnique canonical circuits: {len(unique)} of {len(report.canonical)}")
    typer.echo(unique_per_class_and_level(unique).to_string())
    path = write_deduplication_report(dataset, split_config, report.canonical, report.conflicts)
    typer.echo(f"\nReport: {path}")


@app.command()
def formatos() -> None:
    """F3-F7: derive features, graphs, tokens and gate labels."""
    _not_implemented("formatos")


@app.command()
def particiones() -> None:
    """Create the train/val/test split protocols."""
    _not_implemented("particiones")


@app.command()
def calidad() -> None:
    """Write the dataset quality report."""
    _not_implemented("calidad")


@app.command()
def todo() -> None:
    """Run the whole pipeline in order."""
    _not_implemented("todo")
