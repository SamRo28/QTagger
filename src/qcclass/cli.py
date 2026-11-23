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

app = typer.Typer(
    name="qcclass",
    help="Build the quantum-circuit classification dataset.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[Path, typer.Option("--config", help="Dataset configuration file.")]


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
def oraculos() -> None:
    """Generate the in-house oracle circuits (dj, bv, simon, grover)."""
    _not_implemented("oraculos")


@app.command()
def normalizar() -> None:
    """F2: flatten and normalise raw circuits to the canonical gate set."""
    _not_implemented("normalizar")


@app.command()
def deduplicar() -> None:
    """Detect duplicated canonical circuits."""
    _not_implemented("deduplicar")


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
