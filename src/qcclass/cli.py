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
# Only ``qcclass evaluar-final`` (phase 17) takes this option: it asks for a
# ``qcclass.versionado.final_evaluation_permit``, without which the protected sets (test and
# the unknown circuits of ``abierto``) cannot be read.
EvaluarTestOption = Annotated[
    bool,
    typer.Option(
        "--evaluar-test",
        help="Read the protected sets (test, desconocido); logged in results/accesos_test.log.",
    ),
]
DEFAULT_SERVER_TEST_CONFIG = Path("configs/experimentos/prueba_servidores.yaml")
DEFAULT_BASELINE_CONFIG = Path("configs/experimentos/linea_base.yaml")
DEFAULT_EVALUATION_CONFIG = Path("configs/evaluacion.yaml")
DEFAULT_DIAGNOSTICS_CONFIG = Path("configs/experimentos/diagnosticos_linea_base.yaml")
DEFAULT_GNN_CONFIG = Path("configs/experimentos/gnn_referencia.yaml")
DEFAULT_GNN_SEARCH_CONFIG = Path("configs/experimentos/gnn_busqueda.yaml")


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
    if report.blocks is not None:
        state = "created" if report.blocks_created else "kept"
        per_block = report.blocks["bloque"].value_counts().sort_index().to_dict()
        typer.echo(f"  cross-validation blocks ({state}): groups per block {per_block}")
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


def _check_version(dataset: "DatasetConfig", *, strict: bool = False) -> None:
    """Print whether the data match ``VERSION.json``; exit with code 1 if they do not (or,
    with ``strict``, if Python or a tracked library has another version)."""
    from qcclass.versionado import (
        VersionError,
        library_differences,
        read_version,
        version_differences,
    )

    try:
        version = read_version(dataset)
        differences = version_differences(dataset)
        libraries = library_differences(dataset)
    except VersionError as exc:
        typer.echo(f"Version error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if differences:
        typer.echo(f"Dataset does NOT match {version.nombre}:")
        for difference in differences:
            typer.echo(f"  {difference}")
        raise typer.Exit(code=1)
    typer.echo(
        f"Dataset matches {version.nombre} ({version.huella[:12]}, {len(version.archivos)} files)"
    )
    if not libraries:
        typer.echo(f"Python and libraries match ({len(version.librerias)} versions)")
    for difference in libraries:
        typer.echo(f"  {'ERROR' if strict else 'warning'}, version differs: {difference}")
    if libraries and strict:
        raise typer.Exit(code=1)


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
    from qcclass.versionado import VersionError, write_version

    dataset = _load_dataset_config(config)
    if comprobar == (nombre is not None):
        raise typer.BadParameter("use exactly one of --nombre or --comprobar")
    if nombre is None:
        _check_version(dataset)
        return
    try:
        version = write_version(dataset, nombre)
    except VersionError as exc:
        typer.echo(f"Version error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Version {version.nombre}: {version.huella} ({len(version.archivos)} files)")


@app.command()
def verificar(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    estricto: Annotated[
        bool, typer.Option("--estricto", help="Also fail if a library version differs.")
    ] = False,
) -> None:
    """Check the data fingerprints and the library versions against VERSION.json."""
    _check_version(_load_dataset_config(config), strict=estricto)


@app.command()
def ping(config: ConfigOption = DEFAULT_DATASET_CONFIG) -> None:
    """Check the connection with the MLflow server and the shared Optuna storage."""
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env, machine
    from qcclass.seguimiento.mlflow_cliente import check_mlflow, tracking_uri
    from qcclass.seguimiento.optuna_cliente import connect, storage_uri, study_summary

    dataset = _load_dataset_config(config)
    env = load_env(dataset.rutas.raiz)
    typer.echo(f"Machine: {machine('(QCCLASS_EQUIPO not set)')}; .env: {env or 'not found'}")
    failures = 0
    try:
        uri = tracking_uri()
        typer.echo(f"MLflow {uri}: ok, server {check_mlflow(uri)}")
    except ServerConnectionError as exc:
        typer.echo(f"MLflow: FAILED: {exc}", err=True)
        failures += 1
    try:
        uri = storage_uri()
        studies = study_summary(connect(uri, retries=False))
        typer.echo(f"Optuna {uri}: ok, {len(studies)} studies")
    except ServerConnectionError as exc:
        typer.echo(f"Optuna: FAILED: {exc}", err=True)
        failures += 1
    if failures:
        raise typer.Exit(code=1)


@app.command()
def servidores(
    accion: Annotated[str, typer.Argument(help="arrancar, parar or estado.")],
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
) -> None:
    """Start, stop or inspect the MLflow and Optuna servers (on the server machine)."""
    from qcclass.seguimiento import servidores as servers
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env

    dataset = _load_dataset_config(config)
    load_env(dataset.rutas.raiz)
    try:
        settings = servers.ServerSettings.from_env()
        if accion == "arrancar":
            pids = servers.start(settings)
            typer.echo(f"Servers running: {pids}")
            typer.echo(
                f"MLflow: http://<this machine>:{settings.mlflow_port}  "
                f"Optuna: grpc://<this machine>:{settings.optuna_port}"
            )
        elif accion == "parar":
            stopped = servers.stop(settings)
            typer.echo(f"Stopped: {[name for name, done in stopped.items() if done] or 'none'}")
        elif accion == "estado":
            for service, state in servers.status(settings).items():
                typer.echo(f"{service}: {state}")
        else:
            raise typer.BadParameter("use arrancar, parar or estado")
    except ServerConnectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("prueba-servidores")
def prueba_servidores(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    experimento: Annotated[
        Path, typer.Option(help="Experiment configuration.")
    ] = DEFAULT_SERVER_TEST_CONFIG,
) -> None:
    """Log a test run in MLflow and add trials to a shared Optuna study (phase 9 check)."""
    from qcclass.config import ConfigError, load_experiment_config
    from qcclass.seguimiento.entorno import ServerConnectionError
    from qcclass.seguimiento.prueba import run_server_test
    from qcclass.versionado import VersionError

    dataset = _load_dataset_config(config)
    try:
        settings = load_experiment_config(
            experimento, ("experimento_mlflow", "estudio_optuna", "intentos", "semilla")
        )
        result = run_server_test(dataset, settings)
    except (ConfigError, ServerConnectionError, VersionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"MLflow run {result.run_id} in experiment {result.experiment}")
    typer.echo(f"Optuna study {result.study}: {result.added} trials added by {result.machine}")
    for machine_name, count in sorted(result.trials_by_machine.items()):
        typer.echo(f"  {machine_name}: {count} complete trials")


@app.command()
def medir(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    dispositivo: Annotated[
        str, typer.Option(help="auto, cuda, mps or cpu (auto: cuda > mps > cpu).")
    ] = "auto",
    presupuesto_nodos: Annotated[int, typer.Option(help="Maximum nodes per batch.")] = 100_000,
    epocas: Annotated[int, typer.Option(help="Measured epochs (0: none).", min=0)] = 2,
    max_lotes: Annotated[
        int | None, typer.Option(help="Batches per epoch at most (quick runs).", min=1)
    ] = None,
    num_workers: Annotated[int, typer.Option(help="DataLoader worker processes.", min=0)] = 0,
    amp: Annotated[bool, typer.Option("--amp", help="Mixed precision (CUDA only).")] = False,
    dim: Annotated[int, typer.Option(help="Hidden width of the reference model.", min=1)] = 64,
    capas: Annotated[int, typer.Option(help="GINEConv layers.", min=1)] = 3,
    barrido: Annotated[
        bool, typer.Option("--barrido", help="Also sweep 25k-400k node budgets (50 batches).")
    ] = False,
    resumen: Annotated[
        bool, typer.Option("--resumen", help="Only gather the JSON files into RESUMEN.md.")
    ] = False,
    equipo: Annotated[
        str | None, typer.Option(help="Machine name in the file name (default: host name).")
    ] = None,
) -> None:
    """Measure the training cost of the reference GNN on this machine (train set only)."""
    import os
    import sys

    dataset = _load_dataset_config(config)
    if resumen:
        from qcclass.medicion.medir import MEASUREMENT_DIR, SUMMARY_FILE
        from qcclass.medicion.resumen import write_summary

        try:
            path = write_summary(dataset.rutas.resultados / MEASUREMENT_DIR, SUMMARY_FILE)
        except FileNotFoundError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"Summary written to {path}")
        return
    # read by torch when it loads: set it before the first import of torch
    torch_loaded = "torch" in sys.modules
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from qcclass.medicion.equipo import resolve_device
    from qcclass.medicion.medir import MeasurementOptions, run_measurement
    from qcclass.versionado import VersionError

    try:
        device = resolve_device(dispositivo)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if amp and device != "cuda":
        raise typer.BadParameter("--amp is only available on CUDA")
    options = MeasurementOptions(
        dispositivo=dispositivo,
        presupuesto_nodos=presupuesto_nodos,
        epocas=epocas,
        max_lotes=max_lotes,
        num_workers=num_workers,
        amp=amp,
        dim=dim,
        capas=capas,
        barrido=barrido,
        equipo=equipo,
    )
    fallback = {
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "torch_importado_antes": torch_loaded,
    }
    typer.echo(f"Device: {device}")
    try:
        path, result = run_measurement(
            dataset,
            options,
            device=device,
            mps_fallback=fallback,
            on_model=lambda n: typer.echo(f"Reference model: {n:,} parameters"),
        )
    except VersionError as exc:
        typer.echo(f"Dataset does not match its version, measurement aborted: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_measurement(result)
    typer.echo(f"Result written to {path}")


def _print_measurement(result: dict) -> None:
    """Short console summary of a measurement."""
    if (initial := result["memoria_inicial"]) and "util_mb" in initial:
        typer.echo(
            f"CUDA memory before the first batch: {initial['memoria_reservada_mb']:,.0f} MB "
            f"reserved, {initial['contexto_y_otros_mb']:,.0f} MB context and others, "
            f"{initial['util_mb']:,.0f} of {initial['memoria_total_mb']:,.0f} MB usable"
        )
    worst = result["peor_caso"]
    state = "ok" if worst["ok"] else "OUT OF MEMORY"
    typer.echo(f"Worst case ({worst['n_nodos']:,} nodes): {state}")
    for epoch in result["epocas"]:
        typer.echo(
            f"Epoch {epoch['epoca']}: {epoch['tiempo_total_s']:.1f} s "
            f"(data {epoch['tiempo_datos_s']:.1f} s, compute {epoch['tiempo_computo_s']:.1f} s), "
            f"{epoch['lotes']} batches, {epoch['nodos_s']:,.0f} nodes/s, "
            f"{epoch['grafos_s']:,.1f} graphs/s, loss {epoch['perdida_media']:.4f}"
        )
    if result["mps_fallback"] is not None:
        ops = result["mps_fallback"]["operaciones_cpu"]
        typer.echo(f"MPS ops falling back to CPU: {', '.join(ops) if ops else 'none'}")
    if projection := result["proyeccion"]:
        full = projection["entrenamiento_completo_h"]
        optuna = projection["optuna_h"]
        typer.echo(
            "Projection ("
            + projection["fuente"]
            + "): "
            + ", ".join(f"{e} epochs {h} h" for e, h in full.items())
            + "; Optuna "
            + ", ".join(f"{t} trials {h} h" for t, h in optuna.items())
        )
    if sweep := result["barrido"]:
        typer.echo(f"Largest working budget: {sweep['mayor_presupuesto_ok']}")


@app.command("entrenar-gnn")
def entrenar_gnn(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    experimento: Annotated[Path, typer.Option(help="GNN experiment configuration.")] = (
        DEFAULT_GNN_CONFIG
    ),
    bloque: Annotated[int, typer.Option(help="Cross-validation block held out.")] = 0,
    dispositivo: Annotated[str, typer.Option(help="auto (CUDA, else CPU), cuda, cpu, mps.")] = (
        "auto"
    ),
    sin_precision_mixta: Annotated[
        bool, typer.Option("--sin-precision-mixta", help="Train in float32 on CUDA.")
    ] = False,
    procesos: Annotated[
        int | None, typer.Option(help="Data-loading processes (default QCCLASS_PROCESOS_CARGA).")
    ] = None,
    max_lotes: Annotated[
        int | None, typer.Option(help="Truncate every epoch to N batches (quick checks).")
    ] = None,
    epocas: Annotated[int | None, typer.Option(help="Override the number of epochs.")] = None,
) -> None:
    """Phase 11: train the GNN on one block of the cross-validation (never the test set);
    resumes from its checkpoints when launched again."""
    from dataclasses import replace

    from qcclass.config import ConfigError
    from qcclass.entrenamiento.gnn_experimento import (
        cuda_allocator_environment,
        load_gnn_experiment,
        train_one_block,
    )
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env
    from qcclass.versionado import VersionError

    cuda_allocator_environment()
    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    try:
        experiment = load_gnn_experiment(experimento)
        training = experiment.training
        if sin_precision_mixta:
            training = replace(training, precision_mixta=False)
        if epocas is not None:
            training = replace(training, epocas=epocas)
        load_env(dataset.rutas.raiz)
        result = train_one_block(
            dataset,
            split_config,
            experiment,
            block=bloque,
            device=dispositivo,
            training=training,
            workers=procesos,
            max_batches=max_lotes,
        )
    except (ConfigError, ServerConnectionError, VersionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    epochs = [h["segundos_epoca"] for h in result.history]
    typer.echo(
        f"Block {result.block}: algorithm macro F1 {100 * result.f1_macro_algoritmo:.1f} %, "
        f"family {100 * result.f1_macro_familia:.1f} %, {len(epochs)} epochs, median "
        f"{sorted(epochs)[len(epochs) // 2]:.1f} s/epoch, total {result.seconds / 60:.1f} min"
    )
    peaks = {
        key: max((h.get(key, 0.0) for h in result.history), default=0.0)
        for key in ("memoria_gpu_mb", "memoria_gpu_reservada_mb", "ram_max_con_workers_mb")
    }
    train = [h["segundos_entrenamiento"] for h in result.history]
    typer.echo(
        f"Peaks: GPU {peaks['memoria_gpu_mb']:.0f} MB allocated, "
        f"{peaks['memoria_gpu_reservada_mb']:.0f} MB reserved; RAM with workers "
        f"{peaks['ram_max_con_workers_mb']:.0f} MB; training {sorted(train)[len(train) // 2]:.1f} "
        "s/epoch (median, without evaluation)"
    )
    typer.echo(f"Results: {result.directory}")


@app.command("buscar-gnn")
def buscar_gnn(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    experimento: Annotated[Path, typer.Option(help="GNN search configuration.")] = (
        DEFAULT_GNN_SEARCH_CONFIG
    ),
    dispositivo: Annotated[str, typer.Option(help="auto (CUDA, else CPU), cuda, cpu, mps.")] = (
        "auto"
    ),
    procesos: Annotated[
        int | None, typer.Option(help="Data-loading processes (default QCCLASS_PROCESOS_CARGA).")
    ] = None,
) -> None:
    """Phase 12: Optuna search of the GNN with 5-block cross-validation (never the test set);
    launching it again resumes the study and the interrupted trial."""
    from qcclass.config import ConfigError
    from qcclass.entrenamiento.busqueda_gnn import ESTIMATE_FILE, load_search_config, run_search
    from qcclass.entrenamiento.gnn_experimento import OUTPUT_DIR, cuda_allocator_environment
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env
    from qcclass.seguimiento.optuna_cliente import connect
    from qcclass.versionado import VersionError

    cuda_allocator_environment()
    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    try:
        search = load_search_config(experimento)
        load_env(dataset.rutas.raiz)
        storage = connect()
        study = run_search(
            dataset, split_config, search, storage=storage, device=dispositivo, workers=procesos
        )
    except (ConfigError, ServerConnectionError, VersionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    complete = study.get_trials(states=(optuna_state("COMPLETE"),))
    typer.echo(f"Study {search.name}: {len(study.trials)} trials, {len(complete)} complete")
    if complete:
        best = study.best_trial
        typer.echo(f"Best: trial {best.number}, CV macro F1 {100 * best.value:.2f} %")
    typer.echo(f"Estimate: {dataset.rutas.resultados / OUTPUT_DIR / search.name / ESTIMATE_FILE}")


def optuna_state(name: str):  # noqa: ANN201 - optuna.trial.TrialState
    """Optuna trial state by name (imported lazily)."""
    import optuna

    return optuna.trial.TrialState[name]


@app.command("calibrar-gnn")
def calibrar_gnn(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    experimento: Annotated[Path, typer.Option(help="Calibration configuration.")] = Path(
        "configs/experimentos/gnn_calibracion.yaml"
    ),
    dispositivo: Annotated[
        str, typer.Option(help="cuda (memory model and time) or cpu (time only).")
    ] = "cuda",
) -> None:
    """Phase 12: GPU memory model and real time per configuration of the GNN on this machine
    (CUDA; train part and held-out block of one cross-validation block, never the test set)."""
    from qcclass.entrenamiento.gnn_experimento import cuda_allocator_environment

    cuda_allocator_environment()
    from qcclass.config import ConfigError
    from qcclass.entrenamiento.calibracion import run_calibration
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env
    from qcclass.versionado import VersionError

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    try:
        load_env(dataset.rutas.raiz)
        report, path = run_calibration(dataset, split_config, experimento, device=dispositivo)
    except (ConfigError, ServerConnectionError, VersionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for m in report["memoria"]:
        model = m["modelo"]
        label = f"{model['tipo_capa']} {model['dim']}x{model['capas']}"
        label += f"{' vn' if model['nodo_virtual'] else ''} {model['agregacion']}"
        if "error" in m:
            typer.echo(f"memory {label}: {m['error']}")
        else:
            typer.echo(
                f"memory {label}: {1024 * m['mb_por_nodo']:.1f} KB/node measured, "
                f"{1024 * m['mb_por_nodo_predicho']:.1f} predicted"
            )
    for e in report["estimaciones"]:
        if "error" in e:
            typer.echo(f"{e['nombre']}: {e['error']} (micro-batch {e['microlote_nodos']})")
            continue
        hours = ", ".join(f"{k} ep {v:.1f} h" for k, v in e["horas_por_intento"].items())
        where = "GPU" if e["hilos_cpu"] is None else f"CPU {e['hilos_cpu']} threads"
        typer.echo(
            f"{e['nombre']} ({where}): micro-batch {e['microlote_nodos']}, "
            f"{e['segundos_entrenamiento_epoca']:.1f} s/epoch + {e['segundos_evaluacion']:.1f} s "
            f"per evaluation; GPU {e['pico_asignado_mb']:.0f} MB allocated, "
            f"{e['pico_reservado_mb']:.0f} reserved{' SPILLS' if e['desborda'] else ''}; "
            f"per trial: {hours}"
        )
    typer.echo(f"Report: {path}")


@app.command("comprobar-precision-mixta")
def comprobar_precision_mixta(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    experimento: Annotated[Path, typer.Option(help="GNN experiment configuration.")] = (
        DEFAULT_GNN_CONFIG
    ),
    procesos: Annotated[
        int | None, typer.Option(help="Data-loading processes (default QCCLASS_PROCESOS_CARGA).")
    ] = None,
) -> None:
    """Phase 11: reference configuration on its block with and without mixed precision (CUDA)."""
    from qcclass.config import ConfigError
    from qcclass.entrenamiento.gnn_experimento import (
        check_mixed_precision,
        cuda_allocator_environment,
        load_gnn_experiment,
    )
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env
    from qcclass.versionado import VersionError

    cuda_allocator_environment()
    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    try:
        experiment = load_gnn_experiment(experimento)
        load_env(dataset.rutas.raiz)
        report = check_mixed_precision(dataset, split_config, experiment, workers=procesos)
    except (ConfigError, ServerConnectionError, VersionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for key in ("con", "sin"):
        r = report[key]
        typer.echo(
            f"{key} precision mixta: F1 {100 * r['f1_macro_algoritmo']:.2f} %, "
            f"{r['segundos_epoca_mediana']:.1f} s/epoch, {r['memoria_gpu_mb_max']:.0f} MB"
        )
    verdict = "keep it" if report["mantener_precision_mixta"] else "TURN IT OFF"
    typer.echo(
        f"Difference {100 * report['diferencia_f1_macro_algoritmo']:+.2f} points "
        f"(tolerance {100 * report['tolerancia_f1']:.0f}): {verdict}"
    )


@app.command("diagnosticos-linea-base")
def diagnosticos_linea_base(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    experimento: Annotated[
        Path, typer.Option(help="Diagnostics configuration.")
    ] = DEFAULT_DIAGNOSTICS_CONFIG,
    evaluacion: Annotated[
        Path, typer.Option(help="Evaluation configuration.")
    ] = DEFAULT_EVALUATION_CONFIG,
    partes: Annotated[
        str, typer.Option(help="Parts to run (comma list of barajado, gemelos, confusion, rangos).")
    ] = "barajado,gemelos,confusion,rangos",
    sin_mlflow: Annotated[
        bool, typer.Option("--sin-mlflow", help="Write the files only (log to MLflow later).")
    ] = False,
) -> None:
    """Phase 10 diagnostics of the baseline: shuffled labels, structural twins, confusion,
    levels and range review (cross-validation only, never the test set)."""
    from qcclass.config import ConfigError, load_evaluation_config
    from qcclass.lineabase.configuracion import load_baseline_config
    from qcclass.lineabase.diagnosticos import (
        ShuffledLabelAlarm,
        load_diagnostics_config,
        log_diagnostics,
        run_diagnostics,
    )
    from qcclass.lineabase.ejecutar import build_context
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env
    from qcclass.versionado import VersionError

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    chosen = tuple(_split_list(partes) or ())
    valid = ("barajado", "gemelos", "confusion", "rangos")
    if bad := [p for p in chosen if p not in valid]:
        typer.echo(f"Error: unknown parts {bad} (valid: {list(valid)})", err=True)
        raise typer.Exit(code=1)
    try:
        diagnostics = load_diagnostics_config(experimento)
        baseline = load_baseline_config(diagnostics.baseline_path)
        evaluation = load_evaluation_config(evaluacion)
        load_env(dataset.rutas.raiz)
        ctx = build_context(
            dataset,
            split_config,
            baseline,
            evaluation,
            storage=None,
            experiment=f"{baseline.name}.diagnosticos",
        )
        try:
            result, path = run_diagnostics(ctx, diagnostics, parts=chosen)
        except ShuffledLabelAlarm as alarm:
            typer.echo(f"ALARM: {alarm}", err=True)
            raise typer.Exit(code=2) from alarm
        if not sin_mlflow:
            log_diagnostics(ctx, diagnostics, result, path)
    except (ConfigError, ServerConnectionError, VersionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Diagnostics: {path}")


@app.command("linea-base")
def linea_base(
    config: ConfigOption = DEFAULT_DATASET_CONFIG,
    splits: SplitsOption = DEFAULT_SPLITS_CONFIG,
    experimento: Annotated[
        Path, typer.Option(help="Baseline experiment configuration.")
    ] = DEFAULT_BASELINE_CONFIG,
    evaluacion: Annotated[
        Path, typer.Option(help="Evaluation configuration.")
    ] = DEFAULT_EVALUATION_CONFIG,
    peldanos: Annotated[str | None, typer.Option(help="Only these rungs (comma list).")] = None,
    clasificadores: Annotated[
        str | None, typer.Option(help="Only these classifiers (comma list).")
    ] = None,
    objetivos: Annotated[str | None, typer.Option(help="Only these targets (comma list).")] = None,
    rehacer: Annotated[
        bool, typer.Option("--rehacer", help="Redo the out-of-fold evaluation of finished studies.")
    ] = False,
    sin_resumen: Annotated[
        bool, typer.Option("--sin-resumen", help="Do not write RESUMEN.md.")
    ] = False,
) -> None:
    """Phase 10 baseline: Optuna studies with 5-block cross-validation (never the test set)."""
    from qcclass.config import ConfigError, load_evaluation_config
    from qcclass.lineabase.configuracion import load_baseline_config
    from qcclass.lineabase.ejecutar import run_baseline
    from qcclass.seguimiento.entorno import ServerConnectionError, load_env
    from qcclass.seguimiento.optuna_cliente import connect
    from qcclass.versionado import VersionError

    dataset = _load_dataset_config(config)
    split_config = _load_splits_config(splits, dataset)
    try:
        baseline = load_baseline_config(experimento)
        evaluation = load_evaluation_config(evaluacion)
        load_env(dataset.rutas.raiz)
        storage = connect()
        run = run_baseline(
            dataset,
            split_config,
            baseline,
            evaluation,
            storage=storage,
            rungs=_split_list(peldanos),
            classifiers=_split_list(clasificadores),
            targets=_split_list(objetivos),
            redo=rehacer,
            summary=not sin_resumen,
        )
    except (ConfigError, ServerConnectionError, VersionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for result in run.results:
        typer.echo(
            f"{result['estudio']}: CV macro F1 {100 * result['cv_f1_macro']:.1f} %, "
            f"OOF {100 * result['oof']['f1_macro']['valor']:.1f} %"
        )
    for row in run.finalists.itertuples(index=False):
        typer.echo(f"Finalist {row.peldano}: {row.clasificador} ({100 * row.cv_f1_macro:.1f} %)")
    if run.summary:
        typer.echo(f"Summary: {run.summary}")
