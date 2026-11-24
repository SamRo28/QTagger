"""``RESUMEN.md``: comparison table of every measurement JSON (``qcclass medir --resumen``)."""

import json
import logging
from pathlib import Path

from qcclass.circuitos import write_text_atomic

logger = logging.getLogger(__name__)

COLUMNS = (
    "Equipo",
    "Dispositivo",
    "GPU",
    "Presupuesto",
    "Workers",
    "Lotes/época",
    "s/época",
    "Espera datos",
    "Nodos/s",
    "Grafos/s",
    "Pérdida 1→N",
    "Mem. GPU máx. (MB)",
    "Útil al inicio (MB)",
    "RAM máx. (MB)",
    "Temp. GPU máx./media",
    "Peor caso",
    "Mayor presupuesto (barrido)",
    "100 épocas (h)",
    "Optuna 20 / 50 (h)",
)


def _value(value: object, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        value = f"{value:,}"
    elif isinstance(value, float):
        if abs(value) >= 1000:
            value = f"{value:,.0f}"
        else:
            value = f"{value:,.1f}" if abs(value) >= 10 else f"{value:.3g}"
    return f"{value}{unit}"


def summary_row(result: dict) -> list[str]:
    """One table row of a measurement result."""
    options = result["opciones"]
    epochs = result["epocas"]
    last = epochs[-1] if epochs else {}
    first = epochs[0] if epochs else {}
    projection = result.get("proyeccion") or {}
    sweep = result.get("barrido") or {}
    worst = result["peor_caso"]
    device = result["dispositivo"] + (" + AMP" if options["amp"] else "")
    temperature = (
        f"{_value(last.get('temperatura_gpu_max'))} / {_value(last.get('temperatura_gpu_media'))}"
        if last.get("temperatura_gpu_max") is not None
        else "—"
    )
    losses = f"{first['perdida_media']:.3f} → {last['perdida_media']:.3f}" if epochs else "—"
    optuna = projection.get("optuna_h", {})
    return [
        result["equipo"]["nombre"],
        device,
        _value(result["equipo"].get("gpu")),
        f"{options['presupuesto_nodos']:,}",
        str(options["num_workers"]),
        _value(last.get("lotes")),
        _value(last.get("tiempo_total_s")),
        _value(
            None
            if last.get("fraccion_espera_datos") is None
            else round(100 * last["fraccion_espera_datos"], 1),
            " %",
        ),
        _value(last.get("nodos_s")),
        _value(last.get("grafos_s")),
        losses,
        _value(last.get("memoria_gpu_max_mb")),
        _value((result.get("memoria_inicial") or {}).get("util_mb")),
        _value(last.get("ram_max_mb")),
        temperature,
        f"{worst['n_nodos']:,} nodos, " + ("ok" if worst["ok"] else "sin memoria"),
        _value(sweep.get("mayor_presupuesto_ok")),
        _value(projection.get("entrenamiento_completo_h", {}).get("100")),
        f"{_value(optuna.get('20'))} / {_value(optuna.get('50'))}" if optuna else "—",
    ]


def _sweep_lines(result: dict) -> list[str]:
    sweep = result.get("barrido")
    if not sweep:
        return []
    name = f"{result['equipo']['nombre']} ({result['dispositivo']})"
    lines = [f"**{name}**", "", "| Presupuesto | Estado | Nodos/s | Mem. GPU máx. (MB) |"]
    lines.append("|---|---|---|---|")
    for item in sweep["presupuestos"]:
        state = "ok" if item["ok"] else "sin memoria"
        lines.append(
            f"| {item['presupuesto_nodos']:,} | {state} | {_value(item.get('nodos_s'))} | "
            f"{_value(item.get('memoria_gpu_max_mb'))} |"
        )
    return [*lines, ""]


def write_summary(directory: Path, output_name: str) -> Path:
    """Gather every ``*.json`` of ``directory`` into ``<directory>/<output_name>``.

    Raises:
        FileNotFoundError: if there is no measurement in ``directory``.
    """
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            logger.warning("%s is not a measurement result: skipped", path)
    if not results:
        raise FileNotFoundError(f"no measurement JSON in {directory}")
    lines = [
        "# Medición de rendimiento",
        "",
        "Generado por `qcclass medir --resumen` a partir de los JSON de esta carpeta. Modelo de "
        "referencia (GINEConv), protocolo `aleatorio`, conjunto train. Las cifras de época "
        "son las de la última época medida (caché de disco caliente si hay dos o más).",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    lines += ["| " + " | ".join(summary_row(r)) + " |" for r in results]
    versions = sorted({f"{r['version_dataset']} ({r['huella_dataset'][:12]})" for r in results})
    lines += ["", f"Versión del dataset: {', '.join(versions)}.", ""]
    sweeps = [line for r in results for line in _sweep_lines(r)]
    if sweeps:
        lines += ["## Barrido de presupuestos", "", *sweeps]
    path = Path(directory) / output_name
    write_text_atomic("\n".join(lines).rstrip() + "\n", path)
    return path
