# qcclass — quantum circuit classification by algorithm

Goal: train a graph neural network that, given a quantum circuit **flattened to basic gates**
(`rz`, `sx`, `x`, `cx`), identifies which algorithm it implements (Grover, QFT, QAOA,
Deutsch-Jozsa, ...).

This first stage of the repository covers the project structure and the **generation of the
dataset** in all its formats:

| Format | Content | Location |
|---|---|---|
| F1 | Raw OpenQASM 3 circuits from MQT Bench and in-house oracles | `data/raw/` |
| F2 | Flattened, normalised, deduplicated circuits | `data/canonical/` |
| F3 | Global feature vector per circuit | `data/features/` |
| F4 | Gate dependency DAG (PyTorch Geometric) | `data/graphs/dag/` |
| F5 | Qubit interaction graph (PyTorch Geometric) | `data/graphs/cubits/` |
| F6 | Gate token sequences | `data/tokens/` |
| F7 | Per-gate block labels | `data/gate_labels/` |

## Installation

Requires Python 3.13.

```bash
make install    # creates .venv and runs: pip install -c requirements.lock -e ".[dev]"
source .venv/bin/activate
```

`requirements.lock` pins every dependency version; regenerate it with `make lock` after changing
dependencies in `pyproject.toml`.

## Regenerating the dataset

The whole dataset is generated from the `mqt.bench` package and the configuration in `configs/`;
nothing is downloaded and there are no manual steps.

```bash
make dataset        # runs: qcclass todo (generar -> ... -> calidad)
make dataset-mini   # same pipeline on configs/dataset_mini.yaml (4 algorithms, 3-8 qubits),
                    # written to data_mini/ and results_mini/
```

Reports are written to `results/`: `informe_deduplicacion.md`, `informe_particiones.md` and
`informe_dataset.md` (quality report with a coherence check between `metadata.csv` and the
files on disk). Decisions that refine `PLAN.md` are logged in `docs/DECISIONES.md`.

Individual steps are available as subcommands (`qcclass --help`): `generar`, `oraculos`,
`normalizar`, `deduplicar`, `formatos`, `particiones`, `calidad`.

`make clean-data` deletes every generated file under `data/`.

## Development

```bash
make lint       # ruff check .
make test       # pytest, skipping slow tests
make test-all   # pytest, including slow tests (notebook, whole pipeline)
```
