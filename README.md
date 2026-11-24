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

### Linux / WSL2 with an NVIDIA GPU

No separate lock is needed: on Linux x86_64 the pinned `torch==2.15.0` wheel from PyPI is the
CUDA 13.2 build, and it pins its own `nvidia-*` / `cuda-toolkit` dependencies. CUDA 13
requires an NVIDIA driver **580 or newer**. On WSL2 the driver is the Windows one: update it
on Windows and do not install any NVIDIA driver inside Linux.

```bash
nvidia-smi                       # inside WSL2: driver >= 580, "CUDA Version: 13.x"
# Python 3.13 (any distribution) with uv; or apt install python3.13 python3.13-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
# keep the repository and the data on the Linux file system (~/), not on /mnt/c: reading
# thousands of small graph files through /mnt/c is much slower
git clone git@github.com:SamRo28/QTagger.git ~/QTagger && cd ~/QTagger
make install PYTHON_BOOTSTRAP="$(uv python find 3.13)"
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The dataset is not in git. Copy it from the machine that generated it instead of
regenerating it (floating-point differences between platforms could change the files and
therefore the fingerprint):

```bash
# on the machine with the data
COPYFILE_DISABLE=1 tar czf data.tgz data/metadata.csv data/splits data/graphs/dag data/graphs/etiquetas.json
# on the target machine, in the repository root
tar xzf data.tgz && qcclass verificar
```

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
`normalizar`, `deduplicar`, `formatos`, `particiones`, `calidad`, `versionar`, `verificar`.

`make clean-data` deletes every generated file under `data/`.

The dataset version is recorded in `data/VERSION.json` (git tag `dataset-v1`), together with
the Qiskit and MQT Bench versions that generated it, and checked with
`qcclass versionar --comprobar`; every experiment compares the fingerprints when it starts. Test sets can only be read with `--evaluar-test`; every
access is logged in `results/accesos_test.log`.

## Measuring training cost

`qcclass medir` trains a reference GNN (GINEConv, `src/qcclass/modelos/referencia.py`) on the
train set of the `aleatorio` protocol only, with batches limited by a node budget, and writes
`results/medicion/<machine>_<device>[_amp].json` (machine description, worst-case pass, per
epoch timings, throughput, loss, memory, temperature and a projection of training and Optuna
hours). `results/medicion/` is versioned in git.

```bash
qcclass verificar                                    # data fingerprints vs data/VERSION.json
qcclass medir --dispositivo mps --equipo m3pro       # --amp, --barrido, --max-lotes, --num-workers...
qcclass medir --resumen                              # results/medicion/RESUMEN.md
```

`notebooks/medicion_colab.ipynb` runs the measurement on a Google Colab T4 (required, plain and
with `--amp`); it falls back to torch 2.14.1 + CUDA 12.6 if Colab's driver is older than 580.

## Development

```bash
make lint       # ruff check .
make test       # pytest, skipping slow tests
make test-all   # pytest, including slow tests (notebook, whole pipeline)
```
