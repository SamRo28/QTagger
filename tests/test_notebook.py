"""The inspection notebook runs end to end on a small dataset (slow: starts a kernel)."""

from collections.abc import Callable
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from qcclass.config import DatasetConfig
from qcclass.data.deduplicar import deduplicate
from qcclass.data.normalizar import normalize
from qcclass.data.oraculos import generate_oracles
from qcclass.formatos.construir import build_formats

ConfigFactory = Callable[..., DatasetConfig]  # the ``make_config`` fixture


@pytest.mark.slow
def test_inspection_notebook_runs(
    make_config: ConfigFactory, tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(oraculos={"por_tamano": 2})
    generate_oracles(config, ["grover"], range(3, 6))
    normalize(config)
    deduplicate(config)
    build_formats(config, ("qwalk", "hrs_cumulative_multiplier", "hhl"))

    monkeypatch.setenv("QCCLASS_CONFIG", str(tmp_path / "configs" / "dataset.yaml"))
    notebook_dir = repo_root / "notebooks"
    notebook = nbformat.read(notebook_dir / "01_inspeccion.ipynb", as_version=4)
    client = NotebookClient(
        notebook, timeout=300, resources={"metadata": {"path": str(notebook_dir)}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.get("output_type") == "error"]
    images = [o for o in outputs if "image/png" in o.get("data", {})]
    assert len(images) >= 3  # circuit, DAG and qubit graph
