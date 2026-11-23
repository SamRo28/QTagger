"""Smoke tests for the command-line interface."""

import pytest
from typer.testing import CliRunner

from qcclass.cli import NOT_IMPLEMENTED_MESSAGE, app

SUBCOMMANDS = [
    "generar",
    "oraculos",
    "normalizar",
    "deduplicar",
    "formatos",
    "particiones",
    "calidad",
    "todo",
]

IMPLEMENTED = {"generar", "oraculos", "normalizar", "deduplicar", "formatos", "particiones"}
STUBS = [command for command in SUBCOMMANDS if command not in IMPLEMENTED]

runner = CliRunner()


def test_help_lists_all_subcommands() -> None:
    """``qcclass --help`` must list every pipeline subcommand."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in SUBCOMMANDS:
        assert command in result.output


@pytest.mark.parametrize("command", STUBS)
def test_stub_subcommands_report_not_implemented(command: str) -> None:
    """Placeholder subcommands exit cleanly and say they are not implemented."""
    result = runner.invoke(app, [command])
    assert result.exit_code == 0
    assert NOT_IMPLEMENTED_MESSAGE in result.output
