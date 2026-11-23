"""Smoke tests for the command-line interface."""

from typer.testing import CliRunner

from qcclass.cli import app

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


def test_help_lists_all_subcommands() -> None:
    """``qcclass --help`` must list every pipeline subcommand."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in SUBCOMMANDS:
        assert command in result.output
