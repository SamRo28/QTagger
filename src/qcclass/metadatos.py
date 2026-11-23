"""Read and update the CSV tables of the dataset (metadata and logs).

Tables are read with every column as text so that values written by other pipeline steps
are preserved exactly when a step rewrites the file.
"""

import io
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from qcclass.circuitos import write_text_atomic

METADATA_COLUMNS = (
    "id",
    "algoritmo",
    "familia",
    "nivel",
    "n_qubits",
    "variante_param",
    "variante_comp",
    "semilla",
    "origen",
    "group_id",
    "ruta_raw",
)


def read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read a CSV table as text; return an empty table with ``columns`` if it does not exist."""
    if not path.is_file():
        return pd.DataFrame(columns=list(columns), dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_table(table: pd.DataFrame, path: Path) -> None:
    """Write a CSV table atomically."""
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    write_text_atomic(buffer.getvalue(), path)


def _as_text(table: pd.DataFrame) -> pd.DataFrame:
    """Convert every value to text, so that rows read from disk and new rows compare equal."""
    return table.fillna("").astype(str)


def _ordered_columns(columns: Sequence[str], *tables: pd.DataFrame) -> list[str]:
    """Known columns first, then any extra column found in the tables, without repeats."""
    names = list(columns)
    for table in tables:
        names.extend(name for name in table.columns if name not in names)
    return names


def upsert(path: Path, new_rows: pd.DataFrame, key: str, columns: Sequence[str]) -> pd.DataFrame:
    """Replace the rows of ``path`` whose ``key`` appears in ``new_rows`` and add the rest.

    Rows written by other steps (other keys) and extra columns are kept. The result is
    sorted by ``key`` so that the file is deterministic.
    """
    current = read_table(path, columns)
    new_rows = _as_text(new_rows)
    kept = current[~current[key].isin(new_rows[key])]
    order = _ordered_columns(columns, current, new_rows)
    merged = pd.concat([kept, new_rows], ignore_index=True).reindex(columns=order)
    merged = merged.sort_values(key, kind="stable").reset_index(drop=True)
    write_table(merged, path)
    return merged


def replace_scope(
    path: Path,
    new_rows: pd.DataFrame,
    scope_columns: Sequence[str],
    scopes: Iterable[tuple[str, ...]],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Drop every row of ``path`` whose ``scope_columns`` match one of ``scopes``, then append.

    Used for logs: when a unit of work is redone, its previous log entries are replaced.
    """
    current = read_table(path, columns)
    new_rows = _as_text(new_rows)
    scope_set = {tuple(str(value) for value in scope) for scope in scopes}
    row_scopes = zip(*(current[name] for name in scope_columns), strict=True)
    in_scope = pd.Series([scope in scope_set for scope in row_scopes], index=current.index)
    kept = current[~in_scope.astype(bool)]
    order = _ordered_columns(columns, current, new_rows)
    merged = pd.concat([kept, new_rows], ignore_index=True).reindex(columns=order)
    merged = merged.sort_values(list(scope_columns), kind="stable").reset_index(drop=True)
    write_table(merged, path)
    return merged
