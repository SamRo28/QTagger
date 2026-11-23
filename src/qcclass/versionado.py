"""Dataset versioning and guarded access to the test sets.

Version. ``<rutas.version>`` (``data/VERSION.json``, versioned in git) records the SHA-256
of ``metadata.csv``, of every CSV in ``<splits>/`` and of ``<graphs>/etiquetas.json``, a
global fingerprint of all of them and the versions of the libraries that generated the data
(Qiskit and MQT Bench). Every result produced from the dataset must store
:func:`version_stamp`, which refuses to work if the files no longer match the version, and
every experiment starts with :func:`begin_experiment`, which compares the fingerprints.

Test protection. :func:`split_ids` is the only way to read the identifiers of a split
protocol. Asking for ``test`` without ``evaluar_test=True`` (the ``--evaluar-test`` option of
the commands) raises :class:`LockedTestSetError`; an allowed access appends a line (date,
experiment, dataset version, protocol) to ``<resultados>/accesos_test.log``.
"""

import hashlib
import importlib.metadata
import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from qcclass.circuitos import write_text_atomic
from qcclass.config import DatasetConfig

logger = logging.getLogger(__name__)

LABEL_VOCABULARY_FILE = "etiquetas.json"  # written by the formats step in <graphs>/
TEST_ACCESS_LOG = "accesos_test.log"
TEST_SET = "test"
_CHUNK = 1 << 20
# distributions whose versions determine the generated circuits
GENERATOR_LIBRARIES = ("qiskit", "mqt.bench")


class VersionError(RuntimeError):
    """The dataset version is missing or does not match the files on disk."""


class LockedTestSetError(PermissionError):
    """A test set was requested without the explicit ``--evaluar-test`` option."""


@dataclass(frozen=True)
class DatasetVersion:
    """Contents of ``VERSION.json``."""

    nombre: str
    huella: str
    archivos: dict[str, str]
    creado: str
    git_commit: str
    librerias: dict[str, str]


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def versioned_files(config: DatasetConfig) -> list[Path]:
    """Files covered by the version: metadata, split CSVs and the label vocabulary."""
    files = [config.rutas.metadata]
    files += sorted(config.rutas.splits.glob("*.csv"))
    files.append(config.rutas.graphs / LABEL_VOCABULARY_FILE)
    return files


def _relative(config: DatasetConfig, path: Path) -> str:
    return path.relative_to(config.rutas.raiz).as_posix()


def current_hashes(config: DatasetConfig) -> dict[str, str]:
    """SHA-256 of every versioned file that exists, keyed by its path relative to ``raiz``."""
    return {
        _relative(config, path): file_sha256(path)
        for path in versioned_files(config)
        if path.is_file()
    }


def global_fingerprint(hashes: dict[str, str]) -> str:
    """SHA-256 of the sorted ``path:hash`` lines."""
    lines = "".join(f"{path}:{digest}\n" for path, digest in sorted(hashes.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str:
    """Current git commit of the repository, or an empty string if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def library_versions() -> dict[str, str]:
    """Installed versions of :data:`GENERATOR_LIBRARIES` (``"no instalada"`` if missing)."""
    versions = {}
    for name in GENERATOR_LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "no instalada"
    return versions


def write_version(config: DatasetConfig, name: str) -> DatasetVersion:
    """Hash the versioned files and write ``VERSION.json``.

    Raises:
        VersionError: if any versioned file is missing (the dataset is incomplete).
    """
    if missing := [_relative(config, p) for p in versioned_files(config) if not p.is_file()]:
        raise VersionError(f"cannot version an incomplete dataset, missing: {missing}")
    hashes = current_hashes(config)
    version = DatasetVersion(
        nombre=name,
        huella=global_fingerprint(hashes),
        archivos=hashes,
        creado=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git_commit(config.rutas.raiz),
        librerias=library_versions(),
    )
    text = json.dumps(asdict(version), indent=2, ensure_ascii=False) + "\n"
    write_text_atomic(text, config.rutas.version)
    logger.info("dataset version %s written (%s)", name, version.huella[:12])
    return version


def read_version(config: DatasetConfig) -> DatasetVersion:
    """Read ``VERSION.json``."""
    if not config.rutas.version.is_file():
        raise VersionError(f"no dataset version: {config.rutas.version} does not exist")
    return DatasetVersion(**json.loads(config.rutas.version.read_text(encoding="utf-8")))


def version_differences(config: DatasetConfig) -> list[str]:
    """Differences between ``VERSION.json`` and the files on disk (empty if they match).

    Besides comparing every file hash, the recorded global fingerprint is recomputed from the
    recorded hashes, so a hand-edited ``VERSION.json`` is also reported.
    """
    version = read_version(config)
    recorded = version.archivos
    current = current_hashes(config)
    differences = []
    if global_fingerprint(recorded) != version.huella:
        differences.append("huella global de VERSION.json incoherente con sus archivos")
    differences += [f"falta {path}" for path in sorted(set(recorded) - set(current))]
    differences += [f"nuevo {path}" for path in sorted(set(current) - set(recorded))]
    differences += [
        f"modificado {path}"
        for path in sorted(set(recorded) & set(current))
        if recorded[path] != current[path]
    ]
    return differences


def version_stamp(config: DatasetConfig) -> dict[str, str]:
    """Version fields that every result must store.

    Raises:
        VersionError: if there is no version or the files do not match it.
    """
    version = read_version(config)
    if differences := version_differences(config):
        raise VersionError(f"dataset does not match {version.nombre}: {differences}")
    return {"version_dataset": version.nombre, "huella_dataset": version.huella}


def library_differences(config: DatasetConfig) -> list[str]:
    """Generator libraries whose installed version differs from the one in ``VERSION.json``."""
    recorded = read_version(config).librerias
    installed = library_versions()
    return [
        f"{name}: dataset {recorded.get(name, '?')}, instalada {installed[name]}"
        for name in GENERATOR_LIBRARIES
        if recorded.get(name) != installed[name]
    ]


def begin_experiment(config: DatasetConfig, experiment: str) -> dict[str, str]:
    """Check the dataset at the start of an experiment and return its version stamp.

    Every experiment command calls this before doing anything else: the files must match the
    fingerprints of ``VERSION.json`` (otherwise :class:`VersionError`). A different Qiskit or
    MQT Bench version only logs a warning, because experiments read the stored circuits and
    never regenerate them.
    """
    if not experiment:
        raise ValueError("every experiment must have a name")
    stamp = version_stamp(config)
    for difference in library_differences(config):
        logger.warning("library version differs from the dataset: %s", difference)
    logger.info(
        "experiment %s starts on %s (%s)",
        experiment,
        stamp["version_dataset"],
        stamp["huella_dataset"][:12],
    )
    return stamp


# --------------------------------------------------------------------------------------------
# Guarded access to the splits
# --------------------------------------------------------------------------------------------


def access_log_path(config: DatasetConfig) -> Path:
    """``<resultados>/accesos_test.log``."""
    return config.rutas.resultados / TEST_ACCESS_LOG


def log_test_access(config: DatasetConfig, experiment: str, protocol: str) -> str:
    """Append one access line (date, experiment, dataset version, protocol) and return it."""
    stamp = version_stamp(config)
    line = "\t".join(
        [
            datetime.now(UTC).isoformat(timespec="seconds"),
            experiment,
            stamp["version_dataset"],
            stamp["huella_dataset"][:12],
            protocol,
        ]
    )
    path = access_log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    logger.warning("TEST SET ACCESSED: %s", line)
    return line


def split_ids(
    config: DatasetConfig,
    protocol: str,
    sets: Sequence[str],
    *,
    experiment: str,
    evaluar_test: bool = False,
) -> pd.DataFrame:
    """Rows (``id``, ``conjunto``, ``solo_evaluacion``) of the requested sets of a protocol.

    Raises:
        LockedTestSetError: if ``test`` is requested without ``evaluar_test``.
        VersionError: if the dataset does not match its version.
    """
    if not experiment:
        raise ValueError("every access to the splits must name its experiment")
    stamp_check = version_stamp(config)  # never train or evaluate on unversioned data
    if TEST_SET in sets:
        if not evaluar_test:
            raise LockedTestSetError(
                f"'{experiment}' asked for the test set of '{protocol}' without --evaluar-test"
            )
        log_test_access(config, experiment, protocol)
    path = config.rutas.splits / f"{protocol}.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = table.assign(solo_evaluacion=table["solo_evaluacion"] == "True")
    logger.debug("split %s read for %s (%s)", protocol, experiment, stamp_check["version_dataset"])
    return table[table["conjunto"].isin(list(sets))].reset_index(drop=True)
