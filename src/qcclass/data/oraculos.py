"""In-house oracle circuit generators (dj, bv, simon, grover) built from standard Qiskit gates.

MQT Bench provides a single oracle per size for these algorithms. This module generates
many distinct oracles per size. Every circuit is assembled from named sub-circuits
(``hadamard_inicial``, ``oraculo``, ``difusor``, ``hadamard_final``) appended as
instructions, so that the per-gate labels (F7) can recover the block each gate comes from.

Qubit ``i`` holds bit ``i`` of an integer (Qiskit's little-endian convention); bit strings
in oracle identifiers are written with qubit 0 first.

Sizes (``n`` is always the total number of qubits, ancillas included):

* ``dj``: ``n - 1`` inputs and one output, except non-linear oracles with a product of
  three variables, whose multi-controlled X may take one clean ancilla from the inputs.
* ``bv``: ``n - 1`` input qubits and one output ancilla.
* ``simon``: ``n / 2`` input and ``n / 2`` output qubits (``n`` even).
* ``grover``: ``k`` search qubits plus the clean ancillas of its multi-controlled X, built
  as a v-chain, with Maslov's construction (m15) or with Barenco's (b95), spread evenly
  over the constructions that fit each size, so that neither the width nor a CCX ladder
  gives Grover away.
"""

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path

import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.synthesis import synth_mcx_1_clean_b95, synth_mcx_n_clean_m15

from qcclass.circuitos import canonical_gate_count, qpy_path, write_circuit
from qcclass.config import DatasetConfig
from qcclass.metadatos import METADATA_COLUMNS, read_table, replace_scope, upsert
from qcclass.paralelo import run_tasks
from qcclass.semillas import derive_seed

logger = logging.getLogger(__name__)

ORACLE_ALGORITHMS = ("dj", "bv", "simon", "grover")
ORIGIN = "propio"
LEVEL = "alg"
ERRORS_FILE_NAME = "oraculos_errores.csv"
ERROR_COLUMNS = ("algoritmo", "n", "semilla", "tipo", "mensaje")
ERROR_SCOPE = ("algoritmo", "n")
ERROR_INVALID_SIZE = "tamano_invalido"
ERROR_TOO_LARGE = "demasiado_grande"
ERROR_SKIPPED_SIZE = "omitido_por_tamano"
ERROR_EXPORT = "exportacion"
MAX_ATTEMPTS_PER_ORACLE = 25  # random draws per requested oracle before giving up

# Block names (instruction names in the generated circuits).
BLOCK_INITIAL = "hadamard_inicial"
BLOCK_ORACLE = "oraculo"
BLOCK_DIFFUSER = "difusor"
BLOCK_FINAL = "hadamard_final"
BLOCK_MCX_VCHAIN = "mcx_vchain"
BLOCK_MCX_M15 = "mcx_m15"
BLOCK_MCX_B95 = "mcx_b95"

# Multi-controlled X constructions used by Grover (``construccion_mcx`` metadata column).
MCX_VCHAIN = "v-chain"  # Toffoli ladder, k - 2 clean ancillas
MCX_M15 = "m15"  # Maslov 2016 (qiskit.synthesis.synth_mcx_n_clean_m15), ceil((k - 2) / 2)
MCX_B95 = "b95"  # Barenco 1995 (qiskit.synthesis.synth_mcx_1_clean_b95), 0 or 1 ancilla
MCX_CONSTRUCTIONS = (MCX_VCHAIN, MCX_M15, MCX_B95)
MCX_DIRECT = "directa"  # <= 2 controls: a plain CX / CCX, every construction coincides

# Deutsch-Jozsa oracle types (``tipo_oraculo_dj`` metadata column).
DJ_CONSTANT = "constante"
DJ_LINEAR = "lineal"
DJ_NONLINEAR = "no_lineal"
DJ_NONLINEAR_SHARE = 0.7  # of the balanced oracles


class InvalidSizeError(ValueError):
    """The algorithm has no circuit with this number of qubits."""


@dataclass(frozen=True)
class OracleCircuit:
    """A generated circuit and the identifier of its oracle."""

    circuit: QuantumCircuit
    oracle_id: str
    mcx_construction: str = ""
    dj_kind: str = ""


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------


def _bits(values: Sequence[int]) -> str:
    """Bit string of 0/1 values, qubit 0 first."""
    return "".join(str(value) for value in values)


def _random_bits(rng: random.Random, length: int, *, nonzero: bool = False) -> list[int]:
    """Uniform random bit list, optionally excluding the all-zero one."""
    while True:
        bits = [rng.randint(0, 1) for _ in range(length)]
        if not nonzero or any(bits):
            return bits


def _block(name: str, num_qubits: int, build: Callable[[QuantumCircuit], None]) -> Gate:
    """Build a named sub-circuit and return it as a gate."""
    block = QuantumCircuit(num_qubits, name=name)
    build(block)
    return block.to_gate()


def _with_measured_register(n: int, measured: int) -> QuantumCircuit:
    """Empty circuit with ``n`` qubits and a classical register for the first ``measured``."""
    return QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(measured, "c"))


def _mcx_vchain(num_controls: int) -> Gate:
    """Multi-controlled X with the classic v-chain of Toffolis on clean ancillas.

    Qubit order: controls, target, ``num_controls - 2`` ancillas (which must start and end
    in |0>). Equivalent to the deprecated ``QuantumCircuit.mcx(..., mode="v-chain")``.
    """
    ancillas = max(0, num_controls - 2)

    def build(block: QuantumCircuit) -> None:
        controls = list(range(num_controls))
        target = num_controls
        aux = list(range(num_controls + 1, num_controls + 1 + ancillas))
        if num_controls == 1:
            block.cx(controls[0], target)
            return
        if num_controls == 2:
            block.ccx(controls[0], controls[1], target)
            return
        chain = [(controls[0], controls[1], aux[0])]
        chain += [(controls[i], aux[i - 2], aux[i - 1]) for i in range(2, num_controls - 1)]
        for c1, c2, t in chain:
            block.ccx(c1, c2, t)
        block.ccx(controls[-1], aux[-1], target)
        for c1, c2, t in reversed(chain):
            block.ccx(c1, c2, t)

    return _block(BLOCK_MCX_VCHAIN, num_controls + 1 + ancillas, build)


@cache
def _synthesised_mcx(construction: str, num_controls: int) -> QuantumCircuit:
    """Qiskit synthesis of a multi-controlled X (qubits: controls, target, clean ancillas)."""
    synthesis = {MCX_M15: synth_mcx_n_clean_m15, MCX_B95: synth_mcx_1_clean_b95}[construction]
    circuit = synthesis(num_controls)
    circuit.name = {MCX_M15: BLOCK_MCX_M15, MCX_B95: BLOCK_MCX_B95}[construction]
    return circuit


def _mcx(construction: str, num_controls: int) -> Gate:
    """Multi-controlled X block of the requested construction.

    * ``v-chain``: Toffoli ladder on ``k - 2`` clean ancillas (built here with CCX).
    * ``m15``: ``qiskit.synthesis.synth_mcx_n_clean_m15`` (Maslov 2016), relative-phase
      Toffolis on ``ceil((k - 2) / 2)`` clean ancillas.
    * ``b95``: ``qiskit.synthesis.synth_mcx_1_clean_b95`` (Barenco 1995 / Iten 2016), one
      clean ancilla for ``k >= 5`` and none for ``k <= 4``.
    """
    if construction == MCX_VCHAIN:
        return _mcx_vchain(num_controls)
    return _synthesised_mcx(construction, num_controls).to_gate()


def mcx_ancillas(construction: str, num_controls: int) -> int:
    """Clean ancillas used by a multi-controlled X construction (read from the synthesis)."""
    if num_controls <= 2:
        return 0
    if construction == MCX_VCHAIN:
        return num_controls - 2
    return _synthesised_mcx(construction, num_controls).num_qubits - num_controls - 1


def construction_plan(
    config: DatasetConfig, algorithm: str, n: int, count: int, options: Sequence[str]
) -> list[str]:
    """Balanced, seeded assignment of MCX constructions to the ``count`` oracles of a size."""
    ordered = sorted(options)
    plan = [ordered[i % len(ordered)] for i in range(count)]
    seed = derive_seed(config.semilla_global, "construccion_mcx", algorithm, n)
    random.Random(seed).shuffle(plan)
    return plan


def _apply_mcz(block: QuantumCircuit, search: int, construction: str) -> None:
    """Phase flip of |1...1> on the first ``search`` qubits (H - MCX - H on the last one)."""
    target = search - 1
    block.h(target)
    if search == 2:
        block.cx(0, 1)
    else:
        controls = list(range(search - 1))
        ancillas = mcx_ancillas(construction, search - 1)
        block.append(
            _mcx(construction, search - 1), [*controls, target, *range(search, search + ancillas)]
        )
    block.h(target)


# --------------------------------------------------------------------------------------------
# Deutsch-Jozsa
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _DjOracle:
    """A Deutsch-Jozsa oracle on ``inputs`` input qubits, one output and ``ancillas`` ancillas."""

    kind: str
    oracle_id: str
    inputs: int
    ancillas: int
    construction: str
    build: Callable[[QuantumCircuit], None]


def _dj_constant(rng: random.Random, n: int) -> _DjOracle:
    """f(x) = 0 or 1."""
    value = rng.randint(0, 1)

    def build(block: QuantumCircuit) -> None:
        if value:
            block.x(n - 1)

    return _DjOracle(DJ_CONSTANT, f"const{value}", n - 1, 0, "", build)


def _dj_linear(rng: random.Random, n: int) -> _DjOracle:
    """Balanced parity of a random non-empty subset ``S``, with X around a random ``b``."""
    inputs = n - 1
    subset = _random_bits(rng, inputs, nonzero=True)
    flips = [bit & rng.randint(0, 1) for bit in subset]

    def build(block: QuantumCircuit) -> None:
        wrapped = [i for i, bit in enumerate(flips) if bit]
        if wrapped:
            block.x(wrapped)
        for i, bit in enumerate(subset):
            if bit:
                block.cx(i, inputs)
        if wrapped:
            block.x(wrapped)

    return _DjOracle(DJ_LINEAR, f"bal_S{_bits(subset)}_b{_bits(flips)}", inputs, 0, "", build)


def _dj_terms(rng: random.Random, pivot: int, inputs: int, max_size: int) -> list[tuple[int, ...]]:
    """1 to 3 distinct random products of 2 to ``max_size`` inputs other than ``pivot``."""
    rest = [i for i in range(inputs) if i != pivot]
    terms: set[tuple[int, ...]] = set()
    for _ in range(rng.randint(1, 3)):
        size = rng.randint(2, min(max_size, len(rest)))
        terms.add(tuple(sorted(rng.sample(rest, size))))
    return sorted(terms)


def _dj_nonlinear(rng: random.Random, n: int, construction: str) -> _DjOracle | None:
    """Balanced non-linear ``f(x) = x_i XOR g(rest)``; ``None`` if ``n`` is too small.

    ``g`` is the XOR of 1 to 3 products of 2 or 3 variables. ``f`` is balanced because for
    any value of the other inputs, flipping ``x_i`` flips ``f``. Products of two variables
    are CCX gates; products of three use a multi-controlled X of the given construction,
    whose clean ancillas (if any) widen the circuit.
    """
    three_ancillas = mcx_ancillas(construction, 3)
    max_size = 3 if rng.random() < 0.5 else 2
    inputs = n - 1 - (three_ancillas if max_size == 3 else 0)
    if max_size == 3 and inputs < 4:  # x_i plus at least three other inputs
        max_size, inputs = 2, n - 1
    if inputs < 3:  # x_i plus at least two other inputs
        return None
    pivot = rng.randrange(inputs)
    terms = _dj_terms(rng, pivot, inputs, max_size)
    uses_mcx = any(len(term) == 3 for term in terms)
    ancillas = n - 1 - inputs
    if not uses_mcx and ancillas:  # no product of three: the ancillas are not needed
        inputs, ancillas = n - 1, 0
        pivot = min(pivot, inputs - 1)
    output = inputs

    def build(block: QuantumCircuit) -> None:
        block.cx(pivot, output)
        for term in terms:
            if len(term) == 2:
                block.ccx(*term, output)
            else:
                aux = range(output + 1, output + 1 + ancillas)
                block.append(_mcx(construction, 3), [*term, output, *aux])

    terms_id = "+".join(".".join(str(i) for i in term) for term in terms)
    used = construction if uses_mcx else MCX_DIRECT
    oracle_id = f"nl_k{inputs}_i{pivot}_g{terms_id}"
    return _DjOracle(DJ_NONLINEAR, oracle_id, inputs, ancillas, used, build)


def _dj_choose(rng: random.Random, n: int, construction: str) -> _DjOracle:
    """50 % constant; balanced ones are 70 % non-linear and 30 % linear (parities)."""
    if rng.random() < 0.5:
        return _dj_constant(rng, n)
    if rng.random() < DJ_NONLINEAR_SHARE:
        nonlinear = _dj_nonlinear(rng, n, construction)
        if nonlinear is not None:
            return nonlinear
    return _dj_linear(rng, n)


def deutsch_jozsa(n: int, rng: random.Random, construction: str | None = None) -> OracleCircuit:
    """Deutsch-Jozsa with a random constant, linear or non-linear balanced oracle.

    Qubits: inputs, then the output (prepared in |->), then the clean ancillas of the
    multi-controlled X gates of non-linear oracles (if any).
    """
    if n < 2:
        raise InvalidSizeError("dj needs at least 2 qubits")
    construction = construction or rng.choice(MCX_CONSTRUCTIONS)
    oracle = _dj_choose(rng, n, construction)
    inputs, kickback = oracle.inputs, oracle.inputs + 1
    circuit = _with_measured_register(n, inputs)
    circuit.append(_block(BLOCK_INITIAL, kickback, _prepare_phase_kickback), range(kickback))
    circuit.append(_block(BLOCK_ORACLE, n, oracle.build), range(n))
    circuit.append(_block(BLOCK_FINAL, inputs, lambda b: b.h(range(inputs))), range(inputs))
    circuit.measure(range(inputs), range(inputs))
    return OracleCircuit(
        circuit,
        f"dj_n{n}_{oracle.oracle_id}",
        mcx_construction=oracle.construction,
        dj_kind=oracle.kind,
    )


def _prepare_phase_kickback(block: QuantumCircuit) -> None:
    """Inputs in |+>, last qubit (output ancilla) in |->."""
    block.x(block.num_qubits - 1)
    block.h(range(block.num_qubits))


# --------------------------------------------------------------------------------------------
# Bernstein-Vazirani
# --------------------------------------------------------------------------------------------


def bernstein_vazirani(n: int, rng: random.Random) -> OracleCircuit:
    """Bernstein-Vazirani on ``n - 1`` inputs with a random non-zero secret string."""
    if n < 2:
        raise InvalidSizeError("bv needs at least 2 qubits")
    inputs = n - 1
    secret = _random_bits(rng, inputs, nonzero=True)

    def oracle(block: QuantumCircuit) -> None:
        for i, bit in enumerate(secret):
            if bit:
                block.cx(i, inputs)

    circuit = _with_measured_register(n, inputs)
    circuit.append(_block(BLOCK_INITIAL, n, _prepare_phase_kickback), range(n))
    circuit.append(_block(BLOCK_ORACLE, n, oracle), range(n))
    circuit.append(_block(BLOCK_FINAL, inputs, lambda b: b.h(range(inputs))), range(inputs))
    circuit.measure(range(inputs), range(inputs))
    return OracleCircuit(circuit, f"bv_n{n}_s{_bits(secret)}")


# --------------------------------------------------------------------------------------------
# Simon
# --------------------------------------------------------------------------------------------


def simon(n: int, rng: random.Random) -> OracleCircuit:
    """Simon's algorithm with a random 2-to-1 oracle ``f(x) = f(x XOR s)``.

    Standard construction: copy ``x`` into the output register and, for every ``i`` with
    ``s_i = 1``, XOR ``x_j`` (``j`` = first set bit of ``s``) into output ``i``. The outputs
    are then relabelled by a random permutation and a random X mask, which keeps the
    function 2-to-1 with the same period ``s``.
    """
    if n < 2 or n % 2:
        raise InvalidSizeError("simon needs an even number of qubits")
    k = n // 2
    secret = _random_bits(rng, k, nonzero=True)
    permutation = rng.sample(range(k), k)
    mask = _random_bits(rng, k)
    pivot = secret.index(1)

    def oracle(block: QuantumCircuit) -> None:
        for i in range(k):
            block.cx(i, k + permutation[i])
        for i, bit in enumerate(secret):
            if bit:
                block.cx(pivot, k + permutation[i])
        masked = [k + i for i, bit in enumerate(mask) if bit]
        if masked:
            block.x(masked)

    circuit = _with_measured_register(n, k)
    circuit.append(_block(BLOCK_INITIAL, k, lambda b: b.h(range(k))), range(k))
    circuit.append(_block(BLOCK_ORACLE, n, oracle), range(n))
    circuit.append(_block(BLOCK_FINAL, k, lambda b: b.h(range(k))), range(k))
    circuit.measure(range(k), range(k))
    oracle_id = f"simon_n{n}_s{_bits(secret)}_p{_bits(permutation)}_m{_bits(mask)}"
    return OracleCircuit(circuit, oracle_id)


# --------------------------------------------------------------------------------------------
# Grover
# --------------------------------------------------------------------------------------------


def grover_width(search: int, construction: str) -> int:
    """Total qubits of a Grover circuit with ``search`` search qubits."""
    return search + mcx_ancillas(construction, search - 1)


def grover_constructions(n: int) -> dict[str, int]:
    """MCX constructions that fit exactly ``n`` qubits, with their number of search qubits."""
    options = {}
    for construction in MCX_CONSTRUCTIONS:
        for search in range(2, n + 1):
            if grover_width(search, construction) == n:
                options[construction] = search
    return options


def grover_iterations(search: int, marked: int) -> int:
    """Optimal number of Grover iterations for ``marked`` solutions out of ``2**search``."""
    theta = math.asin(math.sqrt(marked / 2**search))
    return max(1, round(math.pi / (4 * theta) - 0.5))


def grover(n: int, rng: random.Random, construction: str | None = None) -> OracleCircuit:
    """Grover search with a phase oracle marking 1 to ``max(1, N // 8)`` random states.

    The oracle flips the phase of each marked state with X gates and a multi-controlled Z
    (H - MCX - H); the diffuser is the standard inversion about the mean. The MCX uses the
    requested construction if it fits ``n`` qubits, otherwise the one that does (chosen
    with ``rng`` when both fit and none is requested).
    """
    options = grover_constructions(n)
    if not options:
        raise InvalidSizeError(f"no grover circuit has {n} qubits")
    if construction not in options:
        construction = rng.choice(sorted(options)) if len(options) > 1 else next(iter(options))
    search = options[construction]
    states = 2**search
    marked = sorted(rng.sample(range(states), rng.randint(1, max(1, states // 8))))

    def oracle(block: QuantumCircuit) -> None:
        for state in marked:
            zeros = [i for i in range(search) if not state >> i & 1]
            if zeros:
                block.x(zeros)
            _apply_mcz(block, search, construction)
            if zeros:
                block.x(zeros)

    def diffuser(block: QuantumCircuit) -> None:
        block.h(range(search))
        block.x(range(search))
        _apply_mcz(block, search, construction)
        block.x(range(search))
        block.h(range(search))

    circuit = _with_measured_register(n, search)
    circuit.append(_block(BLOCK_INITIAL, search, lambda b: b.h(range(search))), range(search))
    oracle_gate = _block(BLOCK_ORACLE, n, oracle)
    diffuser_gate = _block(BLOCK_DIFFUSER, n, diffuser)
    for _ in range(grover_iterations(search, len(marked))):
        circuit.append(oracle_gate, range(n))
        circuit.append(diffuser_gate, range(n))
    circuit.measure(range(search), range(search))
    states_id = "-".join(_bits([s >> i & 1 for i in range(search)]) for s in marked)
    used = construction if search - 1 > 2 else MCX_DIRECT
    return OracleCircuit(circuit, f"grover_n{n}_m{states_id}", mcx_construction=used)


BUILDERS: dict[str, Callable[[int, random.Random], OracleCircuit]] = {
    "dj": deutsch_jozsa,
    "bv": bernstein_vazirani,
    "simon": simon,
    "grover": grover,
}


# --------------------------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleTask:
    """All sizes of one oracle algorithm."""

    algorithm: str
    family: str
    sizes: tuple[int, ...]
    per_size: int
    existing_ids: frozenset[str]
    force: bool


@dataclass
class OracleResult:
    """Metadata and error rows of one task."""

    rows: list[dict[str, object]] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)
    skipped: int = 0


@dataclass(frozen=True)
class OracleReport:
    """Outcome of an oracle generation run."""

    rows: pd.DataFrame
    errors: pd.DataFrame
    skipped: int


def oracle_group_id(algorithm: str, n: int, seed: int) -> str:
    """``<algorithm>_n<n>_o<seed>``: base-circuit identifier of an oracle circuit."""
    return f"{algorithm}_n{n}_o{seed}"


def oracle_path(config: DatasetConfig, family: str, algorithm: str, group: str) -> Path:
    """``<raw>/alg/<family>/<algorithm>/<group_id>.qasm``."""
    return config.rutas.raw / LEVEL / family / algorithm / f"{group}.qasm"


def errors_path(config: DatasetConfig) -> Path:
    """Location of the oracle error log."""
    return config.rutas.logs / ERRORS_FILE_NAME


def _error(algorithm: str, n: int, seed: int | str, kind: str, message: str) -> dict[str, object]:
    """Error-log row."""
    return {"algoritmo": algorithm, "n": n, "semilla": seed, "tipo": kind, "mensaje": message}


def _metadata_row(
    config: DatasetConfig, task: OracleTask, n: int, seed: int, generated: OracleCircuit
) -> dict[str, object]:
    """Metadata row of a written oracle circuit."""
    group = oracle_group_id(task.algorithm, n, seed)
    path = oracle_path(config, task.family, task.algorithm, group)
    return {
        "id": f"{LEVEL}_{group}",
        "algoritmo": task.algorithm,
        "familia": task.family,
        "nivel": LEVEL,
        "n_qubits": generated.circuit.num_qubits,
        "variante_param": "",
        "variante_comp": "",
        "semilla": seed,
        "origen": ORIGIN,
        "group_id": group,
        "ruta_raw": path.relative_to(config.rutas.raiz).as_posix(),
        "oraculo_id": generated.oracle_id,
        "construccion_mcx": generated.mcx_construction,
        "tipo_oraculo_dj": generated.dj_kind,
    }


def _store(
    config: DatasetConfig, task: OracleTask, n: int, seed: int, generated: OracleCircuit
) -> dict[str, object] | None:
    """Write one circuit unless it is already present; return its metadata row or None."""
    row = _metadata_row(config, task, n, seed, generated)
    path = config.rutas.raiz / str(row["ruta_raw"])
    present = row["id"] in task.existing_ids and path.is_file() and qpy_path(path).is_file()
    if present and not task.force:
        return None
    write_circuit(generated.circuit, path)
    return row


def _generate_size(config: DatasetConfig, task: OracleTask, n: int, result: OracleResult) -> int:
    """Generate up to ``per_size`` distinct oracles of size ``n``; return how many fit.

    Candidates are drawn with seeds derived from the configuration; repeated oracles and
    circuits above ``max_nodos`` (estimated with a plain canonical translation) are skipped.
    """
    options = grover_constructions(n) if task.algorithm == "grover" else MCX_CONSTRUCTIONS
    plan = construction_plan(config, task.algorithm, n, task.per_size, options)
    seen: set[str] = set()
    fitting = 0
    for attempt in range(task.per_size * MAX_ATTEMPTS_PER_ORACLE):
        if len(seen) >= task.per_size:
            break
        seed = derive_seed(config.semilla_global, "oraculo", task.algorithm, n, attempt)
        builder = BUILDERS[task.algorithm]
        if task.algorithm in ("grover", "dj"):
            builder = partial(builder, construction=plan[len(seen)])
        try:
            generated = builder(n, random.Random(seed))
        except InvalidSizeError as exc:
            result.errors.append(_error(task.algorithm, n, "", ERROR_INVALID_SIZE, str(exc)))
            return 0
        if generated.oracle_id in seen:
            continue
        seen.add(generated.oracle_id)
        if (gates := canonical_gate_count(generated.circuit)) > config.max_nodos:
            message = f"{gates} canonical gates > max_nodos={config.max_nodos}"
            result.errors.append(_error(task.algorithm, n, seed, ERROR_TOO_LARGE, message))
            continue
        fitting += 1
        try:
            row = _store(config, task, n, seed, generated)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            result.errors.append(_error(task.algorithm, n, seed, ERROR_EXPORT, message))
            continue
        if row is None:
            result.skipped += 1
        else:
            result.rows.append(row)
    return fitting


def run_oracle_task(task: OracleTask, config: DatasetConfig) -> OracleResult:
    """Generate every size of a task, stopping after a valid size where nothing fits."""
    result = OracleResult()
    stopped_at: int | None = None
    for n in task.sizes:
        if stopped_at is not None:
            message = f"no oracle of n={stopped_at} fits in max_nodos"
            result.errors.append(_error(task.algorithm, n, "", ERROR_SKIPPED_SIZE, message))
            continue
        errors_before = len(result.errors)
        fitting = _generate_size(config, task, n, result)
        invalid = any(row["tipo"] == ERROR_INVALID_SIZE for row in result.errors[errors_before:])
        if not fitting and not invalid:
            stopped_at = n
    return result


def select_oracle_algorithms(
    config: DatasetConfig, requested: Sequence[str] | None
) -> tuple[str, ...]:
    """Oracle algorithms to generate: the requested ones (default all) that are classes."""
    requested = list(requested) if requested else list(ORACLE_ALGORITHMS)
    if unknown := [name for name in requested if name not in ORACLE_ALGORITHMS]:
        raise ValueError(f"no oracle generator for {unknown}; available: {ORACLE_ALGORITHMS}")
    if not_classes := [name for name in requested if name not in config.all_algorithms]:
        raise ValueError(f"{not_classes} are not class algorithms in the configuration")
    return tuple(dict.fromkeys(requested))


def generate_oracles(
    config: DatasetConfig,
    algorithms: Sequence[str],
    sizes: Sequence[int],
    *,
    force: bool = False,
    processes: int = 1,
) -> OracleReport:
    """Generate the oracle circuits, update the metadata and the oracle error log."""
    metadata = read_table(config.rutas.metadata, METADATA_COLUMNS)
    existing = frozenset(metadata.loc[metadata["origen"] == ORIGIN, "id"])
    tasks = [
        OracleTask(
            algorithm=algorithm,
            family=config.family_of(algorithm),
            sizes=tuple(sizes),
            per_size=config.oraculos.por_tamano,
            existing_ids=existing,
            force=force,
        )
        for algorithm in algorithms
    ]
    logger.info("generating oracles for %s, sizes %d-%d", ",".join(algorithms), sizes[0], sizes[-1])
    results = run_tasks(partial(run_oracle_task, config=config), tasks, processes, "oraculos")
    rows = pd.DataFrame([row for r in results for row in r.rows], columns=METADATA_COLUMNS)
    errors = pd.DataFrame([row for r in results for row in r.errors], columns=ERROR_COLUMNS)
    if len(rows):
        upsert(config.rutas.metadata, rows, "id", METADATA_COLUMNS)
    scopes = [(algorithm, n) for algorithm in algorithms for n in sizes]
    replace_scope(errors_path(config), errors, ERROR_SCOPE, scopes, ERROR_COLUMNS)
    skipped = sum(r.skipped for r in results)
    logger.info("%d oracle circuits written, %d already present", len(rows), skipped)
    return OracleReport(rows=rows, errors=errors, skipped=skipped)
