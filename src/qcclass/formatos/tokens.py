"""F6: gate token sequences and vocabulary.

One token per gate, in node order: ``"rz q0 0.785"``, ``"sx q0"``, ``"x q1"``,
``"cx q0 q3"`` (control first). Angles are mapped to ``[-pi, pi)`` and rounded to three
decimals, so that equivalent rotations share a token.
"""

from collections import Counter
from collections.abc import Iterable

from qcclass.formatos.comun import GateRecord, wrap_angle

ANGLE_DECIMALS = 3


def _angle_text(angle: float) -> str:
    """Rounded angle, without a negative zero."""
    value = round(wrap_angle(angle), ANGLE_DECIMALS)
    return f"{0.0 if value == 0 else value:.{ANGLE_DECIMALS}f}"


def token(gate: GateRecord) -> str:
    """Token of a single gate."""
    parts = [gate.name, *(f"q{q}" for q in gate.qubits)]
    if gate.angle is not None:
        parts.append(_angle_text(gate.angle))
    return " ".join(parts)


def tokens(gates: list[GateRecord]) -> list[str]:
    """Token sequence of a circuit."""
    return [token(gate) for gate in gates]


def vocabulary(sequences: Iterable[list[str]]) -> dict[str, int]:
    """Token counts, most frequent first (ties in alphabetical order)."""
    counts = Counter(token for sequence in sequences for token in sequence)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
