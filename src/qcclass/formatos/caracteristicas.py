"""F3: global feature vector per circuit."""

from collections import Counter

import networkx as nx

from qcclass.circuitos import CANONICAL_BASIS
from qcclass.formatos.comun import GateRecord, asap_layers, interaction_graph


def _graph_features(graph: nx.Graph) -> dict[str, float]:
    """Mean degree, density, diameter (0 if disconnected) and components of a graph."""
    nodes = graph.number_of_nodes()
    connected = nodes > 0 and nx.is_connected(graph)
    return {
        "grafo_grado_medio": 2 * graph.number_of_edges() / nodes if nodes else 0.0,
        "grafo_densidad": nx.density(graph) if nodes > 1 else 0.0,
        "grafo_diametro": nx.diameter(graph) if connected and nodes > 1 else 0,
        "grafo_componentes": nx.number_connected_components(graph) if nodes else 0,
    }


def features(gates: list[GateRecord], num_qubits: int) -> dict[str, float]:
    """Feature vector of a canonical circuit.

    Columns: size and depth, count and share of each canonical gate, number of CX, share of
    two-qubit gates, CX per ASAP layer (mean over all layers and maximum), and four
    statistics of the qubit interaction graph.
    """
    total = len(gates)
    counts = Counter(gate.name for gate in gates)
    layers = asap_layers(gates, num_qubits)
    depth = max(layers) + 1 if layers else 0
    cx_per_layer = Counter(
        layer for layer, gate in zip(layers, gates, strict=True) if gate.name == "cx"
    )
    row: dict[str, float] = {"n_qubits": num_qubits, "profundidad": depth, "n_puertas": total}
    for name in CANONICAL_BASIS:
        row[f"n_{name}"] = counts.get(name, 0)
        row[f"prop_{name}"] = counts.get(name, 0) / total if total else 0.0
    row["prop_dos_cubits"] = row["prop_cx"]
    row["cx_por_capa_media"] = sum(cx_per_layer.values()) / depth if depth else 0.0
    row["cx_por_capa_max"] = max(cx_per_layer.values(), default=0)
    row.update(_graph_features(interaction_graph(gates, num_qubits)))
    return row
