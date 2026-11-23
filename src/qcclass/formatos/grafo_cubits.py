"""F5: qubit interaction graph as a PyTorch Geometric graph.

Nodes are qubits (all of them, idle ones included). Two qubits are joined when they share
at least one CX; the edge weight is the number of CX between them and each undirected edge
is stored in both directions. Node attributes: number of single-qubit gates on the qubit
and its degree in the interaction graph.
"""

import torch
from torch_geometric.data import Data

from qcclass.formatos.comun import GateRecord, interaction_graph

NODE_FEATURES = ("n_puertas_1q", "grado")


def qubit_graph(gates: list[GateRecord], num_qubits: int) -> Data:
    """Build the F5 graph (labels are added by the caller)."""
    graph = interaction_graph(gates, num_qubits)
    single = [0] * num_qubits
    for gate in gates:
        if len(gate.qubits) == 1:
            single[gate.qubits[0]] += 1
    x = torch.tensor(
        [[single[q], graph.degree(q)] for q in range(num_qubits)], dtype=torch.float32
    ).reshape(num_qubits, len(NODE_FEATURES))
    sources, targets, weights = [], [], []
    for a, b, data in sorted(graph.edges(data=True)):
        sources += [a, b]
        targets += [b, a]
        weights += [data["weight"], data["weight"]]
    edge_index = torch.tensor([sources, targets], dtype=torch.long).reshape(2, len(sources))
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_qubits)
