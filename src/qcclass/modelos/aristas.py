"""Edge (and angle) transformations of the F4 graphs for the GNN and its ablations.

The stored F4 graph has one edge ``u -> v`` per direct dependency (time direction) with the
four role indicators ``[origen_control, origen_destino, fin_control, fin_destino]``. The GNN
reads ``edge_attr`` of :data:`EDGE_DIM` = 6 columns: the four roles plus ``adelante`` and
``atras`` (direction). Variants (``variante_aristas``):

* ``completo`` (default): both directions. The reverse edge ``v -> u`` has the roles of its
  own ends (``[fin_control, fin_destino, origen_control, origen_destino]``) and ``atras = 1``.
* ``solo_adelante``: only the stored edges (``adelante = 1``).
* ``sin_direccion``: both directions, ``adelante = atras = 0``.
* ``sin_rol``: both directions with direction, roles set to 0.
* ``sin_angulos``: as ``completo``, with ``sin(theta)`` and ``cos(theta)`` of every node set
  to 0 (the plan's ablation; a gate without angle also gets 0 instead of ``cos = 1``).

Every variant keeps the same widths (``x`` 9, ``edge_attr`` 6), so the architecture and the
number of parameters do not change between ablations.
"""

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from qcclass.formatos.grafo_dag import NODE_FEATURES

EDGE_FEATURES = (
    "origen_control",
    "origen_destino",
    "fin_control",
    "fin_destino",
    "adelante",
    "atras",
)
EDGE_DIM = len(EDGE_FEATURES)
ANGLE_COLUMNS = (NODE_FEATURES.index("sin_theta"), NODE_FEATURES.index("cos_theta"))


@dataclass(frozen=True)
class EdgeVariant:
    """What an edge variant keeps."""

    reverse: bool = True
    direction: bool = True
    roles: bool = True
    angles: bool = True


VARIANTS = {
    "completo": EdgeVariant(),
    "solo_adelante": EdgeVariant(reverse=False),
    "sin_direccion": EdgeVariant(direction=False),
    "sin_rol": EdgeVariant(roles=False),
    "sin_angulos": EdgeVariant(angles=False),
}


class EdgeTransform:
    """Transform applied to every graph when it is loaded (see the module docstring)."""

    def __init__(self, variant: str = "completo") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown edge variant {variant!r} (valid: {list(VARIANTS)})")
        self.name = variant
        self.variant = VARIANTS[variant]

    def __call__(self, data: Data) -> Data:
        """A new graph with the transformed edges (and angles)."""
        v = self.variant
        out = data.clone()
        roles = data.edge_attr.float()
        if not v.roles:
            roles = torch.zeros_like(roles)
        count = roles.shape[0]
        forward_flags = torch.zeros(count, 2, dtype=roles.dtype)
        if v.direction:
            forward_flags[:, 0] = 1
        edge_index, edge_attr = data.edge_index, torch.cat([roles, forward_flags], dim=1)
        if v.reverse:
            backward_flags = torch.zeros(count, 2, dtype=roles.dtype)
            if v.direction:
                backward_flags[:, 1] = 1
            reverse_roles = roles[:, [2, 3, 0, 1]]
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = torch.cat(
                [edge_attr, torch.cat([reverse_roles, backward_flags], dim=1)], dim=0
            )
        out.edge_index = edge_index
        out.edge_attr = edge_attr
        if not v.angles:
            x = data.x.clone()
            x[:, list(ANGLE_COLUMNS)] = 0
            out.x = x
        return out

    def __repr__(self) -> str:
        return f"EdgeTransform({self.name!r})"
