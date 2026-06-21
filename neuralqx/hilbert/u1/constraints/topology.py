# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Dependency topology for constructive U(1) gauge fixing.

Gauge relations can depend on other slave edges. This module orders those
dependencies so reconstruction can update slaves after all of their sources are
available and rejects cyclic specifications.
"""

from __future__ import annotations

from collections import defaultdict
from collections import deque
from collections.abc import Sequence

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field

from .errors import U1CyclicGaugeFixingError
from .errors import U1GaugeFixingError
from .relation import GaugeRelation


class GaugeFixingTopology(Struct):
    """Dependency-aware gauge-fixing topology on graph edge indices.

    Args:
        n_edges: Number of base graph edges.
        free: Base edges that remain independently sampled.
        slave_topo: Slave edges in dependency-safe reconstruction order.
        slave_fixers: Signed source edges for each slave in ``slave_topo``.

    Attributes:
        n_edges: Number of base graph edges.
        free: Base edges that remain independently sampled.
        slave_topo: Slave edges in dependency-safe reconstruction order.
        slave_fixers: Signed source edges for each slave in ``slave_topo``.
    """

    n_edges: int = field(static=True)
    """Number of base graph edges."""
    free: tuple[int, ...] = field(static=True)
    """Base edges that remain independently sampled."""
    slave_topo: tuple[int, ...] = field(static=True)
    """Slave edges in dependency-safe reconstruction order."""
    slave_fixers: tuple[tuple[tuple[int, int], ...], ...] = field(static=True)
    """Signed source edges for each slave in ``slave_topo``."""

    def __post_init__(self) -> None:
        """Normalizes topology fields to immutable integer tuples."""
        object.__setattr__(self, "n_edges", int(self.n_edges))
        object.__setattr__(self, "free", tuple(int(edge) for edge in self.free))
        object.__setattr__(
            self, "slave_topo", tuple(int(edge) for edge in self.slave_topo)
        )
        object.__setattr__(
            self,
            "slave_fixers",
            tuple(
                tuple((int(edge), int(sign)) for edge, sign in row)
                for row in self.slave_fixers
            ),
        )

    def __hash__(self) -> int:
        """Returns a structural hash for the topology."""
        return hash(
            (type(self), self.n_edges, self.free, self.slave_topo, self.slave_fixers)
        )

    @property
    def n_free(self) -> int:
        """Number of independently sampled base edges."""
        return len(self.free)


def build_topology(
    relations: Sequence[GaugeRelation], *, n_edges: int
) -> GaugeFixingTopology:
    """Builds free/slave ordering and rejects dependency cycles.

    Args:
        relations: Constructive gauge relations.
        n_edges: Number of base graph edges.

    Returns:
        Dependency-aware gauge-fixing topology.

    Raises:
        U1GaugeFixingError: If any relation references an out-of-range edge.
        U1CyclicGaugeFixingError: If relations contain a dependency cycle.
    """
    slave_fixers = {relation.lhs: relation.rhs for relation in relations}
    if any(edge < 0 or edge >= n_edges for edge in slave_fixers):
        raise U1GaugeFixingError("Gauge-fixing LHS edge out of range.")

    deps: dict[int, list[int]] = defaultdict(list)
    indeg: dict[int, int] = {edge: 0 for edge in range(n_edges)}
    for relation in relations:
        for edge, _sign in relation.rhs:
            if edge < 0 or edge >= n_edges:
                raise U1GaugeFixingError("Gauge-fixing RHS edge out of range.")
            deps[edge].append(relation.lhs)
            indeg[relation.lhs] += 1

    queue = deque(edge for edge in range(n_edges) if indeg[edge] == 0)
    topo: list[int] = []
    while queue:
        edge = queue.popleft()
        topo.append(edge)
        for child in deps.get(edge, ()):
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)

    if len(topo) != n_edges:
        raise U1CyclicGaugeFixingError("Gauge-fixing dependencies contain a cycle.")

    free = tuple(edge for edge in topo if edge not in slave_fixers)
    slave_topo = tuple(edge for edge in topo if edge in slave_fixers)
    fixers = tuple(slave_fixers[edge] for edge in slave_topo)
    return GaugeFixingTopology(
        n_edges=n_edges,
        free=tuple(sorted(free)),
        slave_topo=slave_topo,
        slave_fixers=fixers,
    )
