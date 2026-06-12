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


"""Hexagonal honeycomb lattice graph family.

This module defines a finite honeycomb patch by placing regular hexagons in a
deterministic row/column layout and merging coincident vertices. Periodic
boundary support is intentionally rejected until exact boundary identification
is implemented.
"""

from __future__ import annotations

import math

from ._families import FamilyEdge
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class HexagonalLatticeGraph(Graph):
    """A finite honeycomb lattice made from ``m`` by ``n`` hexagonal cells.

    The constructor generates regular hexagon cycles, canonicalizes coincident
    vertices by rounded planar coordinates, and emits undirected geometric
    segments in a deterministic canonical orientation. The resulting graph is
    planar by default and can be relabelled for non-planar sign conventions.

    Args:
        m: Number of hexagon rows. Must be positive.
        n: Number of hexagon columns. Must be positive.
        periodic: Periodic boundary request. Currently unsupported.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _m: Number of hexagon rows in the generated honeycomb patch.
        _n: Number of hexagon columns in the generated honeycomb patch.
        _periodic: Whether periodic boundary conditions were requested.
    """

    _m: int
    """Number of hexagon rows in the generated honeycomb patch."""

    _n: int
    """Number of hexagon columns in the generated honeycomb patch."""

    _periodic: bool
    """Whether periodic boundary conditions were requested."""

    def __init__(
        self,
        m: int,
        n: int,
        periodic: bool = False,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the finite honeycomb lattice graph.

        Args:
            m: Number of hexagon rows. Must be positive.
            n: Number of hexagon columns. Must be positive.
            periodic: Periodic boundary request. ``True`` is not implemented.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.

        Raises:
            NotImplementedError: If ``periodic`` is ``True``.
        """
        require_int("m", m, minimum=1)
        require_int("n", n, minimum=1)
        if periodic:
            raise NotImplementedError(
                "Periodic hexagonal lattices need exact boundary "
                "identification and are not implemented yet."
            )

        self._m = m
        self._n = n
        self._periodic = bool(periodic)

        super().__init__(
            maybe_relabel_nonplanar(_hexagonal_edges(m, n), non_planar=non_planar),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    @property
    def m(self) -> int:
        """Number of hexagon rows in the generated honeycomb patch."""
        return self._m

    @property
    def n(self) -> int:
        """Number of hexagon columns in the generated honeycomb patch."""
        return self._n

    @property
    def periodic(self) -> bool:
        """Whether periodic boundary conditions were requested."""
        return self._periodic

    def __repr__(self) -> str:
        """Returns a compact representation including lattice dimensions."""
        return graph_repr(self, m=self.m, n=self.n, periodic=self.periodic)


def _hexagonal_edges(m: int, n: int) -> list[FamilyEdge]:
    """Builds canonical coordinate edges for a finite honeycomb patch.

    Args:
        m: Number of hexagon rows.
        n: Number of hexagon columns.

    Returns:
        Sorted coordinate edges forming the union of all generated hexagons.
    """
    vertices: dict[tuple[float, float], tuple[float, float, float]] = {}
    edges: set[FamilyEdge] = set()

    def canonical(x: float, y: float) -> tuple[float, float, float]:
        """Returns the shared rounded coordinate vertex for a point.

        Args:
            x: Raw x coordinate generated from the hexagon layout.
            y: Raw y coordinate generated from the hexagon layout.

        Returns:
            Stable coordinate vertex shared by coincident hexagon corners.
        """
        key = (round(x, 12), round(y, 12))
        if key not in vertices:
            vertices[key] = vertex(key[0], key[1])
        return vertices[key]

    angles = [math.pi / 3.0 * index for index in range(6)]
    for row in range(m):
        for col in range(n):
            center_x = 1.5 * col
            center_y = math.sqrt(3.0) * (row + 0.5 * (col % 2))
            cycle = [
                canonical(center_x + math.cos(angle), center_y + math.sin(angle))
                for angle in angles
            ]
            for index in range(6):
                first = cycle[index]
                second = cycle[(index + 1) % 6]
                edges.add((first, second) if first <= second else (second, first))

    return sorted(edges)


__all__ = [
    "HexagonalLatticeGraph",
]
