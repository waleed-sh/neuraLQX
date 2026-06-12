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


"""Rectangular square-lattice graph family.

This module exposes :class:`Grid2D`, the standard finite rectangular lattice
used by many small graph examples and U(1) lattice models. Vertices are placed
at integer row/column coordinates in the plane, with directed horizontal and
vertical edges generated in a deterministic order.
"""

from __future__ import annotations

from ._families import FamilyEdge
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class Grid2D(Graph):
    """A two-dimensional rectangular square-lattice graph.

    ``Grid2D`` contains ``m * n`` vertices arranged on integer coordinates
    ``(row, col, 0)``. Edges connect nearest horizontal and vertical neighbors.
    When ``periodic`` is enabled, rows and columns wrap around only when the
    corresponding direction has more than two sites, avoiding duplicate
    two-site wrap edges.

    Args:
        m: Number of rows.
        n: Number of columns.
        periodic: If ``True``, connect opposite boundaries in both directions.
        plot: If ``True``, display the graph after construction.
        non_planar: If ``True``, use non-planar sign conventions after a
            deterministic 3D relabelling.
        random_embedding: If ``True``, randomise non-planar coordinates.
        random_embedding_mean: Mean of the random embedding.
        random_embedding_std: Standard deviation of the random embedding.
        random_embedding_seed: Seed for the random embedding.

    Attributes:
        _m: Number of grid rows used to generate the rectangular lattice.
        _n: Number of grid columns used to generate the rectangular lattice.
        _periodic: Whether both grid directions have periodic boundaries.
    """

    _m: int
    """Number of grid rows used to generate the rectangular lattice."""

    _n: int
    """Number of grid columns used to generate the rectangular lattice."""

    _periodic: bool
    """Whether both grid directions have periodic boundaries."""

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
        """Initializes the rectangular grid graph.

        Args:
            m: Number of rows in the grid. Must be positive.
            n: Number of columns in the grid. Must be positive.
            periodic: Whether to connect opposite boundaries in both lattice
                directions.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to relabel the planar coordinates into a
                deterministic non-planar embedding and use non-planar signs.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        require_int("m", m, minimum=1)
        require_int("n", n, minimum=1)

        edges = _grid_edges(m, n, periodic=periodic)

        self._m = m
        self._n = n
        self._periodic = bool(periodic)

        super().__init__(
            maybe_relabel_nonplanar(edges, non_planar=non_planar),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    @property
    def m(self) -> int:
        """Number of grid rows used to generate the rectangular lattice."""
        return self._m

    @property
    def n(self) -> int:
        """Number of grid columns used to generate the rectangular lattice."""
        return self._n

    @property
    def periodic(self) -> bool:
        """Whether both grid directions have periodic boundaries."""
        return self._periodic

    def __repr__(self) -> str:
        """Returns a compact representation including grid dimensions."""
        return graph_repr(self, m=self.m, n=self.n, periodic=self.periodic)


def _grid_edges(m: int, n: int, *, periodic: bool) -> list[FamilyEdge]:
    """Builds deterministic nearest-neighbor edges for a rectangular grid.

    Args:
        m: Number of grid rows.
        n: Number of grid columns.
        periodic: Whether to add wraparound edges in both directions.

    Returns:
        A list of coordinate edges in horizontal-then-vertical order.
    """
    points = {(row, col): vertex(row, col) for row in range(m) for col in range(n)}
    edges: list[FamilyEdge] = []

    for row in range(m):
        for col in range(n - 1):
            edges.append((points[(row, col)], points[(row, col + 1)]))
        if periodic and n > 2:
            edges.append((points[(row, n - 1)], points[(row, 0)]))

    for row in range(m - 1):
        for col in range(n):
            edges.append((points[(row, col)], points[(row + 1, col)]))
    if periodic and m > 2:
        for col in range(n):
            edges.append((points[(m - 1, col)], points[(0, col)]))

    return edges


__all__ = [
    "Grid2D",
]
