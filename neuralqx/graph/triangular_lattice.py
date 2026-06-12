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


"""Triangular lattice graph family.

This module defines a finite triangular lattice patch using a deterministic
skew-coordinate embedding. Edges connect nearest neighbors along the three
positive lattice directions, with optional periodic wrapping.
"""

from __future__ import annotations

import math

from ._families import FamilyEdge
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class TriangularLatticeGraph(Graph):
    """A triangular lattice patch with optional periodic boundaries.

    The graph contains ``m * n`` lattice sites embedded in the plane with every
    other row shifted by half a lattice spacing. Each vertex connects along the
    horizontal, vertical-skew, and diagonal-skew directions. Periodic wrapping
    identifies opposite boundaries by index before coordinates are looked up.

    Args:
        m: Number of lattice rows. Must be at least two.
        n: Number of lattice columns. Must be at least two.
        periodic: Whether to wrap both lattice directions.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _m: Number of lattice rows in the triangular patch.
        _n: Number of lattice columns in the triangular patch.
        _periodic: Whether both lattice directions have periodic boundaries.
    """

    _m: int
    """Number of lattice rows in the triangular patch."""

    _n: int
    """Number of lattice columns in the triangular patch."""

    _periodic: bool
    """Whether both lattice directions have periodic boundaries."""

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
        """Initializes the triangular lattice graph.

        Args:
            m: Number of lattice rows. Must be at least two.
            n: Number of lattice columns. Must be at least two.
            periodic: Whether to wrap both lattice directions.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        require_int("m", m, minimum=2)
        require_int("n", n, minimum=2)

        self._m = m
        self._n = n
        self._periodic = bool(periodic)

        super().__init__(
            maybe_relabel_nonplanar(
                _triangular_edges(m, n, periodic=periodic),
                non_planar=non_planar,
            ),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    @property
    def m(self) -> int:
        """Number of lattice rows in the triangular patch."""
        return self._m

    @property
    def n(self) -> int:
        """Number of lattice columns in the triangular patch."""
        return self._n

    @property
    def periodic(self) -> bool:
        """Whether both lattice directions have periodic boundaries."""
        return self._periodic

    def __repr__(self) -> str:
        """Returns a compact representation including lattice dimensions."""
        return graph_repr(self, m=self.m, n=self.n, periodic=self.periodic)


def _triangular_edges(m: int, n: int, *, periodic: bool) -> list[FamilyEdge]:
    """Builds canonical coordinate edges for a triangular lattice patch.

    Args:
        m: Number of lattice rows.
        n: Number of lattice columns.
        periodic: Whether to wrap neighbor lookup across both boundaries.

    Returns:
        Sorted coordinate edges in canonical endpoint order.
    """
    height = math.sqrt(3.0) / 2.0
    points = {
        (row, col): vertex(col + 0.5 * (row % 2), height * row)
        for row in range(m)
        for col in range(n)
    }

    edges: set[FamilyEdge] = set()
    directions = ((0, 1), (1, 0), (1, -1))
    for row in range(m):
        for col in range(n):
            start = points[(row, col)]
            for drow, dcol in directions:
                next_row = row + drow
                next_col = col + dcol
                if periodic:
                    next_row %= m
                    next_col %= n
                if (next_row, next_col) not in points:
                    continue
                end = points[(next_row, next_col)]
                if start != end:
                    edges.add((start, end) if start <= end else (end, start))
    return sorted(edges)


__all__ = [
    "TriangularLatticeGraph",
]
