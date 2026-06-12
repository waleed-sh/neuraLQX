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


"""Hypercube graph family.

This module exposes :class:`HypercubeGraph`, a deterministic coordinate
realization of the ``N``-dimensional cube graph ``Q_N``. The topology is built
from bit-flip adjacency while the display coordinates are arranged on a compact
two-dimensional grid before optional non-planar relabelling.
"""

from __future__ import annotations

import math

from ._families import FamilyEdge
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class HypercubeGraph(Graph):
    """An ``N``-dimensional hypercube graph ``Q_N``.

    Vertices correspond to binary strings of length ``N``. Two vertices are
    adjacent when their labels differ in exactly one bit. The class defaults to
    planar conventions only for dimensions one and two, because higher
    hypercubes are naturally non-planar as geometric graph families.

    Args:
        N: Dimension of the hypercube. Must be positive.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to use non-planar sign conventions. When ``None``,
            the value defaults to ``True`` for dimensions greater than two.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _N: Hypercube dimension used to build the graph.
    """

    _N: int
    """Hypercube dimension used to build the graph."""

    def __init__(
        self,
        N: int,
        *,
        plot: bool = False,
        non_planar: bool | None = None,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the hypercube graph.

        Args:
            N: Dimension of the hypercube. Must be positive.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use non-planar sign conventions. If
                ``None``, dimensions greater than two are treated as
                non-planar.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        require_int("N", N, minimum=1)
        if non_planar is None:
            non_planar = N > 2

        self._N = N

        super().__init__(
            maybe_relabel_nonplanar(_hypercube_edges(N), non_planar=non_planar),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    @property
    def N(self) -> int:
        """Hypercube dimension used to build the graph."""
        return self._N

    def __repr__(self) -> str:
        """Returns a compact representation including the hypercube dimension."""
        return graph_repr(self, N=self.N)


def _hypercube_edges(dimension: int) -> list[FamilyEdge]:
    """Builds deterministic bit-flip edges for a hypercube.

    Args:
        dimension: Number of binary coordinates in each hypercube vertex.

    Returns:
        Coordinate edges joining labels that differ by one bit.
    """
    n_vertices = 2**dimension
    columns = math.ceil(math.sqrt(n_vertices))
    points = {
        index: vertex(index % columns, index // columns) for index in range(n_vertices)
    }
    edges: list[FamilyEdge] = []
    for index in range(n_vertices):
        for bit in range(dimension):
            neighbour = index ^ (1 << bit)
            if index < neighbour:
                edges.append((points[index], points[neighbour]))
    return edges


__all__ = [
    "HypercubeGraph",
]
