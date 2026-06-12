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


"""Circular ladder graph family.

This module provides :class:`CircularLadderGraph`, the prism-like ladder whose
rails are cycles rather than open paths. The graph is useful when a compact
finite lattice with periodic longitudinal direction is desired.
"""

from __future__ import annotations

from ._families import FamilyEdge
from ._families import circular_vertices
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from .graph import Graph


class CircularLadderGraph(Graph):
    """A circular ladder graph ``C_N x K_2``.

    The graph has an inner cycle, an outer cycle, and ``N`` radial rungs
    connecting corresponding vertices. Coordinates place the two cycles on
    concentric circles, making the default graph planar and visually stable.

    Args:
        N: Cycle length and number of rungs. Must be at least three.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _ladder_length: Cycle length and number of radial rungs.
    """

    _ladder_length: int
    """Cycle length and number of radial rungs."""

    def __init__(
        self,
        N: int,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the circular ladder graph.

        Args:
            N: Cycle length and number of rungs. Must be at least three.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        require_int("N", N, minimum=3)

        self._ladder_length = N

        super().__init__(
            maybe_relabel_nonplanar(_circular_ladder_edges(N), non_planar=non_planar),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    @property
    def ladder_length(self) -> int:
        """Cycle length and number of radial rungs."""
        return self._ladder_length

    def __repr__(self) -> str:
        """Returns a compact representation including the cycle length."""
        return graph_repr(self, ladder_length=self.ladder_length)


def _circular_ladder_edges(length: int) -> list[FamilyEdge]:
    """Builds cycle and rung edges for a circular ladder.

    Args:
        length: Number of vertices in each cycle.

    Returns:
        Coordinate edges around both cycles and between corresponding vertices.
    """
    inner = circular_vertices(length, radius=1.0)
    outer = circular_vertices(length, radius=2.0)
    edges: list[FamilyEdge] = []
    for index in range(length):
        nxt = (index + 1) % length
        edges.append((inner[index], inner[nxt]))
        edges.append((outer[index], outer[nxt]))
        edges.append((inner[index], outer[index]))
    return edges


__all__ = [
    "CircularLadderGraph",
]
