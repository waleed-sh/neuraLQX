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


"""Open ladder graph family.

The module defines :class:`LadderGraph`, a two-rail graph with a finite number
of rungs. It is useful as a small planar lattice with both longitudinal and
transverse connectivity while remaining simpler than a full two-dimensional
grid.
"""

from __future__ import annotations

from ._families import FamilyEdge
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class LadderGraph(Graph):
    """An open two-rail ladder graph with ``N`` rungs.

    The graph has two parallel chains of length ``N``. Consecutive vertices on
    each rail are connected, and each pair of corresponding rail vertices is
    connected by a rung. The default embedding is planar with the two rails at
    ``y = 0`` and ``y = 1``.

    Args:
        N: Number of rungs and vertices per rail. Must be at least two.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _ladder_length: Number of rungs and vertices on each rail.
    """

    _ladder_length: int
    """Number of rungs and vertices on each rail."""

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
        """Initializes the open ladder graph.

        Args:
            N: Number of rungs and vertices per rail. Must be at least two.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        require_int("N", N, minimum=2)

        self._ladder_length = N

        super().__init__(
            maybe_relabel_nonplanar(_ladder_edges(N), non_planar=non_planar),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    @property
    def ladder_length(self) -> int:
        """Number of rungs and vertices on each rail."""
        return self._ladder_length

    def __repr__(self) -> str:
        """Returns a compact representation including the ladder length."""
        return graph_repr(self, ladder_length=self.ladder_length)


def _ladder_edges(length: int) -> list[FamilyEdge]:
    """Builds rail and rung edges for an open ladder.

    Args:
        length: Number of vertices per rail.

    Returns:
        Coordinate edges for both rails followed by rung edges.
    """
    bottom = [vertex(index, 0.0) for index in range(length)]
    top = [vertex(index, 1.0) for index in range(length)]
    edges: list[FamilyEdge] = []
    for index in range(length - 1):
        edges.append((bottom[index], bottom[index + 1]))
        edges.append((top[index], top[index + 1]))
    for index in range(length):
        edges.append((bottom[index], top[index]))
    return edges


__all__ = [
    "LadderGraph",
]
