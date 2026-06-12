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


"""Dodecahedral graph family.

This module exposes :class:`DodecahedralGraph`, a fixed 20-vertex graph with
the connectivity of the regular dodecahedron. The coordinates are arranged on a
circle for deterministic plotting, while the edge list preserves the standard
three-regular topology.
"""

from __future__ import annotations

from ._families import FamilyEdge
from ._families import circular_vertices
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from .graph import Graph


class DodecahedralGraph(Graph):
    """The 20-vertex, 30-edge dodecahedral graph.

    The graph is useful as a compact cubic graph with many pentagonal cycles.
    The constructor accepts the same plotting and embedding options as
    :class:`Graph`, but the topology itself is fixed.

    Args:
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.
    """

    def __init__(
        self,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the fixed dodecahedral graph.

        Args:
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        edges = _dodecahedral_edges()
        super().__init__(
            maybe_relabel_nonplanar(edges, non_planar=non_planar),
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

    def __repr__(self) -> str:
        """Returns a compact representation of the dodecahedral graph."""
        return graph_repr(self)


def _dodecahedral_edges() -> list[FamilyEdge]:
    """Builds the fixed dodecahedral edge list.

    Returns:
        Thirty coordinate edges joining the vertices of the dodecahedral
        graph.
    """
    vertices = circular_vertices(20)
    index_edges = [
        (0, 1),
        (0, 10),
        (0, 19),
        (1, 2),
        (1, 8),
        (2, 3),
        (2, 6),
        (3, 4),
        (3, 19),
        (4, 5),
        (4, 17),
        (5, 6),
        (5, 15),
        (6, 7),
        (7, 8),
        (7, 14),
        (8, 9),
        (9, 10),
        (9, 13),
        (10, 11),
        (11, 12),
        (11, 18),
        (12, 13),
        (12, 16),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
    ]
    return [(vertices[start], vertices[end]) for start, end in index_edges]


__all__ = [
    "DodecahedralGraph",
]
