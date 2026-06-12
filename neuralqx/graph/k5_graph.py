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


"""Complete graph on five vertices.

This module exposes :class:`K5Graph`, the complete graph with five vertices.
The graph is a canonical small non-planar example, so the constructor defaults
to non-planar sign conventions.
"""

from __future__ import annotations

from ._families import circular_vertices
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from .graph import Graph


class K5Graph(Graph):
    """The complete graph ``K_5``.

    ``K_5`` is non-planar, so this class defaults to ``non_planar=True``.
    Set ``non_planar=False`` only when you explicitly want planar sign
    conventions for a projected drawing.

    Args:
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to use non-planar sign conventions.
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
        non_planar: bool = True,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the complete graph on five vertices.

        Args:
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use non-planar sign conventions.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        vertices = circular_vertices(5)
        edges = [
            (vertices[first], vertices[second])
            for first in range(5)
            for second in range(first + 1, 5)
        ]
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
        """Returns a compact representation of the complete graph."""
        return graph_repr(self)


__all__ = [
    "K5Graph",
]
