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


"""Single central vertex graph family.

This module exposes :class:`SingleVertexGraph`, a star graph centered at the
origin with a configurable valence and per-edge orientation. It is a compact
graph family for local vertex experiments and operator sanity checks.
"""

from __future__ import annotations

from ._families import circular_vertices
from ._families import graph_repr
from ._families import maybe_relabel_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class SingleVertexGraph(Graph):
    """A star graph with one central vertex and prescribed orientations.

    The central vertex is placed at the origin. Leaf vertices are distributed
    around a unit circle, and each incident edge is oriented according to the
    supplied orientation sequence. A value of ``1`` directs the edge away from
    the center, while ``-1`` directs it toward the center.

    Args:
        valence: Number of incident edges at the central vertex.
        orientation: Optional orientation sequence with one entry per edge.
            Each entry must be ``1`` or ``-1``.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _vertex_valence: Valence of the central vertex.
        _orientation: Direction convention for each incident edge.
    """

    _vertex_valence: int
    """Valence of the central vertex."""

    _orientation: tuple[int, ...]
    """Direction convention for each incident edge."""

    def __init__(
        self,
        valence: int,
        orientation: list[int] | tuple[int, ...] | None = None,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the single-vertex star graph.

        Args:
            valence: Number of incident edges at the central vertex.
            orientation: Optional orientation sequence with one entry per edge.
                Each entry must be ``1`` or ``-1``.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.

        Raises:
            ValueError: If ``orientation`` has the wrong length or contains
                values other than ``1`` and ``-1``.
        """
        require_int("valence", valence, minimum=1)
        if orientation is None:
            orientation_tuple = tuple(1 for _ in range(valence))
        else:
            orientation_tuple = tuple(orientation)
        if len(orientation_tuple) != valence:
            raise ValueError(
                f"orientation must have length {valence}; got {len(orientation_tuple)}."
            )
        if any(item not in (-1, 1) for item in orientation_tuple):
            raise ValueError("orientation entries must be either +1 or -1.")

        center = vertex(0.0, 0.0)
        leaves = circular_vertices(valence, radius=1.0)
        edges = [
            (center, leaf) if direction == 1 else (leaf, center)
            for leaf, direction in zip(leaves, orientation_tuple)
        ]

        self._vertex_valence = valence
        self._orientation = orientation_tuple

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
    def orientation(self) -> tuple[int, ...]:
        """Orientation convention for each incident edge.

        A value of ``1`` means the matching edge starts at the central vertex.
        A value of ``-1`` means the matching edge ends at the central vertex.
        """
        return self._orientation

    @property
    def vertex_valence(self) -> int:
        """Valence of the central vertex."""
        return self._vertex_valence

    def __repr__(self) -> str:
        """Returns a compact representation including valence and orientation."""
        return graph_repr(
            self,
            vertex_valence=self.vertex_valence,
            orientation=self.orientation,
        )


__all__ = [
    "SingleVertexGraph",
]
