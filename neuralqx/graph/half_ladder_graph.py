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


"""Half-ladder graph family.

This module exposes :class:`HalfLadderGraph`, a chain of ``k`` vertices with
one attached ``mu`` vertex per chain vertex. The class preserves convenient
accessors for the two edge and vertex families used by spherical-symmetry
models.
"""

from __future__ import annotations

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import Vertex
from neuralqx.utils.typing import VertexLike

from ._families import FamilyEdge
from ._families import graph_repr
from ._families import relabel_edges_to_nonplanar
from ._families import require_int
from ._families import vertex
from .graph import Graph


class HalfLadderGraph(Graph):
    """Half-ladder graph used by the spherical-symmetry inspired models.

    The graph has a chain of ``k`` vertices and one external ``mu`` vertex
    attached to each ``k`` vertex. When ``connected=True``, the last ``k``
    vertex is also connected back to the first.

    Args:
        length: Number of k-chain segments. The graph contains ``length + 1``
            k-vertices and the same number of mu-vertices.
        connected: Whether to close the k-chain into a cycle.
        plot: Whether to display the graph immediately after construction.
        non_planar: Whether to relabel coordinates for non-planar sign
            conventions.
        random_embedding: Whether to randomize non-planar coordinates.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian random
            embedding.
        random_embedding_seed: Seed for deterministic random embedding.

    Attributes:
        _vertices_k: Ordered vertices along the main k-chain.
        _vertices_mu: Ordered external mu-vertices.
        _is_connected: Whether the k-chain has a closing edge.
        _edges_k: Ordered edges along the main k-chain.
        _edges_mu: Ordered external mu-edges attached to the k-chain.
        edges_mu_idx: Edge indices for the external mu-edges.
    """

    _vertices_k: tuple[Vertex, ...]
    """Ordered vertices along the main k-chain."""

    _vertices_mu: tuple[Vertex, ...]
    """Ordered external mu-vertices."""

    _is_connected: bool
    """Whether the k-chain has a closing edge."""

    _edges_k: tuple[Edge, ...]
    """Ordered edges along the main k-chain."""

    _edges_mu: tuple[Edge, ...]
    """Ordered external mu-edges attached to the k-chain."""

    edges_mu_idx: tuple[int, ...]
    """Edge indices for the external mu-edges."""

    def __init__(
        self,
        length: int,
        connected: bool = False,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the half-ladder graph.

        Args:
            length: Number of k-chain segments. Must be positive.
            connected: Whether to close the k-chain into a cycle.
            plot: Whether to display the graph immediately after construction.
            non_planar: Whether to use deterministic non-planar relabelling.
            random_embedding: Whether to randomize non-planar coordinates.
            random_embedding_mean: Mean of the Gaussian random embedding.
            random_embedding_std: Standard deviation of the Gaussian random
                embedding.
            random_embedding_seed: Seed for deterministic random embedding.
        """
        require_int("length", length, minimum=1)

        vertices_k = [vertex(index, 0.0) for index in range(length + 1)]
        vertices_mu = [vertex(index, 1.0) for index in range(length + 1)]
        edges_k: list[FamilyEdge] = [
            (vertices_k[index], vertices_k[index + 1]) for index in range(length)
        ]
        if connected:
            edges_k.append((vertices_k[-1], vertices_k[0]))
        edges_mu: list[FamilyEdge] = [
            (start, end) for start, end in zip(vertices_k, vertices_mu)
        ]
        edges = edges_k + edges_mu

        if non_planar:
            relabelled, mapping = relabel_edges_to_nonplanar(edges)
            edges = relabelled
            vertices_k = [mapping[Vertex.from_like(item)] for item in vertices_k]
            vertices_mu = [mapping[Vertex.from_like(item)] for item in vertices_mu]
            edges_k = [_map_edge(edge, mapping) for edge in edges_k]
            edges_mu = [_map_edge(edge, mapping) for edge in edges_mu]

        self._vertices_k = tuple(Vertex.from_like(item) for item in vertices_k)
        self._vertices_mu = tuple(Vertex.from_like(item) for item in vertices_mu)
        self._is_connected = bool(connected)

        super().__init__(
            edges,
            is_planar=not non_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
            plot=plot,
        )

        self._edges_k = tuple(
            self._embedded_edge(Edge.from_like(edge)) for edge in edges_k
        )
        self._edges_mu = tuple(
            self._embedded_edge(Edge.from_like(edge)) for edge in edges_mu
        )
        self._vertices_k = tuple(
            self.embedding_map.get(item, item) for item in self._vertices_k
        )
        self._vertices_mu = tuple(
            self.embedding_map.get(item, item) for item in self._vertices_mu
        )
        self.edges_mu_idx = tuple(self.edge_to_index(edge) for edge in self._edges_mu)

    @property
    def edges_k(self) -> tuple[Edge, ...]:
        """Ordered edges along the main k-chain."""
        return self._edges_k

    @property
    def edges_mu(self) -> tuple[Edge, ...]:
        """Ordered external mu-edges attached to the k-chain."""
        return self._edges_mu

    @property
    def vertices_k(self) -> tuple[Vertex, ...]:
        """Ordered vertices along the main k-chain."""
        return self._vertices_k

    @property
    def vertices_mu(self) -> tuple[Vertex, ...]:
        """Ordered external mu-vertices."""
        return self._vertices_mu

    @property
    def is_connected(self) -> bool:
        """Whether the k-chain has a closing edge."""
        return self._is_connected

    def get_edges_at_k_vertex(
        self,
        vertex_value: VertexLike,
        dual_rep: bool = False,
    ) -> tuple[list[Edge | int], list[Edge | int]]:
        """Returns the mu-edge and k-edges incident to a k-vertex.

        Args:
            vertex_value: Vertex-like value expected to identify a k-chain
                vertex.
            dual_rep: Whether to return edge indices instead of edge objects.

        Returns:
            A pair ``(mu_edges, k_edges)``. Each item is a list of incident
            edge objects unless ``dual_rep`` is ``True``, in which case edge
            indices are returned.

        Raises:
            ValueError: If ``vertex_value`` is not one of the k-chain vertices.
        """
        target = Vertex.from_like(vertex_value)
        if target not in self.vertices_k:
            raise ValueError(f"The requested vertex {target!r} is not a k-vertex.")

        k_edges = [edge for edge in self.edges_k if target in edge.vertices]
        mu_edges = [edge for edge in self.edges_mu if target in edge.vertices]
        if dual_rep:
            return (
                [self.edge_to_index(edge) for edge in mu_edges],
                [self.edge_to_index(edge) for edge in k_edges],
            )
        return mu_edges, k_edges

    def has_k_vertex_neighbours(self, vertex_value: VertexLike, direction: str) -> bool:
        """Checks whether a k-vertex has a left or right k-neighbor.

        Args:
            vertex_value: Vertex-like value expected to identify a k-chain
                vertex.
            direction: Direction to inspect. Must be ``"left"`` or
                ``"right"``.

        Returns:
            ``True`` when a k-edge exists in the requested direction.

        Raises:
            ValueError: If ``direction`` is invalid or ``vertex_value`` is not
                one of the k-chain vertices.
        """
        target = Vertex.from_like(vertex_value)
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'.")
        if target not in self.vertices_k:
            raise ValueError(f"The requested vertex {target!r} is not a k-vertex.")
        index = self.vertices_k.index(target)
        if direction == "left":
            return index > 0 or self.is_connected
        return index < len(self.vertices_k) - 1 or self.is_connected

    def get_kminus_kplus(
        self,
        vertex_value: VertexLike,
        dual_rep: bool = False,
    ) -> list[list[Edge | int]]:
        """Returns left and right k-edges incident to a k-vertex.

        Args:
            vertex_value: Vertex-like value expected to identify a k-chain
                vertex.
            dual_rep: Whether to return edge indices instead of edge objects.

        Returns:
            A two-item list ``[k_minus, k_plus]`` where each entry contains the
            incident k-edges on that side of the vertex.
        """
        target = Vertex.from_like(vertex_value)
        _, k_edges = self.get_edges_at_k_vertex(target, dual_rep=False)
        left: list[Edge] = []
        right: list[Edge] = []
        for edge in k_edges:
            other = edge.end if edge.start == target else edge.start
            if self.vertices_k.index(other) < self.vertices_k.index(target):
                left.append(edge)
            else:
                right.append(edge)
        if dual_rep:
            return [
                [self.edge_to_index(edge) for edge in left],
                [self.edge_to_index(edge) for edge in right],
            ]
        return [left, right]

    def __repr__(self) -> str:
        """Returns a compact representation of half-ladder dimensions."""
        return graph_repr(
            self,
            n_edges_k=len(self.edges_k),
            n_edges_mu=len(self.edges_mu),
            n_vertices_k=len(self.vertices_k),
            n_vertices_mu=len(self.vertices_mu),
            is_connected=self.is_connected,
        )

    def _embedded_edge(self, edge: Edge) -> Edge:
        """Returns ``edge`` after applying any random embedding map.

        Args:
            edge: Canonical edge from the pre-embedding k or mu edge family.

        Returns:
            Edge with endpoints translated through :attr:`embedding_map` when
            a random embedding is active.
        """
        start = self.embedding_map.get(edge.start, edge.start)
        end = self.embedding_map.get(edge.end, edge.end)
        return Edge(start, end, edge.key)


def _map_edge(
    edge: FamilyEdge, mapping: dict[Vertex, tuple[float, float, float]]
) -> FamilyEdge:
    """Maps an edge through deterministic non-planar vertex relabelling.

    Args:
        edge: Original family edge to relabel.
        mapping: Vertex-to-coordinate mapping returned by
            :func:`relabel_edges_to_nonplanar`.

    Returns:
        Relabelled family edge preserving its original orientation.
    """
    canonical = Edge.from_like(edge)
    return (mapping[canonical.start], mapping[canonical.end])


__all__ = [
    "HalfLadderGraph",
]
