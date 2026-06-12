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


"""Abstract graph interface for neuraLQX graph implementations.

The module defines the shared contract implemented by concrete graph classes.
It centralizes edge canonicalization, vertex validation, optional non-planar
random embeddings, and common lookup helpers while leaving derived graph
structures to subclasses.
"""

from __future__ import annotations

import abc

from typing import Any

from neuralqx.utils.typing import DressedMinimalLoop
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import EdgeLike
from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import GraphEdgesLike
from neuralqx.utils.typing import MinimalLoop
from neuralqx.utils.typing import Vertex
from neuralqx.utils.typing import VertexLike

from .utils import edge_to_index as _edge_to_index
from .utils import ensure_edge_keys
from .utils import find_unique_vertices
from .utils import randomize_vertices
from .utils import validate_vertices


class AbstractGraph(abc.ABC):
    """A canonical graph boundary shared by all graph implementations.

    The base class owns only the universal graph boundary logic. Canonical edge
    construction, vertex discovery, planarity flagging, and deterministic
    optional embeddings. Concrete subclasses provide derived graph structures
    such as edge mappings, dual graphs, loops, and signs.

    Attributes:
        _edges: Canonical keyed edges after optional random embedding.
        _vertices: Unique canonical vertices in first-seen edge order.
        _is_planar: Whether planar orientation and plotting conventions apply.
        _original_edges: Canonical keyed edges before optional random embedding.
        _embedding_map: Mapping from original vertices to embedded vertices.
    """

    _edges: GraphEdges
    """Canonical keyed edges after optional random embedding."""

    _vertices: list[Vertex]
    """Unique canonical vertices in first-seen edge order."""

    _is_planar: bool
    """Whether planar orientation and plotting conventions apply."""

    _original_edges: GraphEdges
    """Canonical keyed edges before optional random embedding."""

    _embedding_map: dict[Vertex, Vertex]
    """Mapping from original vertices to embedded vertices."""

    def __init__(
        self,
        edges: GraphEdgesLike,
        *,
        is_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
    ) -> None:
        """Initializes the canonical graph boundary.

        Args:
            edges: Edge declarations accepted by :func:`ensure_edge_keys`.
                Endpoints may be ``Vertex`` objects or coordinate triples.
            is_planar: Whether the graph should use planar orientation
                conventions. Planar graphs keep their input coordinates.
            random_embedding: Whether to replace non-planar input coordinates
                with a deterministic random 3D embedding.
            random_embedding_mean: Mean of the Gaussian coordinate sampler used
                when ``random_embedding`` is enabled.
            random_embedding_std: Standard deviation of the Gaussian coordinate
                sampler used when ``random_embedding`` is enabled.
            random_embedding_seed: Seed passed to the random embedding helper.

        Raises:
            ValueError: If a random embedding is requested for a planar graph.
            TypeError: If an edge or vertex declaration cannot be canonicalized.
        """
        canonical_edges = ensure_edge_keys(edges)
        validate_vertices(canonical_edges)

        self._is_planar = bool(is_planar)
        self._original_edges = list(canonical_edges)
        self._embedding_map = {}

        if random_embedding:
            if self._is_planar:
                raise ValueError(
                    "Random embeddings are only available for non-planar graphs."
                )
            canonical_edges, self._embedding_map = randomize_vertices(
                canonical_edges,
                mean=random_embedding_mean,
                std=random_embedding_std,
                seed=random_embedding_seed,
            )

        self._edges = canonical_edges
        self._vertices = find_unique_vertices(canonical_edges)

    @property
    def edges(self) -> GraphEdges:
        """Canonical keyed graph edges in construction order.

        The returned edges are the normalized representation used internally by
        all graph algorithms. Parallel edges always carry explicit integer
        keys, and non-planar random embeddings are already reflected in the
        endpoint coordinates when that option was requested.
        """
        return self._edges

    @property
    def vertices(self) -> list[Vertex]:
        """Canonical graph vertices discovered from the edge list.

        Vertices are unique and preserve the first-seen order induced by the
        canonical edge sequence. This order is used by plotting, sign tables,
        and deterministic graph-family representations.
        """
        return self._vertices

    @property
    def original_edges(self) -> GraphEdges:
        """Canonical keyed edges before optional random embedding.

        This sequence records the validated user-facing topology prior to any
        randomized coordinate replacement. It is useful for diagnostics when a
        non-planar random embedding changes the displayed vertex coordinates.
        """
        return self._original_edges

    @property
    def embedding_map(self) -> dict[Vertex, Vertex]:
        """Mapping from original vertices to randomized embedded vertices.

        The mapping is empty unless random embedding was requested. When
        present, it records how canonical vertices from :attr:`original_edges`
        were translated to the coordinates exposed by :attr:`vertices`.
        """
        return self._embedding_map

    @property
    def is_planar(self) -> bool:
        """Whether signs and geometry use the planar convention.

        Planar graphs use two-dimensional vertex positions and pairwise
        orientation signs. Non-planar graphs use full three-dimensional
        coordinates and triple-edge orientation signs.
        """
        return self._is_planar

    @property
    def n_edges(self) -> int:
        """Number of canonical primal graph edges."""
        return len(self.edges)

    @property
    def n_vertices(self) -> int:
        """Number of canonical primal graph vertices."""
        return len(self.vertices)

    @property
    @abc.abstractmethod
    def mapping(self) -> dict[Edge, int]:
        """Mapping from canonical primal edges to basis-state indices.

        Concrete graphs provide the edge-degree-of-freedom ordering used by
        Hilbert spaces and dual graph helpers. Implementations should keep this
        mapping deterministic for a fixed edge sequence.
        """

    @property
    @abc.abstractmethod
    def inverse_mapping(self) -> dict[int, Edge]:
        """Mapping from basis-state indices back to canonical primal edges."""

    @property
    @abc.abstractmethod
    def signs(self) -> dict[Vertex, dict[Any, int]]:
        """Local orientation sign table for the primal graph.

        The concrete key shape depends on :attr:`is_planar`. Planar graphs use
        ordered incident edge pairs, while non-planar graphs use ordered
        incident edge triples.
        """

    @property
    @abc.abstractmethod
    def dual_signs(self) -> dict[Any, dict[Any, int]]:
        """Local orientation sign table for the line-graph dual."""

    @property
    @abc.abstractmethod
    def dual_edges(self) -> list[tuple[Edge, Edge]]:
        """Line-graph dual edges represented as pairs of primal edges.

        Each primal edge acts as one dual vertex. A pair appears when the two
        primal edges meet at a primal vertex.
        """

    @property
    @abc.abstractmethod
    def dual_adjacency(self) -> dict[Edge, list[Edge]]:
        """Adjacency lists for the line-graph dual."""

    @abc.abstractmethod
    def minimal_loops(self) -> list[MinimalLoop]:
        """Returns copies of all shortest primal loops.

        Implementations should return detached loop objects so callers cannot
        mutate the graph's cached internal loop state.
        """

    @abc.abstractmethod
    def minimal_dual_loops(self, *, edge_rep: bool = False) -> list:
        """Returns copies of all shortest loops in the line-graph dual.

        Args:
            edge_rep: Whether dual vertices should be expressed by integer
                primal edge indices rather than primal edge objects.
        """

    @abc.abstractmethod
    def dressed_minimal_loops(self) -> list[DressedMinimalLoop]:
        """Returns shortest primal loops with edge-action dressing metadata."""

    @abc.abstractmethod
    def dressed_minimal_dual_loops(self) -> list:
        """Returns shortest dual loops with edge-action dressing metadata."""

    def edge_to_index(self, edge: EdgeLike) -> int:
        """Maps an edge to its basis-state index.

        Args:
            edge: Edge-like object to canonicalize before lookup. The stored
                edge orientation and the reversed orientation are both accepted.

        Returns:
            Integer edge degree-of-freedom index for the requested edge.

        Raises:
            KeyError: If the edge is not part of the graph.
            TypeError: If ``edge`` cannot be converted to an ``Edge``.
            ValueError: If ``edge`` has invalid endpoint or key values.
        """
        canonical_edge = Edge.from_like(edge)
        index = _edge_to_index(canonical_edge, self.mapping)
        if index is None:
            raise KeyError(f"Unknown graph edge {canonical_edge!r}.")
        return index

    def index_to_edge(self, index: int) -> Edge:
        """Maps a basis-state index back to its canonical primal edge.

        Args:
            index: Integer edge degree-of-freedom index.

        Returns:
            The canonical edge stored at ``index``.

        Raises:
            IndexError: If ``index`` is not present in :attr:`inverse_mapping`.
        """
        try:
            return self.inverse_mapping[index]
        except KeyError as exc:
            raise IndexError(f"Edge index {index} is out of range.") from exc

    def valence(self, vertex: VertexLike) -> int:
        """Returns the number of canonical edges incident to a vertex.

        Args:
            vertex: Vertex-like object identifying the requested graph vertex.

        Returns:
            Count of stored edges whose start or end endpoint is ``vertex``.
        """
        canonical_vertex = Vertex.from_like(vertex)
        return sum(canonical_vertex in edge.vertices for edge in self.edges)

    def edges_at(self, vertex: VertexLike) -> list[Edge]:
        """Returns all canonical edges incident to a vertex.

        Args:
            vertex: Vertex-like object identifying the requested graph vertex.

        Returns:
            Stored edges whose start or end endpoint is ``vertex``.
        """
        canonical_vertex = Vertex.from_like(vertex)
        return [edge for edge in self.edges if canonical_vertex in edge.vertices]

    def edges_to(self, vertex: VertexLike) -> list[Edge]:
        """Returns all canonical edges whose end endpoint is a vertex.

        Args:
            vertex: Vertex-like object identifying the requested graph vertex.

        Returns:
            Stored edges directed into ``vertex``.
        """
        canonical_vertex = Vertex.from_like(vertex)
        return [edge for edge in self.edges if edge.end == canonical_vertex]

    def edges_from(self, vertex: VertexLike) -> list[Edge]:
        """Returns all canonical edges whose start endpoint is a vertex.

        Args:
            vertex: Vertex-like object identifying the requested graph vertex.

        Returns:
            Stored edges directed out of ``vertex``.
        """
        canonical_vertex = Vertex.from_like(vertex)
        return [edge for edge in self.edges if edge.start == canonical_vertex]

    @property
    def n_dual_vertices(self) -> int:
        """Number of line-graph dual vertices.

        A line-graph dual vertex corresponds to one primal graph edge, so this
        value is identical to :attr:`n_edges`.
        """
        return self.n_edges

    @property
    def n_dual_edges(self) -> int:
        """Number of line-graph dual edges."""
        return len(self.dual_edges)

    def __repr__(self) -> str:
        """Returns a compact diagnostic representation of the graph.

        The representation intentionally reports derived graph counts rather
        than the full edge list, keeping logs readable for large graph-family
        instances.
        """
        return (
            f"{type(self).__name__}("
            f"n_vertices={self.n_vertices}, "
            f"n_edges={self.n_edges}, "
            f"n_minimal_loops={len(self.minimal_loops())}, "
            f"is_planar={self.is_planar})"
        )


__all__ = [
    "AbstractGraph",
]
