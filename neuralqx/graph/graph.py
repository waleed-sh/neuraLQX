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


"""Concrete graph implementation for neuraLQX.

The module contains the primary user-facing graph class. It augments
``AbstractGraph`` with edge-index mappings, dual graph construction, shortest
loop enumeration, dressed loop metadata, plotting coordinates, and local
orientation-sign tables used by graph-based quantum operators.
"""

from __future__ import annotations

from typing import Any

from neuralqx.utils.typing import DressedMinimalLoop
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import GraphEdgesLike
from neuralqx.utils.typing import MinimalLoop
from neuralqx.utils.typing import Vertex

from .abstract_graph import AbstractGraph
from .utils import build_adjacency_from_edge_pairs
from .utils import build_directed_connectivity
from .utils import build_dual_edges
from .utils import build_edge_mapping
from .utils import compute_graph_signs
from .utils import compute_raw_signs
from .utils import dress_loops
from .utils import dual_positions
from .utils import graph_positions
from .utils import invert_edge_mapping
from .utils import minimal_cycles
from .utils import minimal_raw_cycles


class Graph(AbstractGraph):
    """A directed keyed multigraph with derived loop and dual structures.

    A ``Graph`` is built from oriented edges. Each edge may be supplied either
    as ``(start, end)`` or ``(start, end, key)`` where ``start`` and ``end`` are
    Cartesian coordinate triples ``(x, y, z)`` denoting vertices and ``key``
    distinguishes parallel edges. Unkeyed parallel edges are keyed
    automatically in input order, so compact user input remains unambiguous.

    The class precomputes the structures used by graph-based operators:

    - primal edge-to-index mappings,
    - the line-graph dual,
    - shortest primal and dual loops,
    - dressed primal loops,
    - local orientation signs for planar or non-planar embeddings.

    Args:
        edges: Oriented graph edges. Vertices are coordinate triples.
        is_planar: Whether planar sign conventions should be used. Non-planar
            graphs use 3D triple-edge signs.
        random_embedding: If ``True``, replace the input coordinates by a
            deterministic random 3D embedding. This is available only when
            ``is_planar=False``.
        random_embedding_mean: Mean of the Gaussian random embedding.
        random_embedding_std: Standard deviation of the Gaussian embedding.
        random_embedding_seed: Seed controlling the random embedding.
        plot: If ``True``, display the graph after construction using
            :meth:`plot`.

    Attributes:
        _mapping: Stable mapping from canonical primal edges to state indices.
        _inverse_mapping: Stable mapping from state indices to primal edges.
        _incidence: Incoming and outgoing primal edges at each primal vertex.
        _dual_edges: Edges of the line-graph dual as pairs of primal edges.
        _dual_adjacency: Adjacency lists of the line-graph dual.
        _graph_positions: Coordinates used for primal plotting and signs.
        _dual_positions: Midpoint coordinates used for dual plotting and signs.
        _minimal_loops: Cached shortest primal loops.
        _dressed_minimal_loops: Cached shortest primal loops with dressing.
        _minimal_dual_loops: Cached shortest loops in the line-graph dual.
        _dressed_minimal_dual_loops: Cached dressed shortest dual loops.
        _signs: Local orientation signs of the primal graph.
        _dual_signs: Local orientation signs of the line-graph dual.
    """

    _mapping: dict[Edge, int]
    """Stable mapping from canonical primal edges to state indices."""

    _inverse_mapping: dict[int, Edge]
    """Stable mapping from state indices to primal edges."""

    _incidence: dict[Vertex, dict[str, list[Edge]]]
    """Incoming and outgoing primal edges at each primal vertex."""

    _dual_edges: list[tuple[Edge, Edge]]
    """Edges of the line-graph dual as pairs of primal edges."""

    _dual_adjacency: dict[Edge, list[Edge]]
    """Adjacency lists of the line-graph dual."""

    _graph_positions: dict[Vertex, tuple[float, ...]]
    """Coordinates used for primal plotting and signs."""

    _dual_positions: dict[Edge, tuple[float, ...]]
    """Midpoint coordinates used for dual plotting and signs."""

    _minimal_loops: list[MinimalLoop]
    """Cached shortest primal loops."""

    _dressed_minimal_loops: list[DressedMinimalLoop]
    """Cached shortest primal loops with dressing."""

    _minimal_dual_loops: list[list[tuple[Edge, Edge, int]]]
    """Cached shortest loops in the line-graph dual."""

    _dressed_minimal_dual_loops: list
    """Cached dressed shortest dual loops."""

    _signs: dict[Vertex, dict[Any, int]]
    """Local orientation signs of the primal graph."""

    _dual_signs: dict[Any, dict[Any, int]]
    """Local orientation signs of the line-graph dual."""

    def __init__(
        self,
        edges: GraphEdgesLike,
        *,
        is_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int | str | bytes | None = 123,
        plot: bool = False,
    ) -> None:
        """Initializes and precomputes a concrete graph.

        Args:
            edges: Oriented graph edges. Vertices may be coordinate triples or
                ``Vertex`` objects. Edge declarations may be keyed or unkeyed.
            is_planar: Whether planar orientation conventions should be used.
            random_embedding: Whether to replace non-planar coordinates with a
                deterministic random embedding before precomputation.
            random_embedding_mean: Mean used by the random embedding sampler.
            random_embedding_std: Standard deviation used by the random
                embedding sampler.
            random_embedding_seed: Seed used by the random embedding sampler.
            plot: Whether to display the graph immediately after construction.

        Raises:
            ValueError: If the edge declarations are invalid or if a random
                embedding is requested for a planar graph.
            TypeError: If an edge or vertex declaration cannot be canonicalized.
        """
        super().__init__(
            edges,
            is_planar=is_planar,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
        )

        self._mapping = build_edge_mapping(self.edges)
        self._inverse_mapping = invert_edge_mapping(self._mapping)
        self._incidence = build_directed_connectivity(self.edges)
        self._dual_edges = build_dual_edges(self.edges)
        self._dual_adjacency = build_adjacency_from_edge_pairs(
            self.edges,
            self._dual_edges,
        )
        self._graph_positions = graph_positions(self.vertices, is_planar=self.is_planar)
        self._dual_positions = dual_positions(self.edges, is_planar=self.is_planar)

        self._minimal_loops = minimal_cycles(self.edges)
        self._dressed_minimal_loops = dress_loops(self._minimal_loops, self.edges)
        self._minimal_dual_loops = minimal_raw_cycles(
            [(first, second, 0) for first, second in self._dual_edges]
        )
        self._dressed_minimal_dual_loops = self._dress_dual_loops(
            self._minimal_dual_loops
        )

        self._signs = compute_graph_signs(
            self.edges,
            self.vertices,
            is_planar=self.is_planar,
        )
        self._dual_signs = compute_raw_signs(
            self.edges,
            [(first, second, 0) for first, second in self._dual_edges],
            self._dual_positions,
            is_planar=self.is_planar,
        )

        if plot:
            self.plot(show=True)

    @property
    def mapping(self) -> dict[Edge, int]:
        """Maps each canonical primal edge to its integer edge degree of freedom.

        The mapping is stable in input order and accepts the canonical edge
        orientation. Use :meth:`edge_to_index` when the lookup should also
        accept the reversed orientation.
        """
        return self._mapping

    @property
    def inverse_mapping(self) -> dict[int, Edge]:
        """Inverse of :attr:`mapping`.

        This is the natural conversion from a dual vertex label back to the
        primal edge represented by that dual vertex. The mapping is stable for
        a fixed canonical edge sequence and is used by lookup helpers that
        accept integer dual labels.
        """
        return self._inverse_mapping

    @property
    def signs(self) -> dict[Vertex, dict[Any, int]]:
        """Local orientation sign table for the primal graph.

        For planar graphs the table at a vertex is keyed by ordered edge pairs.
        For non-planar graphs it is keyed by ordered edge triples.
        """
        return self._signs

    @property
    def dual_signs(self) -> dict[Any, dict[Any, int]]:
        """Local orientation sign table for the line-graph dual.

        The table is computed with the same planar or non-planar convention as
        the primal graph. Dual vertices are represented by primal edges.
        """
        return self._dual_signs

    @property
    def dual_edges(self) -> list[tuple[Edge, Edge]]:
        """Edges of the line-graph dual as pairs of primal edges.

        A primal edge is a dual vertex. Two dual vertices are adjacent when
        their primal edges are incident to a common primal vertex.
        """
        return self._dual_edges

    @property
    def dual_adjacency(self) -> dict[Edge, list[Edge]]:
        """Adjacency lists of the line-graph dual.

        Keys and neighbors are primal edges acting as dual vertices. Neighbor
        lists are sorted deterministically so algorithms and tests receive a
        stable traversal order.
        """
        return self._dual_adjacency

    @property
    def graph_positions(self) -> dict[Vertex, tuple[float, ...]]:
        """Coordinates used for primal plotting and sign computations.

        Planar graphs expose ``(x, y)`` pairs. Non-planar graphs expose full
        ``(x, y, z)`` coordinate triples.
        """
        return self._graph_positions

    @property
    def dual_graph_positions(self) -> dict[Edge, tuple[float, ...]]:
        """Midpoint coordinates used for dual plotting and sign computations.

        Each key is a primal edge representing a dual vertex. The coordinate is
        the midpoint of that edge in either two or three dimensions.
        """
        return self._dual_positions

    @property
    def connectivity(self) -> dict[Vertex, dict[str, list[Edge]]]:
        """Incoming and outgoing primal edges at every vertex.

        Each vertex maps to an ``incoming`` list and an ``outgoing`` list. The
        lists preserve canonical edge order within each direction.
        """
        return self._incidence

    def minimal_loops(self) -> list[MinimalLoop]:
        """Returns all shortest primal loops.

        The algorithm preserves the neuraLQX convention from the old graph
        stack: if parallel-edge two-cycles exist, those are returned first and
        exclusively. Otherwise all girth cycles of the underlying simple graph
        are returned. Each loop is a ``Loop`` object exposing ``edges``,
        ``length``, ``vertices``, and optional ``dressing``.
        """
        return [loop.copy() for loop in self._minimal_loops]

    @property
    def n_minimal_loops(self) -> int:
        """Number of cached shortest primal loops."""
        return len(self._minimal_loops)

    def minimal_dual_loops(self, *, edge_rep: bool = False) -> list:
        """Returns all shortest loops in the line-graph dual.

        Args:
            edge_rep: When ``False`` the endpoints of dual edges are primal
                ``Edge`` objects. When ``True`` they are mapped to integer dual
                vertex indices using :meth:`edge_to_index`.
        """
        if not edge_rep:
            return [list(loop) for loop in self._minimal_dual_loops]

        out = []
        for loop in self._minimal_dual_loops:
            out.append(
                [
                    (
                        self.edge_to_index(first),
                        self.edge_to_index(second),
                        key,
                    )
                    for first, second, key in loop
                ]
            )
        return out

    def dressed_minimal_loops(self) -> list[DressedMinimalLoop]:
        """Returns shortest primal loops with creation/annihilation dressing.

        The returned objects are ``Loop`` instances whose ``dressing`` property
        is set. The dressing list has one item per loop edge and contains the
        edge action type and key.
        """
        return [loop.copy() for loop in self._dressed_minimal_loops]

    def dressed_minimal_dual_loops(self) -> list:
        """Returns shortest dual loops with creation/annihilation metadata.

        The returned nested lists are copies of the cached dual-loop entries,
        preserving caller isolation from internal graph state.
        """
        return [list(loop) for loop in self._dressed_minimal_dual_loops]

    def map_edge_to_dual_vertex(self, edge: Edge) -> int:
        """Maps a primal edge to its line-graph dual vertex index.

        Args:
            edge: Canonical or reversed primal edge to locate.

        Returns:
            Integer index of the dual vertex represented by ``edge``.
        """
        return self.edge_to_index(edge)

    def dual_vertex_to_edge(self, vertex: int) -> Edge:
        """Maps a line-graph dual vertex index back to a primal edge.

        Args:
            vertex: Integer dual vertex index.

        Returns:
            Canonical primal edge represented by ``vertex``.
        """
        return self.index_to_edge(vertex)

    def plot(
        self,
        *,
        show: bool = True,
        ax: Any | None = None,
        include_dual: bool = False,
        with_labels: bool = True,
        title: str | None = None,
        layout: str = "spring",
        seed: int | None = None,
        figsize: tuple[float, float] = (7.5, 6.2),
        dpi: int = 160,
        node_size: float = 980.0,
        edge_width: float = 3.2,
        node_color: str = "#ed6a5a",
        edge_color: str = "#ed6a5a",
        dual_color: str = "#7ebdc2",
        label_color: str = "#404040",
        label_font_size: float = 9.5,
        max_rad: float = 0.24,
        elev: float = 25.0,
        azim: float = 45.0,
        cmap: str = "viridis",
        reference_mesh: bool = True,
        depth_colorbar: bool = True,
    ) -> tuple[Any, Any]:
        """Plots a compact visualisation of the graph.

        Planar graphs are drawn as directed curved arrows with parallel edges
        fanned out symmetrically. Non-planar graphs are drawn in 3D with a
        depth-shaded colour treatment. By default, plotting uses a deterministic
        force layout for readability: seed ``42`` for planar graphs and seed
        ``90`` for non-planar graphs, matching the old neuraLQX plotting
        convention.

        Args:
            show: If ``True``, call ``matplotlib.pyplot.show`` before returning.
            ax: Optional existing matplotlib axes.
            include_dual: If ``True``, overlay the line-graph dual.
            with_labels: If ``True``, annotate vertices by their index.
            title: Optional axes title.
            layout: ``"spring"`` for deterministic visual layout or
                ``"coordinates"`` to use graph coordinates directly.
            seed: Optional layout seed. Defaults to ``42`` for planar and ``90``
                for non-planar spring layouts.
            figsize: Figure size used when ``ax`` is not supplied.
            dpi: Figure resolution used when ``ax`` is not supplied.
            node_size: Marker size for primal vertices.
            edge_width: Width of primal edges.
            node_color: Color used for primal vertices.
            edge_color: Color used for primal edges.
            dual_color: Color used for dual edges.
            label_color: Color used for labels.
            label_font_size: Font size used for vertex labels.
            max_rad: Maximum curvature used to fan out parallel planar edges.
            elev: 3D elevation angle in degrees.
            azim: 3D azimuth angle in degrees.
            cmap: Matplotlib colormap used for non-planar depth shading.
            reference_mesh: If ``True``, draw a faint 3D mesh for depth cues.
            depth_colorbar: If ``True``, add a projected-depth colorbar for 3D
                plots.

        Returns:
            The matplotlib ``(figure, axes)`` pair.
        """
        from .utils._plotting import plot_graph

        return plot_graph(
            self,
            show=show,
            ax=ax,
            include_dual=include_dual,
            with_labels=with_labels,
            title=title,
            layout=layout,
            seed=seed,
            figsize=figsize,
            dpi=dpi,
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            edge_color=edge_color,
            dual_color=dual_color,
            label_color=label_color,
            label_font_size=label_font_size,
            max_rad=max_rad,
            elev=elev,
            azim=azim,
            cmap=cmap,
            reference_mesh=reference_mesh,
            depth_colorbar=depth_colorbar,
        )

    def _dress_dual_loops(self, loops: list[list[tuple[Edge, Edge, int]]]) -> list:
        """Attaches creation/annihilation metadata to cached dual loops.

        Args:
            loops: Shortest dual loops represented as keyed directed dual
                edges, where each dual vertex is a primal edge.

        Returns:
            A list of dual-loop entries paired with dressing dictionaries.
            A directed dual edge matching the stored orientation is marked as a
            creation action. The reverse orientation is marked as annihilation.
        """
        stored = set(self._dual_edges)
        dressed = []
        for loop in loops:
            dressed_loop = []
            for first, second, key in loop:
                edge_type = "creation" if (first, second) in stored else "annihilation"
                dressing = {"type": edge_type, "key": key}
                dressed_loop.append(((first, second, key), dressing))
            dressed.append(dressed_loop)
        return dressed


__all__ = [
    "Graph",
]
