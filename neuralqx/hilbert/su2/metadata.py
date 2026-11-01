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


"""Compiled graph incidence metadata for SU(2) spin networks.

SU(2) constraints and proposals repeatedly need graph incidence in flat edge
order. This module compiles that information once into immutable metadata and
provides padded JAX arrays for vectorized kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.graph import AbstractGraph


@dataclass(frozen=True)
class SU2GraphMetadata:
    """Graph incidence metadata in the flat SU(2) edge ordering.

    Attributes:
        n_edges: Number of graph edges.
        n_vertices: Number of graph vertices.
        edge_tokens: Canonical edge objects in flat edge order.
        vertex_ids: Canonical graph vertices in layout order.
        vertex_index: Mapping from vertex id to layout position.
        incident_edges_by_vertex: Incident edge indices for every vertex.
        incident_orient_by_vertex: Orientation signs for incident edges.
        valence_by_vertex: Incident-edge count for every vertex.
    """

    n_edges: int
    """Number of graph edges."""

    n_vertices: int
    """Number of graph vertices."""

    edge_tokens: tuple[Any, ...]
    """Canonical edge objects in flat edge order."""

    vertex_ids: tuple[Any, ...]
    """Canonical graph vertices in layout order."""

    vertex_index: dict[Any, int]
    """Mapping from vertex id to layout position."""

    incident_edges_by_vertex: tuple[tuple[int, ...], ...]
    """Incident edge indices for every vertex."""

    incident_orient_by_vertex: tuple[tuple[int, ...], ...]
    """Orientation signs for incident edges."""

    valence_by_vertex: tuple[int, ...]
    """Incident-edge count for every vertex."""

    @property
    def max_valence(self) -> int:
        """Maximum graph valence across all vertices."""
        return max(self.valence_by_vertex, default=0)

    @property
    def incident_edges_padded(self) -> jax.Array:
        """Padded incident-edge index table."""
        arr = -jnp.ones((self.n_vertices, self.max_valence), dtype=jnp.int32)
        for vertex, edges in enumerate(self.incident_edges_by_vertex):
            if edges:
                arr = arr.at[vertex, : len(edges)].set(
                    jnp.asarray(edges, dtype=jnp.int32)
                )
        return arr

    @property
    def incident_orient_padded(self) -> jax.Array:
        """Padded incident-edge orientation table."""
        arr = jnp.zeros((self.n_vertices, self.max_valence), dtype=jnp.int32)
        for vertex, signs in enumerate(self.incident_orient_by_vertex):
            if signs:
                arr = arr.at[vertex, : len(signs)].set(
                    jnp.asarray(signs, dtype=jnp.int32)
                )
        return arr

    @property
    def incident_mask(self) -> jax.Array:
        """Boolean mask for valid padded incidence entries."""
        arr = jnp.zeros((self.n_vertices, self.max_valence), dtype=jnp.bool_)
        for vertex, edges in enumerate(self.incident_edges_by_vertex):
            if edges:
                arr = arr.at[vertex, : len(edges)].set(True)
        return arr

    @property
    def valence(self) -> jax.Array:
        """Vertex valence array in layout order."""
        return jnp.asarray(self.valence_by_vertex, dtype=jnp.int32)


def compile_su2_graph_metadata(graph: AbstractGraph) -> SU2GraphMetadata:
    """Compiles deterministic SU(2) incidence metadata from a graph.

    Args:
        graph: Graph whose edge and vertex order define the spin-network
            layout.

    Returns:
        Immutable SU(2) graph metadata.
    """
    edge_tokens = tuple(graph.index_to_edge(edge) for edge in range(graph.n_edges))
    vertex_ids = tuple(graph.vertices)
    vertex_index = {vertex: idx for idx, vertex in enumerate(vertex_ids)}
    incident_edges = [[] for _ in vertex_ids]
    incident_orient = [[] for _ in vertex_ids]

    for edge_index, edge in enumerate(edge_tokens):
        start = vertex_index[edge.start]
        end = vertex_index[edge.end]
        incident_edges[start].append(edge_index)
        incident_orient[start].append(+1)
        incident_edges[end].append(edge_index)
        incident_orient[end].append(-1)

    incident_edges_tuple = tuple(tuple(row) for row in incident_edges)
    incident_orient_tuple = tuple(tuple(row) for row in incident_orient)
    valence = tuple(len(row) for row in incident_edges_tuple)
    return SU2GraphMetadata(
        n_edges=graph.n_edges,
        n_vertices=len(vertex_ids),
        edge_tokens=edge_tokens,
        vertex_ids=vertex_ids,
        vertex_index=vertex_index,
        incident_edges_by_vertex=incident_edges_tuple,
        incident_orient_by_vertex=incident_orient_tuple,
        valence_by_vertex=valence,
    )


__all__ = ["SU2GraphMetadata", "compile_su2_graph_metadata"]
