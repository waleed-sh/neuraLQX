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


"""Deterministic embedding helpers for graph coordinates.

The helpers in this module convert canonical vertex objects into plotting and
sign-computation coordinate dictionaries. They also provide reproducible
random embeddings for non-planar graphs when the caller wants coordinates that
are independent of the family construction layout.
"""

from __future__ import annotations

import random

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Vertex

from ._ordering import vertex_key


def randomize_vertices(
    edges: GraphEdges,
    *,
    mean: float = 0.0,
    std: float = 5.0,
    seed: int | str | bytes | None = 123,
) -> tuple[GraphEdges, dict[Vertex, Vertex]]:
    """Returns a deterministic random embedding for a non-planar graph.

    Args:
        edges: Canonical graph edges whose endpoints should be replaced.
        mean: Mean passed to the Gaussian coordinate sampler.
        std: Standard deviation passed to the Gaussian coordinate sampler.
        seed: Seed for the local pseudo-random generator.

    Returns:
        A pair containing embedded edges and the vertex mapping used to produce
        them.
    """
    rng = random.Random(seed)
    vertices = sorted(
        {vertex for edge in edges for vertex in edge.vertices},
        key=vertex_key,
    )
    mapping = {
        vertex: Vertex(
            rng.gauss(mean, std),
            rng.gauss(mean, std),
            rng.gauss(mean, std),
        )
        for vertex in vertices
    }
    embedded_edges = [
        Edge(mapping[edge.start], mapping[edge.end], edge.key) for edge in edges
    ]
    return embedded_edges, mapping


def graph_positions(
    vertices: list[Vertex],
    *,
    is_planar: bool,
) -> dict[Vertex, tuple[float, ...]]:
    """Returns vertex positions used by signs and plotting.

    Args:
        vertices: Canonical graph vertices.
        is_planar: Whether to emit two-dimensional planar positions.

    Returns:
        Mapping from each vertex to either ``(x, y)`` or ``(x, y, z)``.
    """
    if is_planar:
        return {vertex: (vertex.x, vertex.y) for vertex in vertices}
    return {vertex: vertex.as_tuple() for vertex in vertices}


def dual_positions(
    edges: GraphEdges,
    *,
    is_planar: bool,
) -> dict[Edge, tuple[float, ...]]:
    """Returns midpoint positions for line-graph dual vertices.

    Args:
        edges: Canonical primal edges whose midpoints represent dual vertices.
        is_planar: Whether to emit two-dimensional planar positions.

    Returns:
        Mapping from each primal edge to the midpoint coordinate of its
        endpoints.
    """
    if is_planar:
        return {
            edge: (
                0.5 * (edge.start.x + edge.end.x),
                0.5 * (edge.start.y + edge.end.y),
            )
            for edge in edges
        }
    return {
        edge: (
            0.5 * (edge.start.x + edge.end.x),
            0.5 * (edge.start.y + edge.end.y),
            0.5 * (edge.start.z + edge.end.z),
        )
        for edge in edges
    }


__all__ = [
    "dual_positions",
    "graph_positions",
    "randomize_vertices",
]
