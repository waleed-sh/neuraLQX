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


"""Graph-to-cluster construction for exchange transition rules."""

from __future__ import annotations

from collections import deque

from neuralqx.graph import AbstractGraph
from neuralqx.utils.typing import ExchangeEntity


def clusters_from_graph(
    graph: AbstractGraph | None,
    d_max: int,
    entity: ExchangeEntity,
) -> tuple[tuple[int, int], ...]:
    """Derive flat exchange clusters from a graph-distance cutoff."""
    if graph is None:
        raise ValueError("graph is required to derive exchange clusters.")
    if entity == "vertex":
        adjacency = vertex_adjacency(graph)
        n_sites = graph.n_vertices
    elif entity == "edge":
        adjacency = edge_adjacency(graph)
        n_sites = graph.n_edges
    else:
        raise ValueError("entity must be 'vertex' or 'edge'.")

    clusters: list[tuple[int, int]] = []
    for source in range(n_sites):
        distances = shortest_distances(adjacency, source, d_max)
        for target, distance in distances.items():
            if source < target and 0 < distance <= d_max:
                clusters.append((source, target))
    return tuple(clusters)


def vertex_adjacency(graph: AbstractGraph) -> tuple[tuple[int, ...], ...]:
    """Return vertex adjacency as integer indices."""
    vertex_to_index = {vertex: idx for idx, vertex in enumerate(graph.vertices)}
    adjacency: list[set[int]] = [set() for _ in graph.vertices]
    for edge in graph.edges:
        i = vertex_to_index[edge.start]
        j = vertex_to_index[edge.end]
        adjacency[i].add(j)
        adjacency[j].add(i)
    return tuple(tuple(sorted(row)) for row in adjacency)


def edge_adjacency(graph: AbstractGraph) -> tuple[tuple[int, ...], ...]:
    """Return line-graph adjacency as edge indices."""
    adjacency: list[set[int]] = [set() for _ in graph.edges]
    for edge, neighbours in graph.dual_adjacency.items():
        i = graph.edge_to_index(edge)
        for neighbour in neighbours:
            adjacency[i].add(graph.edge_to_index(neighbour))
    return tuple(tuple(sorted(row)) for row in adjacency)


def shortest_distances(
    adjacency: tuple[tuple[int, ...], ...],
    source: int,
    d_max: int,
) -> dict[int, int]:
    """Breadth-first distances from ``source`` truncated at ``d_max``."""
    distances = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        current = queue.popleft()
        if distances[current] >= d_max:
            continue
        for neighbour in adjacency[current]:
            if neighbour in distances:
                continue
            distances[neighbour] = distances[current] + 1
            queue.append(neighbour)
    return distances


__all__ = [
    "clusters_from_graph",
    "edge_adjacency",
    "shortest_distances",
    "vertex_adjacency",
]
