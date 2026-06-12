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


"""Line-graph dual construction helpers.

The functions in this module derive incident-edge structures from canonical
primal edges. They are shared by the public ``Graph`` class when constructing
connectivity tables and the line-graph dual used by loop and sign algorithms.
"""

from itertools import combinations

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Vertex

from ._ordering import edge_key


def build_incidence(edges: GraphEdges) -> dict[Vertex, list[Edge]]:
    """Returns incident edges for every vertex.

    Args:
        edges: Canonical primal edges.

    Returns:
        Mapping from each vertex to the primal edges incident to it. Edge lists
        preserve canonical graph order.
    """
    incidence: dict[Vertex, list[Edge]] = {}
    for edge in edges:
        incidence.setdefault(edge.start, []).append(edge)
        incidence.setdefault(edge.end, []).append(edge)
    return incidence


def build_directed_connectivity(
    edges: GraphEdges,
) -> dict[Vertex, dict[str, list[Edge]]]:
    """Returns incoming and outgoing edge lists for every vertex.

    Args:
        edges: Canonical directed primal edges.

    Returns:
        Mapping from each vertex to ``incoming`` and ``outgoing`` edge lists,
        each preserving canonical graph order.
    """
    connectivity: dict[Vertex, dict[str, list[Edge]]] = {}
    for edge in edges:
        connectivity.setdefault(edge.start, {"incoming": [], "outgoing": []})
        connectivity.setdefault(edge.end, {"incoming": [], "outgoing": []})
        connectivity[edge.start]["outgoing"].append(edge)
        connectivity[edge.end]["incoming"].append(edge)
    return connectivity


def build_dual_edges(edges: GraphEdges) -> list[tuple[Edge, Edge]]:
    """Returns line-graph dual edges as pairs of primal edges.

    Args:
        edges: Canonical primal edges.

    Returns:
        Sorted pairs of primal edges. Each pair indicates that the two primal
        edges meet at at least one primal vertex.
    """
    incidence = build_incidence(edges)
    seen: set[frozenset[Edge]] = set()
    dual_edges: list[tuple[Edge, Edge]] = []

    for vertex in incidence:
        for first, second in combinations(incidence[vertex], 2):
            pair = frozenset((first, second))
            if pair in seen:
                continue
            seen.add(pair)
            ordered = tuple(sorted((first, second), key=edge_key))
            dual_edges.append((ordered[0], ordered[1]))

    return sorted(dual_edges, key=lambda pair: (edge_key(pair[0]), edge_key(pair[1])))


def build_adjacency_from_edge_pairs(
    vertices: list[Edge],
    edges: list[tuple[Edge, Edge]],
) -> dict[Edge, list[Edge]]:
    """Builds deterministic adjacency lists from edge pairs.

    Args:
        vertices: Vertex labels to include in the adjacency mapping.
        edges: Undirected edge pairs between those labels.

    Returns:
        Mapping from each supplied vertex label to sorted neighbor labels.
    """
    adjacency = {vertex: [] for vertex in vertices}
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    return {
        vertex: sorted(neighbours, key=edge_key)
        for vertex, neighbours in adjacency.items()
    }


__all__ = [
    "build_adjacency_from_edge_pairs",
    "build_directed_connectivity",
    "build_dual_edges",
    "build_incidence",
]
