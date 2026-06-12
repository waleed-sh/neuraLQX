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


"""Edge-index mapping helpers for canonical graph edges.

These helpers keep conversion between primal edge objects and integer degrees
of freedom small, deterministic, and shared across graph implementations.
"""

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import GraphEdges


def build_edge_mapping(edges: GraphEdges) -> dict[Edge, int]:
    """Maps canonical graph edges to input-order integer indices.

    Args:
        edges: Canonical keyed graph edges.

    Returns:
        Mapping from each edge to its zero-based position in ``edges``.
    """
    return {edge: index for index, edge in enumerate(edges)}


def reverse_edge(edge: Edge) -> Edge:
    """Returns an edge with start and end swapped.

    Args:
        edge: Canonical edge to reverse.

    Returns:
        New edge with the same parallel-edge key and opposite orientation.
    """
    return Edge(edge.end, edge.start, edge.key)


def edge_to_index(edge: Edge, mapping: dict[Edge, int]) -> int | None:
    """Looks up an edge index while accepting either orientation.

    Args:
        edge: Edge to locate.
        mapping: Canonical edge-to-index mapping.

    Returns:
        The mapped index when either ``edge`` or its reverse exists. Returns
        ``None`` when the edge is absent.
    """
    return mapping.get(edge, mapping.get(reverse_edge(edge)))


def invert_edge_mapping(mapping: dict[Edge, int]) -> dict[int, Edge]:
    """Returns the inverse edge-index mapping.

    Args:
        mapping: Edge-to-index mapping.

    Returns:
        Index-to-edge mapping with the same entries.
    """
    return {index: edge for edge, index in mapping.items()}


__all__ = [
    "build_edge_mapping",
    "edge_to_index",
    "invert_edge_mapping",
    "reverse_edge",
]
