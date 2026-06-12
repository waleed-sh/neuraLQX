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


"""Validation and canonicalisation helpers for graph inputs.

The public graph constructors accept compact edge declarations. This module
normalizes those declarations into canonical ``Edge`` objects, assigns
parallel-edge keys when needed, and raises clear errors for malformed input.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import EdgeLike
from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import GraphEdgesLike
from neuralqx.utils.typing import Vertex


def validate_vertices(edges: GraphEdgesLike) -> None:
    """Validates that every edge declaration can be canonicalized.

    Args:
        edges: Edge declarations supplied by a graph constructor.

    Raises:
        TypeError: If any edge or vertex cannot be converted to canonical graph
            objects.
    """
    for edge_index, edge in enumerate(edges):
        try:
            Edge.from_like(edge)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Invalid edge at index {edge_index}: {exc}") from exc


def ensure_edge_keys(edges: GraphEdgesLike) -> GraphEdges:
    """Returns canonical graph edges with integer parallel-edge keys.

    Unkeyed inputs ``(start, end)`` receive keys ``0, 1, 2, ...`` within each
    unordered pair of endpoints, so repeated physical links are explicit while
    the user's construction syntax stays compact.

    Args:
        edges: Keyed or unkeyed edge declarations. Mixing keyed and unkeyed
            declarations is rejected.

    Returns:
        Canonical ``Edge`` objects with explicit integer keys.

    Raises:
        TypeError: If edge declarations mix keyed and unkeyed forms or have an
            unsupported shape.
    """
    raw_edges = list(edges)
    if not raw_edges:
        return []

    edge_kinds = [_edge_kind(edge, index=index) for index, edge in enumerate(raw_edges)]
    if all(kind == "keyed" for kind in edge_kinds):
        return [Edge.from_like(edge) for edge in raw_edges]

    if any(kind == "keyed" for kind in edge_kinds):
        msg = "Graph edges must be either all keyed or all unkeyed, not mixed."
        raise TypeError(msg)

    next_key: defaultdict[frozenset[Vertex], int] = defaultdict(int)
    keyed_edges: list[Edge] = []

    for edge in raw_edges:
        unkeyed_edge = Edge.from_like(edge)
        unordered_edge = frozenset(unkeyed_edge.vertices)
        key = next_key[unordered_edge]
        next_key[unordered_edge] += 1

        keyed_edges.append(Edge(unkeyed_edge.start, unkeyed_edge.end, key))

    return keyed_edges


def _edge_kind(edge: EdgeLike, *, index: int) -> Literal["keyed", "unkeyed"]:
    """Classifies an edge declaration as keyed or unkeyed.

    Args:
        edge: Edge-like object to classify.
        index: Position of ``edge`` in the caller's edge sequence.

    Returns:
        ``"keyed"`` for ``Edge`` objects or three-item declarations and
        ``"unkeyed"`` for two-item declarations.

    Raises:
        TypeError: If the edge shape is unsupported.
    """
    if isinstance(edge, Edge):
        return "keyed"
    if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)):
        raise TypeError(
            f"Edge at index {index} must be an Edge, ``(start, end)``, "
            f"or ``(start, end, key)``; got {edge!r}."
        )
    if len(edge) == 2:
        return "unkeyed"
    if len(edge) == 3:
        return "keyed"
    raise TypeError(f"Edge at index {index} must have length 2 or 3; got {len(edge)}.")


__all__ = [
    "validate_vertices",
    "ensure_edge_keys",
]
