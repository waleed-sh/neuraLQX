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


"""Deterministic ordering keys for graph algorithms.

The graph package uses these structural sort keys whenever dictionaries, sets,
or unordered endpoint pairs must be traversed deterministically. This keeps
cycle enumeration, dual construction, and plotting stable across Python runs.
"""

from typing import Any

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import Vertex


def vertex_key(vertex: Vertex) -> tuple[float, float, float]:
    """Returns a deterministic sort key for a vertex.

    Args:
        vertex: Canonical graph vertex.

    Returns:
        Coordinate triple ordered by x, y, and z.
    """
    return (vertex.x, vertex.y, vertex.z)


def edge_key(
    edge: Edge,
) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    """Returns a deterministic sort key for a keyed edge.

    Args:
        edge: Canonical keyed graph edge.

    Returns:
        Tuple containing start-vertex key, end-vertex key, and parallel-edge
        key.
    """
    return (vertex_key(edge.start), vertex_key(edge.end), edge.key)


def object_key(value: Any) -> tuple[Any, ...]:
    """Returns a stable structural key for graph algorithm values.

    Args:
        value: Vertex, edge, tuple, or fallback object to compare.

    Returns:
        Structural tuple suitable for sorting heterogeneous graph labels.
    """
    if isinstance(value, Vertex):
        return ("vertex", *vertex_key(value))
    if isinstance(value, Edge):
        return ("edge", edge_key(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(object_key(item) for item in value))
    return (type(value).__name__, repr(value))


def unordered_pair_key(
    left: Any,
    right: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Returns an orientation-independent key for an endpoint pair.

    Args:
        left: First endpoint-like value.
        right: Second endpoint-like value.

    Returns:
        Pair of structural keys sorted so ``(left, right)`` and
        ``(right, left)`` produce the same identity.
    """
    left_key = object_key(left)
    right_key = object_key(right)
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)


__all__ = [
    "edge_key",
    "object_key",
    "unordered_pair_key",
    "vertex_key",
]
