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


"""Orientation-sign utilities for primal and dual graph structures.

The graph API uses local sign tables to encode orientation data at each
vertex. Planar graphs use ordered pairs of incident edges and clockwise
ordering in the plane. Non-planar graphs use ordered triples and the sign of
the 3D tangent determinant.
"""

import math

from collections import defaultdict
from collections.abc import Hashable
from itertools import combinations
from itertools import permutations
from typing import Any

from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Vertex

from ._cycles import RawEdge
from ._ordering import object_key
from ._ordering import unordered_pair_key

_SMALL_ANGLE = 1.0e-2
_TOLERANCE = 1.0e-14


def compute_graph_signs(
    edges: GraphEdges,
    vertices: list[Vertex],
    *,
    is_planar: bool,
) -> dict[Vertex, dict[Any, int]]:
    """Computes local orientation sign tables for a primal graph.

    Args:
        edges: Canonical keyed graph edges.
        vertices: Canonical graph vertices.
        is_planar: Whether to compute planar pair signs or non-planar triple
            signs.

    Returns:
        Mapping from each vertex to a local sign table keyed by canonical
        ``Edge`` objects.
    """
    raw_edges = [(edge.start, edge.end, edge.key) for edge in edges]
    positions = {
        vertex: (vertex.x, vertex.y) if is_planar else vertex.as_tuple()
        for vertex in vertices
    }
    raw_signs = compute_raw_signs(
        vertices,
        raw_edges,
        positions,
        is_planar=is_planar,
    )
    edge_lookup = {raw: edge for raw, edge in zip(raw_edges, edges)}

    out: dict[Vertex, dict[Any, int]] = {}
    for vertex, table in raw_signs.items():
        out[vertex] = {}
        for key, sign in table.items():
            if is_planar:
                first, second = key
                out[vertex][(edge_lookup[first], edge_lookup[second])] = sign
            else:
                first, second, third = key
                out[vertex][
                    (edge_lookup[first], edge_lookup[second], edge_lookup[third])
                ] = sign
    return out


def compute_raw_signs(
    vertices: list[Hashable],
    edges: list[RawEdge],
    positions: dict[Hashable, tuple[float, ...]],
    *,
    is_planar: bool,
) -> dict[Hashable, dict[Any, int]]:
    """Computes local sign tables for generic keyed graph edges.

    Args:
        vertices: Vertex labels to include in the result.
        edges: Raw keyed edge triples.
        positions: Coordinate mapping for all incident vertices.
        is_planar: Whether to compute planar pair signs or non-planar triple
            signs.

    Returns:
        Mapping from each vertex label to its local orientation sign table.
    """
    incidence = _incidence(vertices, edges)
    if is_planar:
        return {
            vertex: _planar_vertex_signs(vertex, incident, positions)
            for vertex, incident in incidence.items()
        }
    return {
        vertex: _nonplanar_vertex_signs(vertex, incident, positions, edges)
        for vertex, incident in incidence.items()
    }


def _incidence(
    vertices: list[Hashable],
    edges: list[RawEdge],
) -> dict[Hashable, list[RawEdge]]:
    """Builds sorted incident-edge lists for raw graph data.

    Args:
        vertices: Vertex labels that should appear in the output.
        edges: Raw keyed edge triples.

    Returns:
        Mapping from vertex label to sorted incident raw edges.
    """
    incidence: dict[Hashable, list[RawEdge]] = {vertex: [] for vertex in vertices}
    for edge in edges:
        start, end, _ = edge
        incidence.setdefault(start, []).append(edge)
        incidence.setdefault(end, []).append(edge)
    return {
        vertex: sorted(incident, key=_raw_edge_key)
        for vertex, incident in incidence.items()
    }


def _planar_vertex_signs(
    vertex: Hashable,
    edges: list[RawEdge],
    positions: dict[Hashable, tuple[float, ...]],
) -> dict[tuple[RawEdge, RawEdge], int]:
    """Computes ordered pair signs at one planar vertex.

    Args:
        vertex: Vertex label whose incident edges are being evaluated.
        edges: Raw edges incident to ``vertex``.
        positions: Two-dimensional coordinate mapping.

    Returns:
        Sign table keyed by ordered pairs of incident raw edges.
    """
    signs: dict[tuple[RawEdge, RawEdge], int] = {}
    for first, second in permutations(edges, 2):
        tangent_a = _tangent(first, vertex, positions, dimensions=2)
        tangent_b = _tangent(second, vertex, positions, dimensions=2)
        raw = _clockwise_order_sign(tangent_a, tangent_b)
        direction = _direction_factor(first, vertex) * _direction_factor(second, vertex)
        signs[(first, second)] = raw * direction
    return signs


def _nonplanar_vertex_signs(
    vertex: Hashable,
    edges: list[RawEdge],
    positions: dict[Hashable, tuple[float, ...]],
    all_edges: list[RawEdge],
) -> dict[tuple[RawEdge, RawEdge, RawEdge], int]:
    """Computes ordered triple signs at one non-planar vertex.

    Args:
        vertex: Vertex label whose incident edges are being evaluated.
        edges: Raw edges incident to ``vertex``.
        positions: Three-dimensional coordinate mapping.
        all_edges: Complete raw edge list used to detect parallel duplicates.

    Returns:
        Sign table keyed by ordered triples of incident raw edges.
    """
    signs: dict[tuple[RawEdge, RawEdge, RawEdge], int] = {}
    duplicate_groups = _duplicate_groups(all_edges)
    tangents = {
        edge: _perturbed_tangent(
            edge,
            vertex,
            positions,
            duplicate_groups=duplicate_groups,
        )
        for edge in edges
    }

    for edge_set in combinations(edges, 3):
        for ordered in permutations(edge_set, 3):
            determinant = _det3(
                tangents[ordered[0]],
                tangents[ordered[1]],
                tangents[ordered[2]],
            )
            signs[ordered] = _sign_scalar(determinant)
    return signs


def _tangent(
    edge: RawEdge,
    vertex: Hashable,
    positions: dict[Hashable, tuple[float, ...]],
    *,
    dimensions: int,
) -> tuple[float, ...]:
    """Returns the unit tangent of an incident edge at a vertex.

    Args:
        edge: Raw edge incident to ``vertex``.
        vertex: Vertex label at which the tangent is measured.
        positions: Coordinate mapping for edge endpoints.
        dimensions: Number of coordinate dimensions to include.

    Returns:
        Unit vector pointing from ``vertex`` toward the opposite endpoint. A
        zero vector is returned when the endpoints coincide numerically.

    Raises:
        ValueError: If ``edge`` is not incident to ``vertex``.
    """
    start, end, _ = edge
    if vertex == start:
        other = end
    elif vertex == end:
        other = start
    else:
        raise ValueError(f"Edge {edge!r} is not incident to vertex {vertex!r}.")

    origin = positions[vertex]
    target = positions[other]
    vector = tuple(float(target[i] - origin[i]) for i in range(dimensions))
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= _TOLERANCE:
        return tuple(0.0 for _ in range(dimensions))
    return tuple(component / norm for component in vector)


def _perturbed_tangent(
    edge: RawEdge,
    vertex: Hashable,
    positions: dict[Hashable, tuple[float, ...]],
    *,
    duplicate_groups: dict[tuple[Any, Any], list[RawEdge]],
) -> tuple[float, float, float]:
    """Returns a tangent with tiny deterministic perturbations for duplicates.

    Args:
        edge: Raw edge incident to ``vertex``.
        vertex: Vertex label at which the tangent is measured.
        positions: Three-dimensional coordinate mapping.
        duplicate_groups: Raw edges grouped by unordered endpoint pair.

    Returns:
        Unit tangent vector. Parallel duplicate edges receive small rotations
        so determinant signs remain deterministic.
    """
    base = _tangent(edge, vertex, positions, dimensions=3)
    if all(abs(component) <= _TOLERANCE for component in base):
        return (0.0, 0.0, 0.0)
    siblings = duplicate_groups[unordered_pair_key(edge[0], edge[1])]
    if len(siblings) <= 1:
        return base  # type: ignore[return-value]

    ordered = sorted(siblings, key=_raw_edge_key)
    index = ordered.index(edge)
    angle = _SMALL_ANGLE * (index - (len(ordered) - 1) / 2.0)
    axis = _perpendicular_axis(base)
    return _normalize(_rotate_about_axis(base, axis, angle))


def _duplicate_groups(edges: list[RawEdge]) -> dict[tuple[Any, Any], list[RawEdge]]:
    """Groups raw edges by unordered endpoint pair.

    Args:
        edges: Raw keyed edge triples.

    Returns:
        Mapping from unordered endpoint identity to raw edges connecting that
        pair.
    """
    groups: dict[tuple[Any, Any], list[RawEdge]] = defaultdict(list)
    for edge in edges:
        groups[unordered_pair_key(edge[0], edge[1])].append(edge)
    return groups


def _direction_factor(edge: RawEdge, vertex: Hashable) -> int:
    """Returns the orientation factor of an edge at a vertex.

    Args:
        edge: Raw edge incident to ``vertex``.
        vertex: Vertex label to evaluate.

    Returns:
        ``1`` when the edge starts at ``vertex`` and ``-1`` otherwise.
    """
    return 1 if edge[0] == vertex else -1


def _clockwise_order_sign(first: tuple[float, ...], second: tuple[float, ...]) -> int:
    """Returns the planar sign from clockwise angular ordering.

    Args:
        first: First two-dimensional tangent vector.
        second: Second two-dimensional tangent vector.

    Returns:
        ``-1`` when ``first`` appears before ``second`` in clockwise angular
        order, ``1`` otherwise, and ``0`` when either tangent is degenerate.
    """
    if _norm(first) <= _TOLERANCE or _norm(second) <= _TOLERANCE:
        return 0
    first_angle = (-math.atan2(first[1], first[0])) % (2.0 * math.pi)
    second_angle = (-math.atan2(second[1], second[0])) % (2.0 * math.pi)
    return -1 if first_angle < second_angle else 1


def _sign_scalar(value: float) -> int:
    """Converts a floating-point scalar to a sign with tolerance.

    Args:
        value: Scalar determinant or orientation value.

    Returns:
        ``1`` for positive values, ``-1`` for negative values, and ``0`` when
        the value is within tolerance of zero.
    """
    if abs(value) <= _TOLERANCE:
        return 0
    return 1 if value > 0 else -1


def _det3(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    """Returns the determinant of three 3D column vectors.

    Args:
        first: First vector.
        second: Second vector.
        third: Third vector.

    Returns:
        Scalar triple product used for non-planar orientation signs.
    """
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - second[0] * (first[1] * third[2] - first[2] * third[1])
        + third[0] * (first[1] * second[2] - first[2] * second[1])
    )


def _norm(vector: tuple[float, ...]) -> float:
    """Returns the Euclidean norm of a vector.

    Args:
        vector: Numeric vector.

    Returns:
        Square-root of the sum of squared components.
    """
    return math.sqrt(sum(component * component for component in vector))


def _perpendicular_axis(vector: tuple[float, ...]) -> tuple[float, float, float]:
    """Builds a unit axis perpendicular to a 3D vector.

    Args:
        vector: Three-dimensional vector.

    Returns:
        Unit vector perpendicular to ``vector``.

    Raises:
        ValueError: If no stable perpendicular axis can be constructed.
    """
    candidates = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for candidate in candidates:
        axis = _cross3(vector, candidate)
        norm = math.sqrt(sum(component * component for component in axis))
        if norm > _TOLERANCE:
            return tuple(component / norm for component in axis)
    raise ValueError(f"Could not build perpendicular axis for vector {vector!r}.")


def _rotate_about_axis(
    vector: tuple[float, ...],
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[float, float, float]:
    """Rotates a vector around a unit axis.

    Args:
        vector: Three-dimensional vector to rotate.
        axis: Unit rotation axis.
        angle: Rotation angle in radians.

    Returns:
        Rotated vector computed with Rodrigues' rotation formula.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = _cross3(axis, vector)
    dot = sum(axis[i] * vector[i] for i in range(3))
    return (
        vector[0] * cos_a + cross[0] * sin_a + axis[0] * dot * (1.0 - cos_a),
        vector[1] * cos_a + cross[1] * sin_a + axis[1] * dot * (1.0 - cos_a),
        vector[2] * cos_a + cross[2] * sin_a + axis[2] * dot * (1.0 - cos_a),
    )


def _cross3(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> tuple[float, float, float]:
    """Returns the three-dimensional cross product.

    Args:
        first: First vector with at least three components.
        second: Second vector with at least three components.

    Returns:
        Cross product using the first three components of each vector.
    """
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalizes a nonzero three-dimensional vector.

    Args:
        vector: Vector to normalize.

    Returns:
        Unit vector with the same direction.

    Raises:
        ValueError: If ``vector`` has zero norm within tolerance.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= _TOLERANCE:
        raise ValueError("Cannot normalize a zero vector.")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _raw_edge_key(edge: RawEdge) -> tuple[Any, ...]:
    """Returns a deterministic structural key for a raw edge.

    Args:
        edge: Raw keyed edge triple.

    Returns:
        Tuple key combining start endpoint, end endpoint, and parallel-edge key.
    """
    return (object_key(edge[0]), object_key(edge[1]), edge[2])


__all__ = [
    "compute_graph_signs",
    "compute_raw_signs",
]
