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


"""Graph helper routines for prebuilt physical operators."""

from __future__ import annotations

from itertools import combinations
from itertools import permutations
from typing import Any

from neuralqx.graph import AbstractGraph
from neuralqx.graph import HalfLadderGraph
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import Vertex


def graph_vertex(graph: AbstractGraph, vertex: Any) -> Vertex:
    r"""Resolve the vertex notation accepted by prebuilt operator factories.

    An integer within the vertex sequence is a position, not a coordinate label.
    Other values are passed to `Vertex.from_like`. This conversion does not
    assert graph membership. The caller's incidence or stencil checks establish
    whether the resulting vertex participates in the requested construction.

    Args:
        graph: Graph whose vertex order defines positional indices.
        vertex: A Vertex, a coordinate-like value, or a positional integer.

    Returns:
        result (Vertex): Vertex object used for graph lookups.

    Raises:
        TypeError: If `vertex` is a boolean or cannot be converted to a vertex.
    """
    # Reject bool explicitly because it otherwise behaves as an integer index.
    if isinstance(vertex, bool):
        # Report an ambiguous vertex argument before indexing the graph.
        raise TypeError("Vertex index must not be a bool.")
    # Integers in range select the graph vertex sequence.
    if isinstance(vertex, int):
        # Freeze the sequence for a consistent bounds check and lookup.
        vertices = tuple(graph.vertices)
        # Only interpret the integer as a position when that position exists.
        if 0 <= vertex < len(vertices):
            # Reuse the graph object so incidence and sign-table keys agree.
            return vertices[vertex]
    # Delegate coordinate and Vertex inputs to the shared conversion rule.
    return Vertex.from_like(vertex)


def incident_edge_tuples(
    graph: AbstractGraph,
    vertex: Vertex,
    order: int,
    *,
    signed_only: bool = True,
) -> tuple[tuple[Edge, ...], ...]:
    r"""Collect ordered edge tuples used by the vertex volume kernels.

    The graph's orientation table normally contains all ordered pairs in two
    spatial dimensions or all ordered triples in three. Keeping that multiplicity
    matters because the volume normalization includes the full ordered sum.
    `signed_only` filters zero entries from an available table. If no qualifying
    entries remain, all ordered incident tuples are returned as a fallback.
    Callers must still look up their signs and omit zero contributions.

    Args:
        graph: Graph supplying incident edges and geometric orientation signs.
        vertex: Resolved vertex at which the volume is evaluated.
        order: Number of distinct edges per tuple, usually two or three.
        signed_only: Exclude zero-sign entries when reading the orientation table.

    Returns:
        result (tuple[tuple[Edge, ...], ...]): Ordered tuples, preserving table
            order when possible and incident-edge permutation order otherwise.
            Fewer than `order` incident edges give an empty tuple.
    """
    # A missing vertex table is handled by the incidence fallback below.
    table = graph.signs.get(vertex, {})
    # Keep only Edge-keyed entries of the requested arity and sign class.
    signed = tuple(
        tuple(key)
        for key, sign in table.items()
        if isinstance(key, tuple)
        and len(key) == order
        and all(isinstance(edge, Edge) for edge in key)
        and (not signed_only or int(sign) != 0)
    )
    # An available filtered table is the authoritative ordered collection.
    if signed:
        # Preserve its order and multiplicity exactly.
        return signed
    # Resolve incidence when the table has no usable entries.
    edges = tuple(graph.edges_at(vertex))
    # Enumerate distinct-edge permutations, leaving sign validation to the caller.
    return tuple(permutations(edges, order))


def orientation_sign(
    graph: AbstractGraph, vertex: Vertex, edges: tuple[Edge, ...]
) -> int:
    r"""Read the geometric orientation of an incident pair or triple.

    Current graphs key their sign tables by tuples of `Edge` objects. Raw edge
    tuples and their string representations are also accepted for compatibility
    with older graph data. Missing entries have sign zero, so callers can omit
    them from the volume sum. This lookup is not the combinatorial permutation
    sign used to assemble the Thiemann triangulation.

    Args:
        graph: Graph containing the geometric sign table.
        vertex: Vertex at which the tangents are compared.
        edges: Ordered incident pair or triple in stored graph orientation.

    Returns:
        result (int): The stored sign, normally -1, 0, or +1.
    """
    # A vertex with no orientation data has no signed contribution.
    table = graph.signs.get(vertex, {})
    # Prefer the typed keys used by current graph implementations.
    if edges in table:
        # Return a Python scalar suitable for static kernel metadata.
        return int(table[edges])
    # Convert to the legacy endpoint-and-key representation.
    raw = tuple(edge.as_tuple() for edge in edges)
    # Accept older tables that retained tuple keys.
    if raw in table:
        # Normalize the legacy value to the same scalar type.
        return int(table[raw])
    # Some upstream tables serialized the entire ordered tuple as a string.
    if str(raw) in table:
        # Read that representation without changing the edge ordering.
        return int(table[str(raw)])
    # Absent or unsupported entries do not contribute to the volume.
    return 0


def edge_emanates_from(edge: Edge, vertex: Vertex) -> bool:
    r"""Determine whether a stored edge points away from an incident vertex.

    This direction relates charge shifts in the graph's stored orientation to
    a segment oriented locally at the vertex. An incoming stored edge has the
    opposite charge shift. A self-loop satisfies the start test and is treated
    as outgoing by this helper.

    Args:
        edge: Stored or explicitly oriented graph edge.
        vertex: Endpoint at which the direction is required.

    Returns:
        result (bool): True for an edge whose start equals `vertex`.

    Raises:
        ValueError: If `vertex` is neither endpoint.
    """
    # Direction at an unrelated vertex is undefined.
    if vertex not in edge.vertices:
        # Reject invalid incidence rather than silently treating it as incoming.
        raise ValueError(f"Edge {edge!r} is not incident to vertex {vertex!r}.")
    # The stored start endpoint defines the outgoing direction.
    return edge.start == vertex


def reduce_first_pair_permutations(
    triples: tuple[tuple[Edge, Edge, Edge], ...],
) -> tuple[tuple[Edge, Edge, Edge], ...]:
    r"""Identify triples differing only by exchange of their first two edges.

    In a 3+1 Thiemann term, the first two entries select a loop and the third
    selects a commutator segment. Exchanging the first pair is redundant because
    the kernel includes the loop and its inverse. Exchanging the third entry is
    not redundant. For permutations of $(a,b,c)$ in that order, this helper
    retains $(a,b,c)$, $(a,c,b)$, and $(b,c,a)$.

    Args:
        triples: Ordered triples, usually permutations of distinct incident edges.

    Returns:
        result (tuple[tuple[Edge, Edge, Edge], ...]): First representative of each
            unordered-first-pair and fixed-third-edge class, in encounter order.
    """
    # The key forgets only the order of the first two entries.
    seen: set[tuple[frozenset[Edge], Edge]] = set()
    # Keep the first representative so its parity remains deterministic.
    out: list[tuple[Edge, Edge, Edge]] = []
    # Read the segment separately from the interchangeable loop pair.
    for first, second, third in triples:
        # A different third edge always defines a different equivalence class.
        key = (frozenset((first, second)), third)
        # The reverse first pair has already supplied this loop-and-segment choice.
        if key in seen:
            # Discard that duplicate without changing the retained orientation.
            continue
        # Mark this class before considering subsequent permutations.
        seen.add(key)
        # Preserve the actual ordering needed by the parity calculation.
        out.append((first, second, third))
    # Return immutable construction metadata.
    return tuple(out)


def unordered_edge_triples(
    edges: tuple[Edge, ...],
) -> tuple[tuple[Edge, Edge, Edge], ...]:
    r"""Choose every three-edge subset once in the supplied edge order.

    Unlike the ordered tuples used in the volume, this enumeration has
    $\binom{n}{3}$ entries. It is useful when a caller handles permutations or
    symmetry factors explicitly. It does not assign orientation signs.

    Args:
        edges: Edge sequence whose order defines each subset's representative.

    Returns:
        result (tuple[tuple[Edge, Edge, Edge], ...]): Three-entry combinations.
    """
    # Combinations omit permutations while preserving the reference edge order.
    return tuple(combinations(edges, 3))


def k_vertex(graph: HalfLadderGraph, vertex: Any) -> tuple[Vertex, int]:
    r"""Resolve a half-ladder vertex in its radial sequence.

    Spherical stencils are indexed along `vertices_k`, which excludes the other
    endpoints of transverse edges. Integer arguments always refer to this radial
    sequence. Coordinate-like inputs must resolve to one of its members.

    Args:
        graph: Half-ladder graph with an ordered radial chain.
        vertex: Radial Vertex, coordinate-like value, or radial positional index.

    Returns:
        result (tuple[Vertex, int]): The radial vertex and its chain position.

    Raises:
        TypeError: If a boolean is supplied as a radial index.
        ValueError: If the index is out of range or the vertex is not radial.
    """
    # A boolean must not select the first or second radial vertex accidentally.
    if isinstance(vertex, bool):
        # Keep the invalid index diagnosis separate from graph membership errors.
        raise TypeError("k-vertex index must not be a bool.")
    # Interpret integers only as positions in the radial chain.
    if isinstance(vertex, int):
        # Reject both negative indexing and positions beyond the chain.
        if not 0 <= vertex < len(graph.vertices_k):
            # Report the supplied index before any stencil access.
            raise ValueError(f"k-vertex index {vertex} is out of range.")
        # Return the graph object and the already validated chain position.
        return graph.vertices_k[vertex], int(vertex)
    # Normalize coordinate input to the vertex type used by the graph.
    candidate = Vertex.from_like(vertex)
    # Transverse endpoints are not valid centers of a radial stencil.
    if candidate not in graph.vertices_k:
        # Explain which resolved vertex failed radial membership.
        raise ValueError(f"{candidate!r} is not a k-vertex of this HalfLadderGraph.")
    # Locate the radial position needed for neighboring stencil sites.
    return candidate, graph.vertices_k.index(candidate)


def other_endpoint(edge: Edge, vertex: Vertex) -> Vertex:
    r"""Follow an incident edge to its opposite endpoint.

    The operation ignores the stored edge orientation. It is used to find the
    left and right radial neighbors when constructing a spherical stencil.
    A self-loop returns the same vertex.

    Args:
        edge: Edge to follow.
        vertex: Starting endpoint.

    Returns:
        result (Vertex): Endpoint opposite the supplied vertex.

    Raises:
        ValueError: If the supplied vertex is not incident to the edge.
    """
    # Following the stored orientation reaches the end endpoint.
    if edge.start == vertex:
        # Return that endpoint without converting its representation.
        return edge.end
    # Following against the stored orientation reaches the start.
    if edge.end == vertex:
        # Traversal direction does not alter the neighboring Vertex object.
        return edge.start
    # No opposite endpoint exists when the starting vertex is unrelated.
    raise ValueError(f"Edge {edge!r} is not incident to vertex {vertex!r}.")


__all__ = [
    "edge_emanates_from",
    "graph_vertex",
    "incident_edge_tuples",
    "k_vertex",
    "orientation_sign",
    "other_endpoint",
    "reduce_first_pair_permutations",
    "unordered_edge_triples",
]
