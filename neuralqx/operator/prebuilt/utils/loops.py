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


"""Loop parsing and WCL triangulation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

from neuralqx.hilbert.u1.spaces.abstract import AbstractU1Hilbert
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import Loop
from neuralqx.utils.typing import Vertex

from .u1 import canonical_edge
from .u1 import edge_index


def type_to_step(value: Any) -> int:
    r"""Interpret a dressing label as a signed lattice step.

    Creation labels denote +1 and annihilation labels denote -1. Integer-like
    values are classified by sign, with zero treated as -1. For compatibility
    with existing loop records, an unrecognized value defaults to +1. This is a
    permissive parser, not validation of a physical loop or its orientation.

    Args:
        value: Dressing string or value convertible to an integer.

    Returns:
        result (int): Either +1 or -1.
    """
    # Named dressings are recognized before attempting numeric conversion.
    if isinstance(value, str):
        # Ignore surrounding whitespace and capitalization in named labels.
        token = value.strip().lower()
        # These names all describe the positive ket charge shift.
        if token in {"creation", "+", "up", "plus", "raise", "raising"}:
            # Use a positive step for creation labels and the compatibility fallback.
            return +1
        # These names all describe the inverse charge shift.
        if token in {"annihilation", "-", "down", "minus", "lower", "lowering"}:
            # An annihilation dressing decreases the charge by one lattice step.
            return -1
    # Numeric records may encode the direction instead of naming an operation.
    try:
        # Only the sign matters, with zero following the legacy negative convention.
        return +1 if int(value) > 0 else -1
    # Preserve the legacy positive default when a value cannot be interpreted.
    except Exception:
        # Use a positive step for creation labels and the compatibility fallback.
        return +1


def parse_loop(
    hilbert: AbstractU1Hilbert, loop: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r"""Resolve a loop representation to parallel edge-index and direction tuples.

    Dressed loops carry the traversal signs needed for a holonomy. Their labels
    are interpreted relative to the stored graph edges, independently of the
    endpoint notation passed to edge lookup. A plain edge sequence has no such
    dressing, so every entry receives a positive shift for compatibility with
    the existing API. In particular, reversing the endpoints of an undressed
    edge does not turn that shift negative. Use a dressed loop when orientation
    is part of the operator definition.

    Entries remain in traversal order and repeated edges are retained. The
    result describes charge shifts, not matrix elements. The consuming operator
    must decide whether it is constructing ket action or matrix-row connectivity.

    Args:
        hilbert: U(1) space whose graph maps edges to base indices.
        loop: A Loop, an object exposing `dressed_edges`, a sequence of dressed
            `(edge, mapping)` entries, or a plain edge sequence.

    Returns:
        result (tuple[tuple[int, ...], tuple[int, ...]]): Base-edge indices and
            equally sized signed-step tuples. Empty input gives two empty tuples.
    """
    # The typed Loop representation exposes both geometry and optional dressing.
    if isinstance(loop, Loop):
        # Read dressing before canonical edge lookup can discard orientation.
        dressed = loop.dressed_edges
        # An undressed Loop uses the same positive-step convention as a plain sequence.
        if dressed is None:
            # Resolve every edge and retain one positive step for each occurrence.
            return tuple(edge_index(hilbert, edge) for edge in loop.edges), tuple(
                +1 for _ in loop.edges
            )
        # Dressed input supplies its own signed shift for every edge.
        return parse_dressed_edges(hilbert, dressed)

    # Accept compatible loop objects without requiring the concrete Loop class.
    if hasattr(loop, "dressed_edges"):
        # Read dressing before canonical edge lookup can discard orientation.
        dressed = loop.dressed_edges
        # Use their dressing when available, otherwise continue to sequence parsing.
        if dressed is not None:
            # Dressed input supplies its own signed shift for every edge.
            return parse_dressed_edges(hilbert, dressed)

    # Freeze iterable input so detection and parsing see the same entries.
    items = tuple(loop) if isinstance(loop, Iterable) else ()
    # An empty or non-iterable input represents an empty shift sequence.
    if not items:
        # Return matching empty tuples, which act as an identity shift.
        return (), ()
    # Require every entry to carry a mapping before selecting dressed parsing.
    if all(_looks_dressed(item) for item in items):
        # Interpret mapping entries as directions rather than graph endpoints.
        return parse_dressed_edges(hilbert, items)
    # Plain entries retain the historical all-positive shift convention.
    return tuple(edge_index(hilbert, item) for item in items), tuple(+1 for _ in items)


def parse_dressed_edges(
    hilbert: AbstractU1Hilbert,
    dressed_edges: Iterable[Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r"""Separate dressed traversals into base-edge indices and signed steps.

    Each input entry pairs an edge with either a dressing mapping or a direction
    label. A mapping uses its `type` entry and defaults to `creation` when that
    entry is absent. The order and any repeated edge visits are preserved so the
    shift kernel can accumulate the complete traversal.

    Args:
        hilbert: U(1) space supplying base-edge lookup.
        dressed_edges: Iterable of `(edge, dressing)` pairs.

    Returns:
        result (tuple[tuple[int, ...], tuple[int, ...]]): Matching tuples of base
            indices and directions, with directions interpreted by `type_to_step`.
    """
    # Store geometric edge positions independently of their traversal signs.
    edges: list[int] = []
    # The parallel list preserves one direction for every edge occurrence.
    steps: list[int] = []
    # Read entries in traversal order, including repeated visits.
    for edge, dressing in dressed_edges:
        # Canonical indexing selects the state component without choosing its shift.
        edges.append(edge_index(hilbert, edge))
        # Graph-generated dressed loops store operation labels in a mapping.
        if isinstance(dressing, Mapping):
            # A missing type follows the established creation default.
            steps.append(type_to_step(dressing.get("type", "creation")))
        # Direct labels are also accepted by this explicit dressed parser.
        else:
            # Normalize the direct direction label to one signed lattice step.
            steps.append(type_to_step(dressing))
    # Freeze matching metadata sequences for use inside operator kernels.
    return tuple(edges), tuple(steps)


def canonical_loop_edges(hilbert: AbstractU1Hilbert, loop: Loop) -> tuple[Edge, ...]:
    r"""Resolve traversed loop edges to their stored graph representatives.

    This is an incidence helper. It preserves traversal order and edge keys but
    replaces endpoint orientation with the graph orientation. Holonomy directions
    must be read separately from the loop dressing.

    Args:
        hilbert: Space whose graph owns the loop edges.
        loop: Loop exposing an ordered `edges` sequence.

    Returns:
        result (tuple[Edge, ...]): Stored graph edges in loop traversal order.
    """
    # Canonicalize each visit without sorting or removing repeated edge entries.
    return tuple(canonical_edge(hilbert, edge) for edge in loop.edges)


def loop_edges_at_vertex(
    hilbert: AbstractU1Hilbert, loop: Loop, vertex: Vertex
) -> tuple[Edge, Edge] | None:
    r"""Extract the two incident edges needed for a planar loop contribution.

    The ordering is the loop edge sequence ordering after canonicalization. It
    is not automatically outgoing-then-incoming. The planar triangulation uses
    this ordering to assign permutation signs. A loop must have exactly two
    incident entries to supply an unambiguous segment pair.

    Args:
        hilbert: Space providing canonical graph edge lookup.
        loop: Loop whose local incidence is being inspected.
        vertex: Resolved graph vertex.

    Returns:
        result (tuple[Edge, Edge] | None): The ordered incident pair, or None when
            the loop has fewer or more than two incident entries at the vertex.
    """
    # Filter in loop order while retaining distinct visits to the same base edge.
    incident = [
        edge for edge in canonical_loop_edges(hilbert, loop) if vertex in edge.vertices
    ]
    # The planar construction requires exactly one two-edge passage through the vertex.
    if len(incident) != 2:
        # Signal that this loop cannot supply the required local pair.
        return None
    # Preserve the order used to define the combinatorial pair sign.
    return incident[0], incident[1]


def loops_containing_edges(
    hilbert: AbstractU1Hilbert,
    loops: tuple[Loop, ...],
    first: Edge,
    second: Edge,
) -> tuple[Loop, ...]:
    r"""Find minimal-loop candidates containing both selected base edges.

    Membership ignores traversal direction but retains graph edge keys, so
    parallel edges are not conflated. The returned sequence preserves the input
    loop order. Its length is the loop-average denominator in a 3+1 triangulation
    term. This helper does not itself check minimality or orient the selected
    loops. Its caller supplies minimal loops and fixes their direction at the
    vertex afterwards.

    Args:
        hilbert: Space defining base-edge identity.
        loops: Candidate loops, normally `graph.dressed_minimal_loops()`.
        first: First edge of the incident pair.
        second: Second edge of the incident pair.

    Returns:
        result (tuple[Loop, ...]): All candidates containing both edges, or an
            empty tuple when the pair cannot be completed by a candidate loop.
    """
    # Compare base-edge identity so a reversed traversal still matches its edge.
    wanted = {edge_index(hilbert, first), edge_index(hilbert, second)}
    # Preserve each matching candidate for the later loop average.
    out: list[Loop] = []
    # Check candidates in the graph-provided order.
    for loop in loops:
        # Membership depends on which edges occur, not their direction or multiplicity.
        present = {edge_index(hilbert, edge) for edge in loop.edges}
        # Both selected incident edges must belong to the same candidate loop.
        if wanted.issubset(present):
            # Retain the original Loop so its dressing remains available.
            out.append(loop)
    # An empty result is handled by skipping the corresponding triangulation term.
    return tuple(out)


def _looks_dressed(item: Any) -> bool:
    r"""Recognize sequence entries with an explicit dressing mapping.

    This conservative shape check distinguishes `(edge, mapping)` records from
    ordinary endpoint pairs. It does not validate the edge or mapping contents.
    Direct non-mapping labels are supported by `parse_dressed_edges` when that
    parser is called explicitly, rather than inferred from a generic sequence.

    Args:
        item: One entry from a possible loop sequence.

    Returns:
        result (bool): Whether the entry is a two-item tuple or list ending in a mapping.
    """
    # Only a trailing mapping unambiguously identifies a dressed sequence entry.
    return (
        isinstance(item, tuple | list)
        and len(item) == 2
        and isinstance(item[1], Mapping)
    )


__all__ = [
    "canonical_loop_edges",
    "loop_edges_at_vertex",
    "loops_containing_edges",
    "parse_dressed_edges",
    "parse_loop",
    "type_to_step",
]
