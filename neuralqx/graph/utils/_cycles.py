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


"""Simple-cycle and shortest-cycle utilities for keyed graph edges.

The algorithms in this module operate on canonical graph edges and lightweight
raw edge triples. They preserve deterministic ordering while handling
parallel-edge two-cycles and ordinary girth cycles in the underlying simple
graph. Exhaustive enumeration also includes self-loops and every keyed-edge
realization of a simple vertex cycle.
"""

from collections import defaultdict
from collections import deque
from collections.abc import Hashable
from collections.abc import Iterator
from itertools import combinations
from itertools import pairwise
from itertools import product
from numbers import Integral
from typing import Any

import networkx as nx

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Loop
from neuralqx.utils.typing import RawEdge

from ._ordering import object_key
from ._ordering import unordered_pair_key


def all_cycles(edges: GraphEdges, length: int | None = None) -> list[Loop]:
    """Enumerate all simple undirected cycles, retaining parallel-edge choices.

    A simple cycle visits each vertex once before closing. Self-loops count
    as length one, and two distinct parallel edges form a length-two cycle.
    Longer cycles include all combinations of the available edge keys.
    Cyclic rotations and reversed traversals identify the same loop.

    NetworkX enumerates vertex cycles with the length bound applied during
    its search. Each vertex cycle is then expanded to keyed edge loops.
    Stored edge directions do not restrict traversal, matching the geometry
    used by the minimal-loop API. Dressing is applied separately.

    Args:
        edges: Canonical keyed primal graph edges.
        length: Inclusive maximum number of edges per loop. None includes
            every simple cycle. Zero returns an empty list.

    Returns:
        result (list[Loop]): New undressed loops, sorted by length and then
            by their canonical edge sequence. Each representative has a
            deterministic starting vertex and traversal direction.

    Raises:
        TypeError: If length is neither None nor an integer, or is a boolean.
        ValueError: If length is negative.
    """
    # Validate before building a graph or starting an unbounded enumeration.
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, Integral):
            raise TypeError("length must be a non-negative integer or None.")
        if length < 0:
            raise ValueError("length must be non-negative.")
        length = int(length)
        if length == 0:
            return []

    # An undirected multigraph retains edge keys while permitting reverse traversal.
    graph = nx.MultiGraph()
    graph.add_edges_from(_edges_to_raw(edges))
    loops: dict[tuple[Any, ...], list[RawEdge]] = {}

    # Passing the bound into NetworkX avoids enumerating longer cycles first.
    for vertices in nx.simple_cycles(graph, length_bound=length):
        # NetworkX returns node sequences, so restore every allowed edge-key choice.
        for raw_loop in _keyed_cycle_realizations(graph, vertices):
            # Normalize both traversal directions to identify an undirected loop.
            forward = canonical_rotate(raw_loop)
            backward = canonical_rotate(
                [(end, start, key) for start, end, key in reversed(raw_loop)]
            )
            canonical = min((forward, backward), key=_cycle_identity)
            loops[_cycle_identity(canonical)] = canonical

    # Stable ordering makes bounded results agree with filtering an unbounded call.
    ordered = sorted(
        loops.values(), key=lambda loop: (len(loop), _cycle_identity(loop))
    )
    return [Loop(Edge(start, end, key) for start, end, key in loop) for loop in ordered]


def _keyed_cycle_realizations(
    graph: nx.MultiGraph, vertices: list[Hashable]
) -> Iterator[list[RawEdge]]:
    """Expand a simple vertex cycle into its distinct keyed edge traversals.

    A node cycle does not distinguish parallel edges. A self-loop therefore
    emits one realization per key, and a two-vertex cycle emits one per pair
    of distinct keys. For three or more vertices, each segment can choose
    its key independently because the segments are distinct.

    Args:
        graph: Undirected multigraph containing the cycle's keyed edges.
        vertices: Nonempty node sequence produced by NetworkX simple_cycles,
            without a repeated closing vertex.

    Yields:
        result (list[RawEdge]): Closed edge sequence in the supplied vertex
            direction. Its reverse is identified by the caller.
    """
    if len(vertices) == 1:
        # Different self-loop keys describe different length-one cycles.
        vertex = vertices[0]
        for key in sorted(graph[vertex][vertex]):
            yield [(vertex, vertex, key)]
    elif len(vertices) == 2:
        # A return along the same edge is a backtrack, not a simple cycle.
        first, second = vertices
        for outward, inward in combinations(sorted(graph[first][second]), 2):
            yield [(first, second, outward), (second, first, inward)]
    else:
        # Include the closing segment when expanding the independent key choices.
        segments = list(zip(vertices, vertices[1:] + vertices[:1], strict=True))
        choices = (sorted(graph[start][end]) for start, end in segments)
        for keys in product(*choices):
            yield [
                (start, end, key)
                for (start, end), key in zip(segments, keys, strict=True)
            ]


def minimal_cycles(edges: GraphEdges) -> list[Loop]:
    """Returns all shortest primal cycles as canonical ``Loop`` objects.

    Args:
        edges: Canonical keyed graph edges to inspect.

    Returns:
        Shortest loops represented as ``Loop`` instances with canonical edge
        objects.
    """
    raw_loops = minimal_raw_cycles(_edges_to_raw(edges))
    return [
        Loop(Edge(start, end, key) for start, end, key in loop) for loop in raw_loops
    ]


def minimal_raw_cycles(edges: list[RawEdge]) -> list[list[RawEdge]]:
    """Returns all shortest cycles in a keyed multigraph.

    Parallel-edge two-cycles are returned first because they are the shortest
    possible loops. If no such cycles exist, all girth cycles in the underlying
    simple graph are enumerated.

    Args:
        edges: Raw keyed edge triples ``(start, end, key)``.

    Returns:
        Shortest closed raw-edge walks in deterministic canonical orientation.
    """
    raw_edges = list(edges)
    two_cycles = _parallel_two_cycles(raw_edges)
    if two_cycles:
        return _deduplicate_cycles(two_cycles)

    adjacency = _simple_adjacency(raw_edges)
    girth, witness_edges = _girth_and_witness_edges(adjacency)
    if girth == 0:
        return []

    segment_keys = _segment_keys(raw_edges)
    seen: set[tuple[Any, ...]] = set()
    loops: list[list[RawEdge]] = []

    for start, end in _simple_edges(adjacency):
        segment = unordered_pair_key(start, end)
        if segment not in witness_edges:
            continue
        paths = _all_shortest_paths_avoid_edge(
            adjacency,
            start,
            end,
            segment,
            target_len=girth - 1,
        )
        for path in paths:
            raw_loop: list[RawEdge] = []
            for left, right in pairwise(path):
                raw_loop.append((left, right, _pick_key(segment_keys, left, right)))
            raw_loop.append((end, start, _pick_key(segment_keys, start, end)))

            loop = canonical_rotate(raw_loop)
            reverse = canonical_rotate(
                [(right, left, key) for left, right, key in reversed(loop)]
            )
            identity = min(_cycle_identity(loop), _cycle_identity(reverse))
            if identity not in seen:
                seen.add(identity)
                loops.append(loop)

    return [loop for loop in loops if len(loop) == girth]


def canonical_rotate(loop: list[RawEdge]) -> list[RawEdge]:
    """Rotates a closed edge loop to a deterministic representative.

    Args:
        loop: Raw closed edge walk to normalize.

    Returns:
        A rotation of ``loop`` with the smallest structural identity. If the
        edges do not form a closed walk, the original list is returned.
    """
    if not loop:
        return loop
    candidates: list[list[RawEdge]] = []
    n_edges = len(loop)
    for index in range(n_edges):
        candidate = loop[index:] + loop[:index]
        if all(
            candidate[j][1] == candidate[(j + 1) % n_edges][0] for j in range(n_edges)
        ):
            candidates.append(candidate)
    if not candidates:
        return loop
    return min(candidates, key=_cycle_identity)


def _edges_to_raw(edges: GraphEdges) -> list[RawEdge]:
    """Converts canonical ``Edge`` objects to raw edge triples.

    Args:
        edges: Canonical keyed graph edges.

    Returns:
        Raw triples preserving start vertex, end vertex, and parallel-edge key.
    """
    return [(edge.start, edge.end, edge.key) for edge in edges]


def _parallel_two_cycles(edges: list[RawEdge]) -> list[list[RawEdge]]:
    """Finds shortest loops formed by parallel edges between two vertices.

    Args:
        edges: Raw keyed edge triples to group by unordered endpoint pair.

    Returns:
        Two-edge closed loops produced by parallel edges, normalized by
        :func:`canonical_rotate`.
    """
    by_pair: dict[tuple[Any, Any], list[RawEdge]] = defaultdict(list)
    for start, end, key in edges:
        by_pair[unordered_pair_key(start, end)].append((start, end, key))

    loops: list[list[RawEdge]] = []
    for siblings in by_pair.values():
        if len(siblings) < 2:
            continue
        for first, second in combinations(siblings, 2):
            first_start, first_end, _ = first
            second_start, second_end, second_key = second
            if second_start == first_end and second_end == first_start:
                closing = second
            else:
                closing = (second_end, second_start, second_key)
            loops.append(canonical_rotate([first, closing]))
    return loops


def _deduplicate_cycles(cycles: list[list[RawEdge]]) -> list[list[RawEdge]]:
    """Removes duplicate cycles while preserving first-seen order.

    Args:
        cycles: Candidate raw cycles that may contain repeated identities.

    Returns:
        Cycles whose structural identities are unique.
    """
    seen: set[tuple[Any, ...]] = set()
    out: list[list[RawEdge]] = []
    for cycle in cycles:
        identity = _cycle_identity(cycle)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(cycle)
    return out


def _simple_adjacency(edges: list[RawEdge]) -> dict[Hashable, set[Hashable]]:
    """Builds an undirected simple adjacency graph from raw edges.

    Args:
        edges: Raw keyed edge triples. Self-loops are ignored for the simple
            adjacency structure.

    Returns:
        Mapping from each vertex to its neighboring vertices.
    """
    adjacency: dict[Hashable, set[Hashable]] = defaultdict(set)
    for start, end, _ in edges:
        if start == end:
            continue
        adjacency[start].add(end)
        adjacency[end].add(start)
    return dict(adjacency)


def _simple_edges(
    adjacency: dict[Hashable, set[Hashable]],
) -> list[tuple[Hashable, Hashable]]:
    """Returns deterministic undirected edges from a simple adjacency graph.

    Args:
        adjacency: Undirected adjacency mapping.

    Returns:
        Sorted endpoint pairs with each unordered segment appearing once.
    """
    seen: set[tuple[Any, Any]] = set()
    edges: list[tuple[Hashable, Hashable]] = []
    for start, neighbours in adjacency.items():
        for end in neighbours:
            key = unordered_pair_key(start, end)
            if key in seen:
                continue
            seen.add(key)
            if object_key(end) < object_key(start):
                first, second = end, start
            else:
                first, second = start, end
            edges.append((first, second))
    return sorted(edges, key=lambda edge: (object_key(edge[0]), object_key(edge[1])))


def _segment_keys(edges: list[RawEdge]) -> dict[tuple[Any, Any], list[int]]:
    """Collects sorted parallel-edge keys for each simple segment.

    Args:
        edges: Raw keyed edge triples.

    Returns:
        Mapping from unordered endpoint-pair identity to sorted edge keys.
    """
    keys: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for start, end, key in edges:
        if start == end:
            continue
        keys[unordered_pair_key(start, end)].append(key)
    return {segment: sorted(values) for segment, values in keys.items()}


def _pick_key(
    segment_keys: dict[tuple[Any, Any], list[int]],
    start: Hashable,
    end: Hashable,
) -> int:
    """Selects the deterministic representative key for a simple segment.

    Args:
        segment_keys: Mapping produced by :func:`_segment_keys`.
        start: First endpoint of the segment.
        end: Second endpoint of the segment.

    Returns:
        Smallest available key for the unordered endpoint pair.

    Raises:
        ValueError: If the segment is absent from ``segment_keys``.
    """
    try:
        return segment_keys[unordered_pair_key(start, end)][0]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Undirected segment {{{start!r}, {end!r}}} not found."
        ) from exc


def _girth_and_witness_edges(
    adjacency: dict[Hashable, set[Hashable]],
) -> tuple[int, set[tuple[Any, Any]]]:
    """Computes graph girth and the segments that witness it.

    Args:
        adjacency: Undirected simple adjacency mapping.

    Returns:
        Pair containing the girth length and the unordered segments that close
        shortest cycles. A girth of ``0`` means no cycle was found.
    """
    best = 10**12
    witnesses: set[tuple[Any, Any]] = set()

    for source in sorted(adjacency, key=object_key):
        distance: dict[Hashable, int] = {source: 0}
        parent: dict[Hashable, Hashable | None] = {source: None}
        queue: deque[Hashable] = deque([source])

        while queue:
            vertex = queue.popleft()
            if distance[vertex] + 1 >= best:
                continue
            for neighbour in sorted(adjacency[vertex], key=object_key):
                if neighbour not in distance:
                    distance[neighbour] = distance[vertex] + 1
                    parent[neighbour] = vertex
                    queue.append(neighbour)
                    continue
                if parent[vertex] == neighbour or parent[neighbour] == vertex:
                    continue

                cycle_len = distance[vertex] + distance[neighbour] + 1
                witness = unordered_pair_key(vertex, neighbour)
                if cycle_len < best:
                    best = cycle_len
                    witnesses = {witness}
                elif cycle_len == best:
                    witnesses.add(witness)

    if best == 10**12:
        return 0, set()
    return int(best), witnesses


def _all_shortest_paths_avoid_edge(
    adjacency: dict[Hashable, set[Hashable]],
    start: Hashable,
    end: Hashable,
    avoid: tuple[Any, Any],
    *,
    target_len: int,
    max_paths: int = 100000,
) -> list[list[Hashable]]:
    """Enumerates shortest simple paths while excluding one segment.

    Args:
        adjacency: Undirected simple adjacency mapping.
        start: Path start vertex.
        end: Path end vertex.
        avoid: Unordered segment identity that must not be traversed.
        target_len: Required path length in edges.
        max_paths: Upper bound on the number of returned paths.

    Returns:
        Simple paths from ``start`` to ``end`` with exactly ``target_len``
        edges, up to ``max_paths`` entries.
    """
    distance: dict[Hashable, int] = {start: 0}
    predecessors: dict[Hashable, list[Hashable]] = defaultdict(list)
    queue: deque[Hashable] = deque([start])

    while queue:
        vertex = queue.popleft()
        if distance[vertex] >= target_len:
            continue

        for neighbour in sorted(adjacency[vertex], key=object_key):
            if unordered_pair_key(vertex, neighbour) == avoid:
                continue
            new_distance = distance[vertex] + 1
            if neighbour not in distance:
                distance[neighbour] = new_distance
                predecessors[neighbour].append(vertex)
                queue.append(neighbour)
            elif new_distance == distance[neighbour]:
                predecessors[neighbour].append(vertex)

    if distance.get(end) != target_len:
        return []

    paths: list[list[Hashable]] = []
    stack: list[tuple[Hashable, list[Hashable]]] = [(end, [end])]
    while stack and len(paths) < max_paths:
        vertex, reverse_path = stack.pop()
        if vertex == start:
            path = list(reversed(reverse_path))
            if len(path) == len(set(path)):
                paths.append(path)
            continue
        for predecessor in predecessors.get(vertex, []):
            stack.append((predecessor, [*reverse_path, predecessor]))

    return paths


def _cycle_identity(loop: list[RawEdge]) -> tuple[Any, ...]:
    """Builds a deterministic structural identity for a raw cycle.

    Args:
        loop: Raw edge cycle to identify.

    Returns:
        Tuple key suitable for sorting and duplicate detection.
    """
    return tuple((object_key(start), object_key(end), key) for start, end, key in loop)


__all__ = [
    "RawEdge",
    "all_cycles",
    "canonical_rotate",
    "minimal_cycles",
    "minimal_raw_cycles",
]
