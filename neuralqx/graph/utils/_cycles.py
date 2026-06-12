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


"""Shortest-cycle utilities for keyed graph edges.

The algorithms in this module operate on canonical graph edges and lightweight
raw edge triples. They preserve deterministic ordering while handling
parallel-edge two-cycles and ordinary girth cycles in the underlying simple
graph.
"""

from collections import defaultdict
from collections import deque
from collections.abc import Hashable

from itertools import combinations

from typing import Any

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Loop
from neuralqx.utils.typing import RawEdge

from ._ordering import object_key
from ._ordering import unordered_pair_key


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
            for left, right in zip(path[:-1], path[1:]):
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
            stack.append((predecessor, reverse_path + [predecessor]))

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
    "canonical_rotate",
    "minimal_cycles",
    "minimal_raw_cycles",
]
