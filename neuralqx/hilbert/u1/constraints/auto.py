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


"""Automatic spanning-tree U(1) gauge fixing.

Automatic gauge fixing chooses a spanning tree of the graph and expresses tree
edges through chord edges by solving the reduced incidence system. The result is
a constructive relation set compatible with the U(1) gauge-fixing topology.
"""

from __future__ import annotations

from collections import defaultdict
from collections import deque
from typing import Any

import numpy as np

from neuralqx.graph import AbstractGraph

from .errors import U1GaugeFixingError
from .relation import GaugeRelation


def auto_gauge_fixing(graph: AbstractGraph) -> tuple[GaugeRelation, ...]:
    """Generates spanning-tree gauge fixing from graph incidence data.

    Args:
        graph: Graph whose edge incidence determines gauge relations.

    Returns:
        Constructive gauge relations for automatically selected tree edges.

    Raises:
        U1GaugeFixingError: If the graph is disconnected or generated
            coefficients are unsupported.
    """
    n_vertices = graph.n_vertices
    n_edges = graph.n_edges
    if n_edges == 0:
        return ()

    incidence = _incidence_matrix(graph)
    tree_edges = _spanning_tree_edge_indices(graph)
    if len(tree_edges) < max(0, n_vertices - 1):
        raise U1GaugeFixingError(
            "Cannot auto-generate gauge fixing for a disconnected graph."
        )
    tree_edges = tree_edges[: max(0, n_vertices - 1)]
    chord_edges = [edge for edge in range(n_edges) if edge not in set(tree_edges)]
    if not tree_edges:
        return ()

    reduced = np.delete(incidence, n_vertices - 1, axis=0)
    tree_matrix = reduced[:, tree_edges]
    chord_matrix = reduced[:, chord_edges]
    if chord_matrix.shape[1] == 0:
        return tuple(GaugeRelation(lhs=edge, rhs=()) for edge in tree_edges)

    coeffs = -np.linalg.solve(tree_matrix.astype(float), chord_matrix.astype(float))
    coeffs = np.rint(coeffs).astype(int)
    return tuple(_relations_from_coefficients(tree_edges, chord_edges, coeffs))


def _incidence_matrix(graph: AbstractGraph) -> np.ndarray:
    """Builds the oriented vertex-edge incidence matrix for a graph."""
    vertex_to_index = {vertex: idx for idx, vertex in enumerate(graph.vertices)}
    incidence = np.zeros((graph.n_vertices, graph.n_edges), dtype=int)
    for edge_index, edge in enumerate(graph.edges):
        incidence[vertex_to_index[edge.start], edge_index] = 1
        incidence[vertex_to_index[edge.end], edge_index] = -1
    return incidence


def _relations_from_coefficients(
    tree_edges: list[int],
    chord_edges: list[int],
    coeffs: np.ndarray,
) -> list[GaugeRelation]:
    """Converts solved incidence coefficients into gauge relations."""
    relations = []
    for row, tree_edge in enumerate(tree_edges):
        rhs = []
        for col, coeff in enumerate(coeffs[row]):
            if coeff == 0:
                continue
            if abs(int(coeff)) != 1:
                raise U1GaugeFixingError(
                    f"Auto gauge fixing generated unsupported coefficient {coeff}."
                )
            rhs.append((chord_edges[col], int(coeff)))
        relations.append(GaugeRelation(lhs=tree_edge, rhs=tuple(rhs)))
    return relations


def _spanning_tree_edge_indices(graph: AbstractGraph) -> list[int]:
    """Returns edge indices selected by breadth-first spanning tree search."""
    adjacency: dict[Any, list[tuple[Any, int]]] = defaultdict(list)
    for edge_index, edge in enumerate(graph.edges):
        adjacency[edge.start].append((edge.end, edge_index))
        adjacency[edge.end].append((edge.start, edge_index))

    if not graph.vertices:
        return []

    seen = {graph.vertices[0]}
    queue = deque([graph.vertices[0]])
    tree_edges: list[int] = []
    while queue and len(tree_edges) < graph.n_vertices - 1:
        vertex = queue.popleft()
        for neighbor, edge_index in adjacency[vertex]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
            tree_edges.append(edge_index)
            if len(tree_edges) == graph.n_vertices - 1:
                break
    return tree_edges
