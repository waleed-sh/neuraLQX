#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from neuralqx.graph.core import AbstractGraph


def _stable_sort(values: set[Any]) -> tuple[Any, ...]:
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=lambda x: (type(x).__name__, repr(x))))


@dataclass(frozen=True)
class SU2GraphMetadata:
    n_edges: int
    n_vertices: int
    edge_tokens: tuple[tuple[Any, ...], ...]
    vertex_ids: tuple[Any, ...]
    vertex_index: dict[Any, int]
    incident_edges_by_vertex: tuple[tuple[int, ...], ...]
    incident_orient_by_vertex: tuple[tuple[int, ...], ...]
    valence_by_vertex: tuple[int, ...]
    incident_edges_padded: jnp.ndarray
    incident_orient_padded: jnp.ndarray
    incident_mask: jnp.ndarray
    valence: jnp.ndarray

    @property
    def max_valence(self) -> int:
        return max(self.valence_by_vertex, default=0)


# TODO: should this be moved to the graph API? some of these things are provided by the API itself
def compile_su2_graph_metadata(graph: AbstractGraph) -> SU2GraphMetadata:

    n_edges = int(graph.n_edges)
    edge_tokens = tuple(tuple(graph.index_to_edge(i)) for i in range(n_edges))

    vertices: set[Any] = set(getattr(graph.handler, "vertices", ()))
    incidents: dict[Any, list[tuple[int, int]]] = defaultdict(list)

    for edge_index, edge in enumerate(edge_tokens):
        if len(edge) < 2:
            raise ValueError(f"Graph edge {edge!r} has fewer than two endpoints.")
        tail, head = edge[0], edge[1]
        vertices.add(tail)
        vertices.add(head)
        incidents[tail].append((edge_index, +1))
        incidents[head].append((edge_index, -1))

    vertex_ids = _stable_sort(vertices)
    vertex_index = {v: i for i, v in enumerate(vertex_ids)}

    incident_edges: list[tuple[int, ...]] = []
    incident_orient: list[tuple[int, ...]] = []
    valences: list[int] = []

    for vertex in vertex_ids:
        local = sorted(incidents.get(vertex, ()), key=lambda item: (item[0], -item[1]))
        incident_edges.append(tuple(edge for edge, _orient in local))
        incident_orient.append(tuple(orient for _edge, orient in local))
        valences.append(len(local))

    n_vertices = len(vertex_ids)
    max_valence = max(valences, default=0)
    padded_width = max(max_valence, 1)

    edges_padded = -jnp.ones((n_vertices, padded_width), dtype=jnp.int32)
    orient_padded = jnp.zeros((n_vertices, padded_width), dtype=jnp.int8)
    mask = jnp.zeros((n_vertices, padded_width), dtype=bool)

    for v_pos, (edges, orientations) in enumerate(zip(incident_edges, incident_orient)):
        if not edges:
            continue
        width = len(edges)
        edges_padded = edges_padded.at[v_pos, :width].set(
            jnp.asarray(edges, dtype=jnp.int32)
        )
        orient_padded = orient_padded.at[v_pos, :width].set(
            jnp.asarray(orientations, dtype=jnp.int8)
        )
        mask = mask.at[v_pos, :width].set(True)

    return SU2GraphMetadata(
        n_edges=n_edges,
        n_vertices=n_vertices,
        edge_tokens=edge_tokens,
        vertex_ids=vertex_ids,
        vertex_index=vertex_index,
        incident_edges_by_vertex=tuple(incident_edges),
        incident_orient_by_vertex=tuple(incident_orient),
        valence_by_vertex=tuple(valences),
        incident_edges_padded=edges_padded,
        incident_orient_padded=orient_padded,
        incident_mask=mask,
        valence=jnp.asarray(valences, dtype=jnp.int32),
    )


__all__ = [
    "SU2GraphMetadata",
    "compile_su2_graph_metadata",
]
