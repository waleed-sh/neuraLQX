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


"""Parsing helpers for normalized graph inputs.

This module contains small utilities that derive canonical vertex collections
from validated edge sequences while preserving deterministic graph order.
"""

from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Vertex


def find_unique_vertices(edges: GraphEdges) -> list[Vertex]:
    """Returns unique vertices in first-seen edge order.

    Args:
        edges: Canonical keyed graph edges.

    Returns:
        Vertices discovered by scanning each edge start and end endpoint in
        graph order.
    """
    vertices: list[Vertex] = []
    seen: set[Vertex] = set()
    for edge in edges:
        for vertex in edge.vertices:
            if vertex not in seen:
                vertices.append(vertex)
                seen.add(vertex)
    return vertices


__all__ = [
    "find_unique_vertices",
]
