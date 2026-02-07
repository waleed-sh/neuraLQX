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

from dataclasses import dataclass
from typing import ClassVar, Dict, List

from .symmetry import Symmetry
from .utils import Edge, _orient_like, _canon_edge


@dataclass(frozen=True)
class Automorphism(Symmetry):
    """
    Vertex automorphism with induced edge mapping on canonical edges.

    An automorphism consists of:

    - a vertex relabeling map ``vertex_map``, and
    - an induced edge map ``edge_map_canon`` on canonical edges (optionally including permutations
      of parallel-edge keys within unordered vertex pairs).

    When queried with an oriented edge, the edge is canonicalized, mapped via ``edge_map_canon``,
    and then oriented to match the query edge's orientation. Iteration yields pairs in the order
    and orientation of the user-provided ``domain``.

    :param vertex_map: Mapping from source vertices to target vertices.
    :param edge_map_canon: Mapping from canonical edges to canonical edges induced by ``vertex_map``
                           (and possible within-pair key permutations).
    :param domain: Oriented edges supplied by the user; used for iteration/printing.
    :param name: Optional name for the automorphism. Defaults to ``"auto"``.
    """

    vertex_map: Dict[int, int]
    """A map between vertices."""

    edge_map_canon: Dict[Edge, Edge]
    """A map between canonical edges."""

    domain: List[Edge]
    """Oriented edges supplied from the user."""

    name: str = "auto"
    """Default name for an automorphism."""

    def __getitem__(self, e: Edge) -> Edge:
        """
        Map an edge under the automorphism.

        The query edge is canonicalized for lookup in ``edge_map_canon``, and the mapped canonical edge
        is then oriented to match the query edge.

        :param e: Edge to map (may be oriented).
        :returns: Mapped edge oriented like ``e``.
        :raises KeyError: If the canonicalized edge is not in the mapping.
        """

        return _orient_like(self.edge_map_canon[_canon_edge(e)], e)

    def map_vertex(self, v: int) -> int:
        """
        Map a vertex under the automorphism.

        :param v: Source vertex id.
        :returns: Image vertex id ``vertex_map[v]``.
        :raises KeyError: If ``v`` is not present in ``vertex_map``.
        """

        return self.vertex_map[v]

    def __iter__(self):
        """
        Iterate over oriented (edge, image) pairs in the user-provided domain order.

        :returns: Iterator over ``(e, self[e])`` for each oriented edge ``e`` in ``domain``.
        """

        for e in self.domain:
            yield (e, self[e])

    def __len__(self):
        """
        Return the number of canonical edges in the automorphism mapping.

        :returns: Size of ``edge_map_canon``.
        """

        return len(self.edge_map_canon)
