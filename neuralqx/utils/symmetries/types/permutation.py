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
class Permutation(Symmetry):
    """
    Edge permutation symmetry defined on canonical edges.

    The permutation is stored as a mapping on canonical (orientation-agnostic) edges. When queried
    with an oriented edge, the mapping is applied to the canonical form and the result is then
    oriented to match the query edge. Iteration yields pairs in the order and orientation of the
    user-provided ``domain``.

    :param mapping_canon: Mapping from canonical edges to canonical edges implementing the
                          permutation.
    :param domain: Oriented edges supplied by the user; used for iteration/printing.
    :param name: Optional name for the permutation. Defaults to ``"perm"``.
    """

    mapping_canon: Dict[Edge, Edge]
    """A mapping from canonical edges to canonical edges."""

    domain: List[Edge]
    """Oriented edges supplied from the user."""

    name: str = "perm"
    """Default name for a permutation."""

    def __getitem__(self, e: Edge) -> Edge:
        """
        Map an edge under the permutation.

        The query edge is canonicalized for lookup in ``mapping_canon``, and the mapped canonical edge
        is then oriented to match the query edge.

        :param e: Edge to map (may be oriented).
        :returns: Mapped edge oriented like ``e``.
        :raises KeyError: If the canonicalized edge is not in the mapping.
        """

        return _orient_like(self.mapping_canon[_canon_edge(e)], e)

    def __iter__(self):
        """
        Iterate over oriented (edge, image) pairs in the user-provided domain order.

        :returns: Iterator over ``(e, self[e])`` for each oriented edge ``e`` in ``domain``.
        """

        # yield oriented pairs in the same orientation as provided by user
        for e in self.domain:
            yield (e, self[e])

    def __len__(self):
        """
        Return the number of canonical edges in the permutation mapping.

        :returns: Size of ``mapping_canon``.
        """

        return len(self.mapping_canon)
