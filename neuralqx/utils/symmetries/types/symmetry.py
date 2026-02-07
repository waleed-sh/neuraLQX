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
from typing import ClassVar

from .utils import Edge


@dataclass(frozen=True)
class Symmetry:
    """
    Minimal base class for symmetries that map edges to edges.

    Subclasses are expected to implement ``__getitem__(edge) -> edge`` (typically by looking up the
    canonical form of the queried edge and then re-orienting the result to match the query).
    This base class provides a uniform ``map_edge(edge) -> edge`` API for users that prefer an
    explicit method call, and also makes instances callable via ``__call__``.

    :returns: Instances of subclasses map an input edge to its image edge under the symmetry.
    """

    def map_edge(self, e: Edge) -> Edge:
        """
        Map an edge under this symmetry.

        By default, this delegates to ``self[e]`` and therefore requires subclasses to implement
        ``__getitem__``. Subclasses may override this method directly if they do not use
        ``__getitem__``-based access.

        :param e: Edge to map.
        :returns: The mapped edge.
        :raises TypeError: If the subclass does not implement ``__getitem__`` and does not override
                           ``map_edge``.
        """

        try:
            # delegate to __getitem__ implemented in subclasses
            return self[e]
        except Exception as ex:
            raise TypeError(
                f"{type(self).__name__} must implement __getitem__(edge) or override "
                f"map_edge(edge)."
            ) from ex

    def __call__(self, e: Edge) -> Edge:
        """
        Call the symmetry as a function on an edge.

        This is equivalent to calling :meth:`map_edge`.

        :param e: Edge to map.
        :returns: The mapped edge.
        """

        return self.map_edge(e)
