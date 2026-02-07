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

from typing import Tuple

Edge = Tuple[int, int, int]
"""An edge type, this defaults to the non-planar edge representation (v_out, v_in, key)"""


def _canon_edge(e: Edge) -> Edge:
    """
    A helper function which returns the canonical, orientation-agnostic form of a given edge.

    :param e: the edge to canonically order
    :return: a canonically ordered edge
    """
    u, v, k = e
    return (u, v, k) if u <= v else (v, u, k)


def _is_reversed_vs_canon(e: Edge) -> bool:
    """
    Returns True if the given edge ``e`` is reversed relative to its canonical orientation.
    """
    u, v, k = e
    cu, cv, ck = _canon_edge(e)
    return (u, v, k) != (cu, cv, ck)


def _orient_like(edge_canon: Edge, like: Edge) -> Edge:
    """
    Returns the given edge ``edge_canon`` oriented the same way (forward/reverse) as the ``like``
    edge.

    :param edge_canon: the edge to orient
    :param like: the edge to mimic the orientation from
    :return: the ``edge_canon`` oriented in the same manner as the edge ``like``
    """
    u, v, k = edge_canon
    return (v, u, k) if _is_reversed_vs_canon(like) else (u, v, k)
