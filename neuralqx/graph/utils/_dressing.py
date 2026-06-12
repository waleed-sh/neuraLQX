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


"""Loop dressing helpers for graph-derived operators.

This module annotates minimal loops with creation and annihilation metadata.
The metadata records whether each loop edge follows the stored graph
orientation or uses the reverse orientation.
"""

from neuralqx.utils.typing import GraphEdges
from neuralqx.utils.typing import Loop
from neuralqx.utils.typing import MinimalLoop

from ._mapping import reverse_edge


def dress_loops(
    loops: list[MinimalLoop],
    graph_edges: GraphEdges,
) -> list[Loop]:
    """Adds creation/annihilation metadata to minimal loops.

    Args:
        loops: Minimal primal loops to annotate.
        graph_edges: Canonical graph edges that define the stored orientation.

    Returns:
        New ``Loop`` objects with a dressing entry for every loop edge. Stored
        orientations are marked as creation actions, while reversed
        orientations are marked as annihilation actions.
    """
    stored = set(graph_edges)
    reversed_stored = {reverse_edge(edge) for edge in graph_edges}
    dressed: list[Loop] = []

    for loop in loops:
        loop_dressing = []
        for edge in loop:
            if edge in stored:
                edge_type = "creation"
            elif edge in reversed_stored:
                edge_type = "annihilation"
            else:
                edge_type = "annihilation"
            loop_dressing.append({"type": edge_type, "key": edge.key})
        dressed.append(loop.with_dressing(loop_dressing))

    return dressed


__all__ = [
    "dress_loops",
]
