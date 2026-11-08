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


"""Internal graph utilities shared by concrete graph implementations.

This subpackage contains the normalization, cycle detection, dual-graph,
embedding, orientation-sign, and plotting helpers that support the public graph
API. The names are re-exported here for graph-package internals, while
``plot_graph`` is loaded lazily so importing ``neuralqx.graph`` does not require
matplotlib unless plotting is actually requested.
"""

from typing import TYPE_CHECKING
from typing import Any

from ._cycles import all_cycles
from ._cycles import canonical_rotate
from ._cycles import minimal_cycles
from ._cycles import minimal_raw_cycles
from ._dressing import dress_loops
from ._dual import build_adjacency_from_edge_pairs
from ._dual import build_directed_connectivity
from ._dual import build_dual_edges
from ._dual import build_incidence
from ._embedding import dual_positions
from ._embedding import graph_positions
from ._embedding import randomize_vertices
from ._mapping import build_edge_mapping
from ._mapping import edge_to_index
from ._mapping import invert_edge_mapping
from ._mapping import reverse_edge
from ._parsing import find_unique_vertices
from ._signs import compute_graph_signs
from ._signs import compute_raw_signs
from ._validate import ensure_edge_keys
from ._validate import validate_vertices

if TYPE_CHECKING:
    from ._plotting import plot_graph


def __getattr__(name: str) -> Any:
    """Lazily exposes plotting helpers without importing matplotlib eagerly."""
    if name == "plot_graph":
        from ._plotting import plot_graph

        return plot_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "all_cycles",
    "build_adjacency_from_edge_pairs",
    "build_directed_connectivity",
    "build_dual_edges",
    "build_edge_mapping",
    "build_incidence",
    "canonical_rotate",
    "compute_graph_signs",
    "compute_raw_signs",
    "dress_loops",
    "dual_positions",
    "edge_to_index",
    "ensure_edge_keys",
    "find_unique_vertices",
    "graph_positions",
    "invert_edge_mapping",
    "minimal_cycles",
    "minimal_raw_cycles",
    "plot_graph",
    "randomize_vertices",
    "reverse_edge",
    "validate_vertices",
]
