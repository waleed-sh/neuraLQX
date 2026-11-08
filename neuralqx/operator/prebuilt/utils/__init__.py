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


"""Utilities shared by shipped prebuilt operators."""

from .graph import edge_emanates_from
from .graph import graph_vertex
from .graph import incident_edge_tuples
from .graph import k_vertex
from .graph import orientation_sign
from .graph import other_endpoint
from .graph import reduce_first_pair_permutations
from .loops import canonical_loop_edges
from .loops import loop_edges_at_vertex
from .loops import loops_containing_edges
from .loops import parse_loop
from .loops import type_to_step
from .triangulation import Thiemann2p1Term
from .triangulation import Thiemann3p1Term
from .triangulation import admissible_thiemann_3p1_vertices
from .triangulation import collect_thiemann_2p1_terms
from .triangulation import collect_thiemann_3p1_terms
from .u1 import canonical_edge
from .u1 import copy_sites
from .u1 import edge_index
from .u1 import noncyclic_shift_site
from .u1 import q_bounds
from .u1 import shift_sites
from .u1 import site_index
from .u1 import wrap_add_value
from .validation import require_gauge_dimensions
from .validation import require_half_ladder_graph
from .validation import require_nonplanar_graph
from .validation import require_planar_graph
from .validation import require_single_vertex_three_edges
from .validation import require_u1_hilbert
from .validation import validate_copy
from .validation import validate_site

__all__ = [
    "Thiemann2p1Term",
    "Thiemann3p1Term",
    "admissible_thiemann_3p1_vertices",
    "canonical_edge",
    "canonical_loop_edges",
    "collect_thiemann_2p1_terms",
    "collect_thiemann_3p1_terms",
    "copy_sites",
    "edge_emanates_from",
    "edge_index",
    "graph_vertex",
    "incident_edge_tuples",
    "k_vertex",
    "loop_edges_at_vertex",
    "loops_containing_edges",
    "noncyclic_shift_site",
    "orientation_sign",
    "other_endpoint",
    "parse_loop",
    "q_bounds",
    "reduce_first_pair_permutations",
    "require_gauge_dimensions",
    "require_half_ladder_graph",
    "require_nonplanar_graph",
    "require_planar_graph",
    "require_single_vertex_three_edges",
    "require_u1_hilbert",
    "shift_sites",
    "site_index",
    "type_to_step",
    "validate_copy",
    "validate_site",
    "wrap_add_value",
]
