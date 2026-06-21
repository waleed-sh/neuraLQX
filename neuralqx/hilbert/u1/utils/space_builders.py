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


"""Construction helpers for homogeneous U(1) spaces.

These helpers convert scalar charge ranges into edge-local degree-of-freedom
declarations and cache plaquette metadata used by U(1) proposal kernels.
"""

from __future__ import annotations

from typing import Any

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.local import LocalRange
from neuralqx.hilbert.local import VectorRange
from neuralqx.hilbert.u1.proposals import build_plaquette_arrays
from neuralqx.hilbert.u1.proposals import graph_plaquettes


def edge_space_for_gauge_dimension(
    gauge_dimensions: int,
    local_space: LocalRange,
) -> LocalRange | VectorRange:
    """Returns a scalar or vector edge local space for a U(1)^N field.

    Args:
        gauge_dimensions: Number of independent U(1) components per edge.
        local_space: Scalar local charge range.

    Returns:
        ``local_space`` for a single component, or a ``VectorRange`` with one
        scalar component per gauge dimension.
    """
    if int(gauge_dimensions) == 1:
        return local_space
    return VectorRange(local_space, int(gauge_dimensions))


def plaquette_metadata(
    graph: AbstractGraph,
) -> tuple[tuple[tuple[int, ...], ...], int, Any, Any]:
    """Resolves graph loops and padded metadata for JAX plaquette moves.

    Args:
        graph: Graph whose minimal loops define plaquette proposals.

    Returns:
        Tuple containing loop edge-index tuples, maximum loop length, padded
        index array, and valid-entry mask.
    """
    plaquettes = graph_plaquettes(graph)
    lmax = max((len(loop) for loop in plaquettes), default=0)
    idx, mask = build_plaquette_arrays(plaquettes, lmax)
    return plaquettes, lmax, idx, mask
