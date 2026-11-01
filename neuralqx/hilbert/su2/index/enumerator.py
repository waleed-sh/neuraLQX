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


"""Exact enumeration for small SU(2) spin-network Hilbert spaces.

Exact enumeration is intended for validation, small examples, and indexable
test systems. It walks all edge doubled-spin sectors and appends every
admissible left-comb channel assignment.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from neuralqx.hilbert.su2.errors import SU2ExactEnumerationTooLargeError
from neuralqx.hilbert.su2.layout import SpinNetworkLayout
from neuralqx.hilbert.su2.metadata import SU2GraphMetadata
from neuralqx.hilbert.su2.recoupling import all_left_comb_channels


def enumerate_exact_states(
    *,
    metadata: SU2GraphMetadata,
    layout: SpinNetworkLayout,
    two_j_max: int,
    max_states: int = 100_000,
) -> np.ndarray:
    """Materializes all gauge-invariant basis states for small systems.

    Args:
        metadata: Compiled graph incidence metadata.
        layout: Spin-network flat-state layout.
        two_j_max: Maximum doubled-spin edge label.
        max_states: Safety limit for the number of returned states.

    Returns:
        Two-dimensional NumPy array of exact valid basis states.

    Raises:
        SU2ExactEnumerationTooLargeError: If enumeration exceeds
            ``max_states``.
    """
    states: list[np.ndarray] = []
    edge_domains = (range(int(two_j_max) + 1),) * metadata.n_edges
    for edge_values in product(*edge_domains):
        choices_by_vertex: list[tuple[tuple[int, ...], ...]] = []
        for vertex, edge_sites in enumerate(metadata.incident_edges_by_vertex):
            local_edges = tuple(edge_values[edge] for edge in edge_sites)
            maxima = layout.vertex_channel_max_labels[vertex]
            choices = all_left_comb_channels(local_edges, maxima)
            if not choices:
                break
            choices_by_vertex.append(choices)
        else:
            for vertex_channels in product(*choices_by_vertex):
                state = np.zeros(layout.size, dtype=np.int32)
                state[: metadata.n_edges] = np.asarray(edge_values, dtype=np.int32)
                for vertex, channels in enumerate(vertex_channels):
                    for site, value in zip(
                        layout.vertex_channel_sites[vertex],
                        channels,
                        strict=True,
                    ):
                        state[site] = int(value)
                states.append(state)
                if len(states) > int(max_states):
                    raise SU2ExactEnumerationTooLargeError(int(max_states))
    if not states:
        return np.zeros((0, layout.size), dtype=np.int32)
    return np.stack(states, axis=0)


__all__ = ["enumerate_exact_states"]
