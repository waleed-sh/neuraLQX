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


"""Plaquette loop metadata for U(1) proposal kernels.

The U(1) plaquette proposal operates on graph minimal loops represented by
base edge indices. These helpers convert graph loops to padded JAX arrays that
can be used in jitted proposal kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from neuralqx.graph import AbstractGraph


def graph_plaquettes(graph: AbstractGraph) -> tuple[tuple[int, ...], ...]:
    """Resolves graph minimal loops to base edge-index plaquettes.

    Args:
        graph: Graph whose minimal loops define plaquettes.

    Returns:
        Tuple of non-empty plaquette edge-index tuples in the graph's edge
        order. The returned metadata is consumed by U(1) plaquette proposal
        kernels and cached on U(1) Hilbert spaces.
    """
    loops = []
    for loop in graph.minimal_loops():
        edges = loop.edges if hasattr(loop, "edges") else tuple(loop)
        loops.append(tuple(graph.edge_to_index(edge) for edge in edges))
    return tuple(loop for loop in loops if loop)


def build_plaquette_arrays(
    plaquettes: tuple[tuple[int, ...], ...],
    lmax: int,
) -> tuple[jax.Array, jax.Array]:
    """Builds padded plaquette edge-index and mask arrays.

    Args:
        plaquettes: Plaquette edge-index tuples.
        lmax: Maximum plaquette length used as the padded width.

    Returns:
        Padded edge-index array and boolean valid-entry mask.
    """
    idx = -jnp.ones((len(plaquettes), lmax), dtype=jnp.int32)
    mask = jnp.zeros((len(plaquettes), lmax), dtype=jnp.bool_)
    for row, loop in enumerate(plaquettes):
        width = len(loop)
        idx = idx.at[row, :width].set(jnp.asarray(loop, dtype=jnp.int32))
        mask = mask.at[row, :width].set(True)
    return idx, mask
