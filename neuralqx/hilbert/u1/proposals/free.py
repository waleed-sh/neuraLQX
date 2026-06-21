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


"""Free-edge U(1) proposal kernels.

Free-edge proposals update only the independent base edges of a U(1) space.
Gauge-invariant spaces project the updated states back onto their constructive
gauge-fixing slice after the local updates are applied.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.hilbert.u1.utils.arithmetic import modular_add
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.hilbert.utils import restore_trailing

from .moves import FreeEdgeFlipAllGauge
from .moves import FreeEdgeFlipSingleGauge


def propose_free_single(
    space: Any,
    move: FreeEdgeFlipSingleGauge,
    key: jax.Array,
    states: Any,
    free_edges: tuple[int, ...],
) -> jax.Array:
    """Updates free variables from the pooled set of all gauge copies.

    Args:
        space: U(1) Hilbert space being updated.
        move: Free-edge move descriptor.
        key: JAX pseudo-random key.
        states: Input state array.
        free_edges: Base edges eligible for free updates.

    Returns:
        Proposed state array with original shape restored.
    """
    arr, single, batch = ensure_2d_trailing(states, space.size)
    candidates = tuple(
        gauge * space.tiny_size + edge
        for gauge in range(space.gauge_dimensions)
        for edge in free_edges
    )
    out = update_random_sites(space, arr, key, candidates, move.n_edges, move.adjacency)
    out = space._project_after_update(out)
    return restore_trailing(out, single, batch)


def propose_free_all(
    space: Any,
    move: FreeEdgeFlipAllGauge,
    key: jax.Array,
    states: Any,
    free_edges: tuple[int, ...],
) -> jax.Array:
    """Updates free variables independently inside each gauge copy.

    Args:
        space: U(1) Hilbert space being updated.
        move: Free-edge move descriptor.
        key: JAX pseudo-random key.
        states: Input state array.
        free_edges: Base edges eligible for free updates.

    Returns:
        Proposed state array with original shape restored.
    """
    arr, single, batch = ensure_2d_trailing(states, space.size)
    out = arr
    keys = jax.random.split(key, space.gauge_dimensions)
    for gauge, subkey in enumerate(keys):
        candidates = tuple(gauge * space.tiny_size + edge for edge in free_edges)
        out = update_random_sites(
            space, out, subkey, candidates, move.n_edges, move.adjacency
        )
    out = space._project_after_update(out)
    return restore_trailing(out, single, batch)


def update_random_sites(
    space: Any,
    states: jax.Array,
    key: jax.Array,
    candidates: tuple[int, ...],
    n_sites: int,
    adjacency: bool,
) -> jax.Array:
    """Applies random replacement or adjacent updates to chosen flat sites.

    Args:
        space: U(1) Hilbert space being updated.
        states: Two-dimensional state batch.
        key: JAX pseudo-random key.
        candidates: Flat sites eligible for updates.
        n_sites: Number of sites to choose per row.
        adjacency: Whether to use adjacent modular updates.

    Returns:
        Updated two-dimensional state batch.
    """
    if n_sites <= 0:
        return states
    if n_sites > len(candidates):
        raise ValueError(
            f"Cannot select {n_sites} U(1) sites from {len(candidates)} candidates."
        )
    candidate_sites = jnp.asarray(candidates, dtype=jnp.int32)
    keys = jax.random.split(key, states.shape[0])

    def update_one(state: jax.Array, subkey: jax.Array) -> jax.Array:
        """Updates one state row at randomly chosen candidate sites."""
        key_sites, key_values = jax.random.split(subkey)
        sites = jax.random.choice(
            key_sites,
            candidate_sites,
            shape=(n_sites,),
            replace=False,
        )
        values = (
            _adjacent_values(space, state, key_values, sites)
            if adjacency
            else _random_values(
                space,
                state,
                key_values,
                n_sites,
            )
        )
        return state.at[sites].set(values)

    return jax.vmap(update_one)(states, keys)


def _adjacent_values(
    space: Any, state: jax.Array, key: jax.Array, sites: jax.Array
) -> jax.Array:
    """Generates adjacent modular values for selected sites."""
    step = jnp.asarray(space.q_step, dtype=state.dtype)
    delta = jax.random.choice(
        key,
        jnp.asarray([-step, step], dtype=state.dtype),
        shape=sites.shape,
        replace=True,
    )
    return modular_add(
        state[sites],
        delta,
        q_min=space.q_min,
        step=space.q_step,
        local_size=space.local_size,
        dtype=state.dtype,
    )


def _random_values(
    space: Any, state: jax.Array, key: jax.Array, n_sites: int
) -> jax.Array:
    """Generates random local U(1) values for selected sites."""
    idx = jax.random.randint(
        key,
        (n_sites,),
        minval=0,
        maxval=space.local_size,
        dtype=jnp.int32,
    )
    return space.local_space.indices_to_values(idx, dtype=state.dtype)
