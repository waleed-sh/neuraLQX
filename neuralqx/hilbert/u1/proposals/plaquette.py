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


"""Plaquette U(1) proposal kernels.

Plaquette proposals choose minimal graph loops and add or subtract one charge
step along every edge in the loop. Gauge-invariant spaces reconstruct slave
edges after each update to remain on the configured gauge-fixing slice.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.hilbert.u1.utils.arithmetic import modular_add
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.hilbert.utils import restore_trailing


def propose_plaquette_single(space: Any, key: jax.Array, states: Any) -> jax.Array:
    """Applies a plaquette update in one random gauge copy.

    Args:
        space: U(1) Hilbert space being updated.
        key: JAX pseudo-random key.
        states: Input state array.

    Returns:
        Proposed state array with original shape restored.
    """
    arr, single, batch = ensure_2d_trailing(states, space.size)
    _ensure_plaquettes(space)
    keys = jax.random.split(key, arr.shape[0])
    out = jax.vmap(lambda state, row_key: _single_state(space, state, row_key))(
        arr, keys
    )
    out = space._project_after_update(out)
    return restore_trailing(out, single, batch)


def propose_plaquette_all(space: Any, key: jax.Array, states: Any) -> jax.Array:
    """Applies independent plaquette updates in every gauge copy.

    Args:
        space: U(1) Hilbert space being updated.
        key: JAX pseudo-random key.
        states: Input state array.

    Returns:
        Proposed state array with original shape restored.
    """
    arr, single, batch = ensure_2d_trailing(states, space.size)
    _ensure_plaquettes(space)
    out = arr
    keys = jax.random.split(key, space.gauge_dimensions)
    for gauge, subkey in enumerate(keys):
        row_keys = jax.random.split(subkey, out.shape[0])
        out = jax.vmap(
            lambda state, row_key: plaquette_update(
                space,
                state,
                row_key,
                jnp.asarray(gauge, dtype=jnp.int32),
            )
        )(out, row_keys)
    out = space._project_after_update(out)
    return restore_trailing(out, single, batch)


def plaquette_update(
    space: Any,
    state: jax.Array,
    key: jax.Array,
    gauge: jax.Array,
) -> jax.Array:
    """Applies one random plaquette update to one flat state.

    Args:
        space: U(1) Hilbert space being updated.
        state: One flat state row.
        key: JAX pseudo-random key.
        gauge: Gauge-component index to update.

    Returns:
        Updated state row.
    """
    key_loop, key_step = jax.random.split(key)
    loop_id = jax.random.randint(
        key_loop,
        (),
        0,
        len(space._plaquettes),
        dtype=jnp.int32,
    )
    loop = jax.lax.dynamic_index_in_dim(
        space._plaquette_idx,
        loop_id,
        axis=0,
        keepdims=False,
    )
    mask = jax.lax.dynamic_index_in_dim(
        space._plaquette_mask,
        loop_id,
        axis=0,
        keepdims=False,
    )
    delta = _plaquette_delta(space, state, key_step)
    return jax.lax.fori_loop(
        0,
        space._plaquette_lmax,
        lambda component, current: _plaquette_component_update(
            space,
            current,
            gauge,
            loop,
            mask,
            delta,
            component,
        ),
        state,
    )


def _single_state(space: Any, state: jax.Array, key: jax.Array) -> jax.Array:
    """Applies a plaquette update to one row and one random gauge component."""
    key_gauge, key_update = jax.random.split(key)
    gauge = jax.random.randint(
        key_gauge,
        (),
        0,
        space.gauge_dimensions,
        dtype=jnp.int32,
    )
    return plaquette_update(space, state, key_update, gauge)


def _plaquette_delta(space: Any, state: jax.Array, key: jax.Array) -> jax.Array:
    """Samples the signed charge step used by a plaquette update."""
    step = jnp.asarray(space.q_step, dtype=state.dtype)
    return jax.random.choice(key, jnp.asarray([-step, step], dtype=state.dtype))


def _plaquette_component_update(
    space: Any,
    state: jax.Array,
    gauge: jax.Array,
    loop: jax.Array,
    mask: jax.Array,
    delta: jax.Array,
    component: int,
) -> jax.Array:
    """Updates one padded plaquette component when the mask is active."""
    edge = jnp.where(mask[component], loop[component], 0)
    site = gauge * space.tiny_size + edge
    return jax.lax.cond(
        mask[component],
        lambda active: _set_modular_site(space, active, site, delta),
        lambda inactive: inactive,
        state,
    )


def _set_modular_site(
    space: Any,
    state: jax.Array,
    site: jax.Array,
    delta: jax.Array,
) -> jax.Array:
    """Adds ``delta`` to one flat site with U(1) wrapping."""
    value = modular_add(
        state[site],
        delta,
        q_min=space.q_min,
        step=space.q_step,
        local_size=space.local_size,
        dtype=state.dtype,
    )
    return state.at[site].set(value)


def _ensure_plaquettes(space: Any) -> None:
    """Validates that the space exposes at least one plaquette."""
    if not space._plaquettes:
        raise NotImplementedError("No plaquettes available for this graph.")
