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


"""Plum-dispatched proposal moves for Hilbert states.

The public ``propose`` entry point routes generic flat-site moves and
domain-specific U(1) or SU(2) moves through local multiple dispatch. Generic
moves update local indices and then convert them back to state values.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from plum import dispatch

from neuralqx.hilbert.proposal.moves import AdjacentSiteUpdate
from neuralqx.hilbert.proposal.moves import UniformSiteUpdate
from neuralqx.hilbert.su2.proposals import AbstractSU2Move
from neuralqx.hilbert.su2.spaces import SU2GaugeInvariantHilbert
from neuralqx.hilbert.space import DiscreteHilbertSpace
from neuralqx.hilbert.u1.proposals import AbstractU1Move
from neuralqx.hilbert.u1.spaces import U1Hilbert
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.hilbert.utils import restore_trailing


def propose(
    space: DiscreteHilbertSpace,
    move: Any,
    key: jax.Array,
    states: Any,
) -> jax.Array:
    """Applies a proposal move using neuraLQX-local multiple dispatch.

    Args:
        space: Discrete Hilbert space whose states are being updated.
        move: Proposal move descriptor.
        key: JAX pseudo-random key.
        states: Input state array.

    Returns:
        Proposed state array with the same shape as ``states``.
    """
    return _nqx_propose(space, move, key, states)


@dispatch
def _nqx_propose(
    space: SU2GaugeInvariantHilbert,
    move: AbstractSU2Move,
    key: jax.Array,
    states: Any,
) -> jax.Array:
    """Delegates SU(2) move proposals to the SU(2) Hilbert implementation."""
    return space.propose(move, key, states)


@dispatch
def _nqx_propose(
    space: U1Hilbert,
    move: AbstractU1Move,
    key: jax.Array,
    states: Any,
) -> jax.Array:
    """Delegates U(1) move proposals to the U(1) Hilbert implementation."""
    return space.propose(move, key, states)


@dispatch
def _nqx_propose(
    space: DiscreteHilbertSpace,
    move: UniformSiteUpdate,
    key: jax.Array,
    states: Any,
) -> jax.Array:
    """Applies a uniform resampling move to generic flat sites."""
    arr, single, batch = ensure_2d_trailing(states, space.size)
    sites, local = _chosen_sites_and_local(space, key, arr, move.n_sites)
    k_values = jax.random.split(key, 2)[1]
    local_sizes = jnp.asarray(space.local_sizes, dtype=jnp.int32)
    selected_sizes = local_sizes[sites]
    current = jnp.take_along_axis(local, sites, axis=-1)

    if move.avoid_current:
        span = jnp.maximum(selected_sizes - 1, 1)
        raw = jax.random.randint(k_values, sites.shape, 0, span, dtype=jnp.int32)
        updates = jnp.where(selected_sizes <= 1, current, raw + (raw >= current))
    else:
        updates = jax.random.randint(
            k_values, sites.shape, 0, selected_sizes, dtype=jnp.int32
        )

    return _restore_proposal(space, arr, local, sites, updates, single, batch)


@dispatch
def _nqx_propose(
    space: DiscreteHilbertSpace,
    move: AdjacentSiteUpdate,
    key: jax.Array,
    states: Any,
) -> jax.Array:
    """Applies an adjacent-index move to generic flat sites."""
    arr, single, batch = ensure_2d_trailing(states, space.size)
    sites, local = _chosen_sites_and_local(space, key, arr, move.n_sites)
    k_direction = jax.random.split(key, 2)[1]
    local_sizes = jnp.asarray(space.local_sizes, dtype=jnp.int32)
    current = jnp.take_along_axis(local, sites, axis=-1)
    deltas = jnp.where(
        jax.random.bernoulli(k_direction, shape=sites.shape),
        1,
        -1,
    ).astype(jnp.int32)
    if move.boundary == "wrap":
        updates = (current + deltas) % local_sizes[sites]
    else:
        updates = jnp.clip(current + deltas, 0, local_sizes[sites] - 1)
    return _restore_proposal(space, arr, local, sites, updates, single, batch)


def _chosen_sites_and_local(
    space: DiscreteHilbertSpace,
    key: jax.Array,
    states: jax.Array,
    n_sites: int,
) -> tuple[jax.Array, jax.Array]:
    """Chooses distinct update sites and computes current local indices.

    Args:
        space: Discrete Hilbert space being updated.
        key: JAX pseudo-random key.
        states: Two-dimensional state array.
        n_sites: Number of distinct sites to choose per row.

    Returns:
        Pair containing chosen site indices and local-index states.
    """
    if n_sites > space.size:
        raise ValueError("n_sites cannot exceed Hilbert-space size.")
    keys = jax.random.split(jax.random.split(key, 2)[0], states.shape[0])
    sites = jax.vmap(
        lambda subkey: jax.random.choice(
            subkey, space.size, shape=(n_sites,), replace=False
        )
    )(keys).astype(jnp.int32)
    return sites, space.states_to_local_indices(states).astype(jnp.int32)


def _restore_proposal(
    space: DiscreteHilbertSpace,
    before: jax.Array,
    local: jax.Array,
    sites: jax.Array,
    updates: jax.Array,
    single: bool,
    batch: tuple[int, ...],
) -> jax.Array:
    """Restores proposed local indices to state values and shape.

    Args:
        space: Discrete Hilbert space being updated.
        before: Original two-dimensional states.
        local: Original two-dimensional local-index states.
        sites: Chosen sites for each row.
        updates: Proposed local-index values for ``sites``.
        single: Whether the original input was a single state.
        batch: Original batch shape before flattening.

    Returns:
        Proposed states restored to the caller's input shape. Invalid
        constrained proposals are replaced by the original rows.
    """
    rows = jnp.arange(before.shape[0])[:, None]
    proposed_local = local.at[rows, sites].set(updates)
    proposed = space.local_indices_to_states(proposed_local, dtype=before.dtype)
    if space.constrained:
        proposed = jnp.where(space.is_valid(proposed)[..., None], proposed, before)
    return restore_trailing(proposed, single, batch)


__all__ = ["propose"]
