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

from __future__ import annotations


from plum import dispatch

import jax
import jax.numpy as jnp

from neuralqx.utils.misc.arithmetic import mod_add
from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace


@dispatch
def flip_state(
    space: AbstractHilbertSpace,
    sigma: jax.Array,
    key: jax.Array,
    *,
    number_of_edges: int = 1,
    adjacency: bool = False,
    scope: str = "single",
) -> jax.Array:
    """
    Default flip proposal for unconstrained Hilbert spaces.

    This function proposes new configurations by modifying a subset of sites ("edges") in the
    provided state(s). It is designed for unconstrained spaces where local changes do not require
    subsequent global repair steps (unlike gauge-fixed / constrained spaces, which must reimpose
    constraints after any free-degree change).

    The function supports two flipping modes:

    - ``adjacency=False`` (default):
      Each selected site is assigned a new value sampled uniformly from the set of allowed basis
      states (as provided by ``space.allowed_basis_states``).

    - ``adjacency=True``:
      Each selected site is incremented or decremented by one "step". If the space exposes
      ``q_min``, ``q_max``, and ``q_step`` attributes, the update is performed using modular
      addition via ``mod_add`` so the value remains inside the allowed range. Otherwise, the update
      falls back to plain addition.

    Batching behavior:
    - If ``sigma`` has shape ``(N,)``, it is treated as a single configuration and returned
      with shape ``(N,)``.
    - If ``sigma`` has shape ``(B, N)``, each configuration in the batch is flipped
      independently using a distinct PRNG subkey, and the returned array has shape ``(B, N)``.

    Note:
    - The indices to flip are sampled without replacement, so the same site will not be
      chosen twice within a single proposal.
    - When ``adjacency=False``, the "new value" may coincide with the old value unless the
      caller enforces otherwise. This is typically acceptable for generic proposals but can
      be tightened if needed (at some computational cost).
    - The ``scope`` argument is accepted for interface compatibility with constrained cores,
      but it is not used here.

    :param space: The Hilbert space core that defines the configuration size and allowed
        basis states.
    :param sigma: A single configuration of shape ``(N,)`` or a batch of configurations of
        shape ``(B, N)``.
    :param key: A JAX PRNG key used to sample flip locations and new values.
    :param number_of_edges: The number of sites to modify in each configuration. Must satisfy
        ``1 <= number_of_edges <= space.size``.
    :param adjacency: If ``True``, perform a ±step update (optionally modular). If ``False``,
        assign random allowed values.
    :param scope: Ignored. Present only to maintain compatibility with higher-level interfaces
        that pass a scope string.
    :return: A configuration (or batch) with the requested number of sites modified. The shape
        matches the input shape: ``(N,)`` for a single input state, otherwise ``(B, N)``.
    :raises ValueError: If ``number_of_edges`` is not in the valid range.
    """

    single = sigma.ndim == 1
    if single:
        sigma = sigma[None, :]

    B = int(sigma.shape[0])
    N = int(space.size)

    if number_of_edges < 1 or number_of_edges > N:
        raise ValueError(f"number_of_edges must be in [1, {N}]")

    qn = jnp.asarray(space.allowed_basis_states.all_states())

    # step/range info (U1-like spaces have these)
    q_min = getattr(space, "q_min", None)
    q_max = getattr(space, "q_max", None)
    q_step = getattr(space, "q_step", None)

    def _flip_one(s: jax.Array, k: jax.Array) -> jax.Array:
        k_idx, k_val = jax.random.split(k, 2)
        idx = jax.random.choice(k_idx, N, shape=(number_of_edges,), replace=False)

        if adjacency:
            # +/- step
            step = jnp.asarray(q_step if q_step is not None else 1, dtype=s.dtype)
            deltas = jax.random.choice(
                k_val,
                jnp.array([-step, step], dtype=s.dtype),
                shape=(number_of_edges,),
                replace=True,
            )

            if (q_min is not None) and (q_max is not None) and (q_step is not None):
                new_vals = mod_add(
                    s[idx], deltas, q_min=q_min, q_max=q_max, step=q_step
                ).astype(s.dtype)
            else:
                new_vals = (s[idx] + deltas).astype(s.dtype)

            return s.at[idx].set(new_vals)

        # random allowed value
        ridx = jax.random.randint(k_val, (number_of_edges,), 0, qn.shape[0])
        vals = qn[ridx].astype(s.dtype)
        return s.at[idx].set(vals)

    keys = jax.random.split(key, B)
    out = jax.vmap(_flip_one)(sigma, keys)
    return out[0] if single else out
