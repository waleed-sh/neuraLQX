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

"""
Plum dispatch overload for ``flip_state`` on U(1) constrained Hilbert spaces.
"""

from plum import dispatch

import jax
import jax.numpy as jnp

from neuralqx.utils.misc.arithmetic import mod_add
from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace
from ..unconstrained_core import UnconstrainedHilbertU1Core
from ..constrained_core import ConstrainedHilbertU1Core


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
    Generic flip proposal for unconstrained Hilbert spaces.

    ``scope`` is accepted for API compatibility but ignored.
    """

    single = sigma.ndim == 1
    if single:
        sigma = sigma[None, :]

    B = int(sigma.shape[0])
    N = int(space.size)

    if number_of_edges < 1 or number_of_edges > N:
        raise ValueError(f"number_of_edges must be in [1, {N}]")

    qn = jnp.asarray(space.allowed_basis_states.all_states())

    q_min = getattr(space, "q_min", None)
    q_max = getattr(space, "q_max", None)
    q_step = getattr(space, "q_step", None)

    def _flip_one(s: jax.Array, k: jax.Array) -> jax.Array:
        k_idx, k_val = jax.random.split(k, 2)
        idx = jax.random.choice(k_idx, N, shape=(number_of_edges,), replace=False)

        if adjacency:
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

        ridx = jax.random.randint(k_val, (number_of_edges,), 0, qn.shape[0])
        vals = qn[ridx].astype(s.dtype)
        return s.at[idx].set(vals)

    keys = jax.random.split(key, B)
    out = jax.vmap(_flip_one)(sigma, keys)
    return out[0] if single else out


@dispatch
def flip_state(
    space: UnconstrainedHilbertU1Core,
    sigma: jax.Array,
    key: jax.Array,
    *,
    number_of_edges: int = 1,
    adjacency: bool = False,
    scope: str = "single",
) -> jax.Array:
    """
    Flip proposal for unconstrained U(1) Hilbert spaces.

    ``scope`` is accepted for API compatibility but ignored.
    """

    single = sigma.ndim == 1
    if single:
        sigma = sigma[None, :]

    B = int(sigma.shape[0])
    N = int(space.size)

    if number_of_edges < 1 or number_of_edges > N:
        raise ValueError(f"number_of_edges must be in [1, {N}]")

    qn = jnp.asarray(space.allowed_basis_states.all_states())

    q_min = getattr(space, "q_min", None)
    q_max = getattr(space, "q_max", None)
    q_step = getattr(space, "q_step", None)

    def _flip_one(s: jax.Array, k: jax.Array) -> jax.Array:
        k_idx, k_val = jax.random.split(k, 2)
        idx = jax.random.choice(k_idx, N, shape=(number_of_edges,), replace=False)

        if adjacency:
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

        ridx = jax.random.randint(k_val, (number_of_edges,), 0, qn.shape[0])
        vals = qn[ridx].astype(s.dtype)
        return s.at[idx].set(vals)

    keys = jax.random.split(key, B)
    out = jax.vmap(_flip_one)(sigma, keys)
    return out[0] if single else out


@dispatch
def flip_state(
    space: ConstrainedHilbertU1Core,
    sigma: jax.Array,
    key: jax.Array,
    *,
    number_of_edges: int = 1,
    adjacency: bool = False,
    scope: str = "single",
) -> jax.Array:
    """
    Gauge-invariant flip move for
    :class:`neuralqx.hilbert.u1.constrained_core.ConstrainedHilbertU1Core`.

    This overload plugs into the global ``flip_state`` multimethod (via Plum dispatch).
    It performs a proposal move inside the gauge-invariant subspace by:

    1) Reshaping the input state(s) into ``v = space.view(sigma)`` with shape ``(B, G, E)``.
    2) Modifying only free edges (independent degrees of freedom) according to ``scope``.
    3) Flattening back to ``(B, G*E)`` and calling ``space._reimpose_gauge_fixing`` to
       deterministically recompute all slave edges in topological order, ensuring gauge invariance
       after the flip.

    Two scopes are supported:

    - ``scope="single"``:
      Flips ``number_of_edges`` many free-edge sites chosen from the union of all gauge copies.

    - ``scope="all_copies"``:
      Flips one free edge per gauge copy (so ``number_of_edges`` must be 1).

    :param space: The constrained U(1) Hilbert core.
    :param sigma: State or batch of states to be modified. Shape ``(N,)`` or ``(B,N)``.
    :param key: JAX PRNGKey. One independent subkey is used per batch element.
    :param number_of_edges: Number of free-edge sites to modify.
    :param adjacency: If True, apply a ± step move modulo the allowed range; otherwise assign a new
        random allowed value.
    :param scope: Either ``"single"`` or ``"all_copies"``.

    :return: Gauge-invariant state(s) after the flip, with the same batching as the input.

    :raises ValueError: If ``scope="all_copies"`` and ``number_of_edges != 1``.
    """

    single = sigma.ndim == 1
    if single:
        sigma = sigma[None, :]

    gf = space.gauge_fixing
    E = space.tiny_size
    G = space.gauge_dimensions
    qn = jnp.asarray(space.allowed_basis_states.all_states())

    v = space.view(sigma)
    B = v.shape[0]
    keys = jax.random.split(key, B)

    free_idx = jnp.asarray(gf.free, dtype=jnp.int32)

    def _flip_one(vb, kb):
        k_idx, k_val = jax.random.split(kb, 2)

        if scope == "all_copies":
            if number_of_edges != 1:
                raise ValueError(
                    "scope='all_copies' currently supports number_of_edges=1 only."
                )
            # pick one free edge index per gauge copy
            pick = jax.random.choice(k_idx, free_idx, shape=(G,), replace=True)
            if adjacency:
                deltas = jax.random.choice(
                    k_val, jnp.array([-1, 1]), shape=(G,), replace=True
                )
                for gd in range(G):
                    old = vb[gd, pick[gd]]
                    new = mod_add(
                        old,
                        deltas[gd],
                        q_min=space.q_min,
                        q_max=space.q_max,
                        step=space.q_step,
                    )
                    vb = vb.at[gd, pick[gd]].set(new.astype(vb.dtype))
            else:
                ridx = jax.random.randint(k_val, (G,), 0, qn.shape[0])
                vals = qn[ridx].astype(vb.dtype)
                for gd in range(G):
                    vb = vb.at[gd, pick[gd]].set(vals[gd])
            return vb

        # scope == "single": pick edges from flattened union (gd, free_edge)
        # choose number_of_edges pairs (gd, f)
        gd = jax.random.randint(k_idx, (number_of_edges,), 0, G)
        fpos = jax.random.randint(k_idx, (number_of_edges,), 0, free_idx.shape[0])
        picked_edges = free_idx[fpos]

        if adjacency:
            deltas = jax.random.choice(
                k_val, jnp.array([-1, 1]), shape=(number_of_edges,), replace=True
            )
            for i in range(number_of_edges):
                old = vb[gd[i], picked_edges[i]]
                new = mod_add(
                    old,
                    deltas[i],
                    q_min=space.q_min,
                    q_max=space.q_max,
                    step=space.q_step,
                )
                vb = vb.at[gd[i], picked_edges[i]].set(new.astype(vb.dtype))
        else:
            ridx = jax.random.randint(k_val, (number_of_edges,), 0, qn.shape[0])
            vals = qn[ridx].astype(vb.dtype)
            for i in range(number_of_edges):
                vb = vb.at[gd[i], picked_edges[i]].set(vals[i])
        return vb

    v2 = jax.vmap(_flip_one)(v, keys)
    out = space.flatten(v2)
    out = space.reimpose_gauge_fixing(out)
    return out[0] if single else out
