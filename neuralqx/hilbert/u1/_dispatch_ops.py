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

from neuralqx.utils.jit import mod_sum_jax
from neuralqx.utils.misc.arithmetic import mod_add
from .constrained_core import ConstrainedHilbertU1Core


@dispatch
def random_state(
    space: ConstrainedHilbertU1Core, key: jax.Array, *, size: int = 1
) -> jax.Array:
    """
    Gauge-invariant random state generator for
    :class:`neuralqx.hilbert.u1._constrained_core.ConstrainedHilbertU1Core`.

    This overload plugs into the global `state_ops.random_state` multimethod (via Plum dispatch)
    and implements gauge-invariant sampling by construction:

    1) Sample only the free (independent) edges uniformly from the allowed local basis labels,
       for every gauge copy.
    2) Construct a partially-filled configuration tensor ``v`` of shape ``(B, G, E)`` where:
       - ``B = size`` (batch size),
       - ``G = space.gauge_dimensions`` (number of gauge copies),
       - ``E = space.tiny_size`` (edges per gauge copy).
    3) Deterministically fill all slave edges in topological order using the gauge-fixing relations
       stored in ``space.gauge_fixing``.
    4) Flatten back to NetKet layout ``(B, G*E)`` using ``space.flatten``.

    This ensures that every returned state satisfies the gauge-fixing constraints exactly (up to the
    modular arithmetic defined by ``q_min/q_max/q_step``), without relying on rejection sampling.

    Important implementation notes:
    - The constraint application loop runs in Python. This is fine for moderate constraint sizes and
      typical usage, but it is not intended to be aggressively JIT-optimized as-is. If you need this
      to be fully JIT-friendly, store fixers as JAX arrays with static shapes and use
      `jax.lax.fori_loop` (or precompiled scan) instead of a Python loop.
    - The modular reduction is performed by ``mod_sum_jax`` over the last axis of the fixer
      contributions (the per-slave dependency list).

    :param space: The constrained U(1) Hilbert core. Must expose:
        - ``allowed_basis_states`` (NetKet StaticRange-like),
        - ``dtype``,
        - ``gauge_fixing`` with fields ``free``, ``slave_topo``, ``slave_fixers``,
        - ``tiny_size``, ``gauge_dimensions``,
        - ``flatten(v)`` converting ``(B,G,E)`` to ``(B,G*E)``.
    :param key: JAX PRNGKey used to sample free-edge values.
    :param size: Number of states to generate (batch size). The output is always batched.

    :return: A batch of gauge-invariant states of shape ``(size, space.size)`` in the standard block
        layout ``[copy0 | copy1 | ...]``, where each copy has ``E = space.tiny_size`` sites.
    """

    # (local_size,)
    qn = jnp.asarray(space.allowed_basis_states.all_states())
    gf = space.gauge_fixing

    E = space.tiny_size
    G = space.gauge_dimensions
    F = len(gf.free)

    # sample free values for all gauge copies at once: shape (B,G,F)
    k = key
    ridx = jax.random.randint(k, (size, G, F), 0, qn.shape[0])
    free_vals = qn[ridx].astype(space.dtype)

    # build empty (B,G,E) and set free
    v = jnp.zeros((size, G, E), dtype=space.dtype)
    free_idx = jnp.asarray(gf.free, dtype=jnp.int32)
    v = v.at[:, :, free_idx].set(free_vals)

    # fill slaves
    for slave, fixers in zip(gf.slave_topo, gf.slave_fixers):
        dep_idx = jnp.asarray([d for (d, _s) in fixers], dtype=jnp.int32)
        signs = jnp.asarray([s for (_d, s) in fixers], dtype=v.dtype)
        contrib = v[:, :, dep_idx] * signs[None, None, :]
        val = mod_sum_jax(
            contrib, q_min=space.q_min, q_max=space.q_max, step=space.q_step
        )
        v = v.at[:, :, slave].set(val.astype(v.dtype))

    return space.flatten(v)


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
    :class:`neuralqx.hilbert.u1._constrained_core.ConstrainedHilbertU1Core`.

    This overload plugs into the global `state_ops.flip_state` multimethod (via Plum dispatch).
    It performs a proposal move inside the gauge-invariant subspace by:

    1) Reshaping the input state(s) into ``v = space.view(sigma)`` with shape ``(B, G, E)``.
    2) Modifying only free edges (independent degrees of freedom) according to ``scope``.
    3) Flattening back to ``(B, G*E)`` and calling ``space._reimpose_gauge_fixing`` to
       deterministically recompute all slave edges in topological order, ensuring gauge invariance
       after the flip.

    Two scopes are supported:

    - ``scope="single"``:
      Flips ``number_of_edges`` many free-edge sites chosen from the union of all gauge copies.
      Concretely, each flip selects a pair ``(gauge_copy, free_edge_index)`` and updates exactly
      that single local degree of freedom. (Note: as implemented, the selections are drawn with
      replacement across flips, so the same site may be hit multiple times when
      ``number_of_edges > 1``.)

    - ``scope="all_copies"``:
      Flips one free edge per gauge copy (so ``number_of_edges`` must be 1). Each gauge copy picks
      its own free edge and assigns it a new value (or applies an adjacency step). This is the
      "parallel flip across gauge copies" mode.

    Adjacency vs random replacement:

    - If ``adjacency=False``: each targeted site is assigned a new value sampled uniformly from
      ``space.allowed_basis_states.all_states()``.
    - If ``adjacency=True``: each targeted site is incremented/decremented modulo the allowed range
      using ``mod_add``.
      Note: this implementation currently samples deltas from ``{-1, +1}``. If the space uses a
      non-unit step (``space.q_step != 1``), you likely want deltas from ``{-q_step, +q_step}``
      instead.

    Shape conventions:
    - Input ``sigma`` may be a single state of shape ``(N,)`` or a batch of states ``(B, N)`` where
      ``N = space.size = G*E``.
    - Output matches the input batching: returns ``(N,)`` if given a single state, else ``(B, N)``.

    Performance/JAX notes:
    - The inner per-state update uses Python `for` loops to apply multiple flips and to loop over
      gauge copies in ``scope="all_copies"``. This is clear and correct, but it is not the most
      JIT-friendly pattern. If you intend to `jax.jit` this path heavily, consider rewriting the
      update logic using `jax.lax.fori_loop`/scatter updates with vectorized indexing.

    :param space: The constrained U(1) Hilbert core. Must expose:
        - ``gauge_fixing.free``,
        - ``view(sigma)`` and ``flatten(v)``,
        - modular metadata ``q_min``, ``q_max``, ``q_step``,
        - ``allowed_basis_states.all_states()``,
        - ``_reimpose_gauge_fixing(states_flat)`` returning gauge-invariant states.
    :param sigma: State or batch of states to be modified. Shape ``(N,)`` or ``(B,N)``.
    :param key: JAX PRNGKey. One independent subkey is used per batch element.
    :param number_of_edges: Number of free-edge sites to modify.
        In ``scope="all_copies"`` this must be 1 (one flip per gauge copy).
    :param adjacency: If True, apply a ± step move modulo the allowed range; otherwise assign a new
        random allowed value.
    :param scope: Either ``"single"`` (flip within the union of free edges across all gauge copies)
        or ``"all_copies"`` (flip one free edge per gauge copy).

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
