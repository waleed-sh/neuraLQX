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
Plum dispatch overload for ``random_state`` on U(1) constrained Hilbert spaces.
"""

from plum import dispatch

import jax
import jax.numpy as jnp

from neuralqx.utils.jit import mod_sum_jax
from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace
from ..unconstrained_core import UnconstrainedHilbertU1Core
from ..constrained_core import ConstrainedHilbertU1Core


@dispatch
def random_state(
    space: AbstractHilbertSpace,
    key: jax.Array,
    *,
    size: int = 1,
) -> jax.Array:
    """
    Generic random state generator for unconstrained Hilbert spaces.

    This overload exists as a compatibility default for AbstractHilbertSpace
    subclasses that do not register a more specific operation overload.
    """

    qn = jnp.asarray(space.allowed_basis_states.all_states())
    ridx = jax.random.randint(key, (size, int(space.size)), 0, qn.shape[0])
    return qn[ridx].astype(space.dtype)


@dispatch
def random_state(
    space: UnconstrainedHilbertU1Core,
    key: jax.Array,
    *,
    size: int = 1,
) -> jax.Array:
    """
    Random state generator for unconstrained U(1) Hilbert spaces.

    Samples each site i.i.d. from ``space.allowed_basis_states``.
    """

    qn = jnp.asarray(space.allowed_basis_states.all_states())
    ridx = jax.random.randint(key, (size, int(space.size)), 0, qn.shape[0])
    return qn[ridx].astype(space.dtype)


@dispatch
def random_state(
    space: ConstrainedHilbertU1Core, key: jax.Array, *, size: int = 1
) -> jax.Array:
    """
    Gauge-invariant random state generator for
    :class:`neuralqx.hilbert.u1.constrained_core.ConstrainedHilbertU1Core`.

    This overload plugs into the global ``random_state`` multimethod (via Plum dispatch)
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
