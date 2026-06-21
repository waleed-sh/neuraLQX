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


"""Constructive projection onto a U(1) gauge-fixing slice.

Gauge-invariant U(1) spaces keep free-edge values and reconstruct every slave
edge from the configured dependency topology. These helpers operate on
two-dimensional state batches and are used from JIT-compiled wrappers.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.hilbert.u1.utils.arithmetic import signed_modular_sum


def reimpose_gauge_fixing_batch(space: Any, states: jax.Array) -> jax.Array:
    """Recomputes slave edges from free edges for a two-dimensional batch.

    Args:
        space: Gauge-invariant U(1) Hilbert space.
        states: Two-dimensional state batch.

    Returns:
        State batch with every slave edge reconstructed in topological order.
    """
    out = states
    for slave, fixers in zip(space.topology.slave_topo, space.topology.slave_fixers):
        out = _set_slave_for_all_gauges(space, out, slave, fixers)
    return out


def _set_slave_for_all_gauges(
    space: Any,
    states: jax.Array,
    slave: int,
    fixers: tuple[tuple[int, int], ...],
) -> jax.Array:
    """Sets one slave edge for every gauge component.

    Args:
        space: Gauge-invariant U(1) Hilbert space.
        states: Two-dimensional state batch.
        slave: Base edge index reconstructed by the relation.
        fixers: Signed source base edges.

    Returns:
        Updated state batch with the slave edge assigned for all gauge
        dimensions.
    """
    out = states
    for gauge in range(space.gauge_dimensions):
        lhs_site = gauge * space.tiny_size + slave
        rhs_sites = jnp.asarray(
            [gauge * space.tiny_size + edge for edge, _sign in fixers],
            dtype=jnp.int32,
        )
        signs = jnp.asarray([sign for _edge, sign in fixers], dtype=out.dtype)
        value = signed_modular_sum(
            jnp.take(out, rhs_sites, axis=-1),
            signs,
            q_min=space.q_min,
            step=space.q_step,
            local_size=space.local_size,
            dtype=out.dtype,
        )
        out = out.at[:, lhs_site].set(value)
    return out
