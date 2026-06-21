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


"""JIT entry points for U(1) sampling and gauge reconstruction.

The functions in this module are intentionally small wrappers around JAX-jitted
operations. They keep random U(1) sampling and constructive gauge reconstruction
compilable while allowing the public Hilbert methods to handle shape and dtype
normalization.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax

from neuralqx.hilbert.u1.constraints import reimpose_gauge_fixing_batch
from neuralqx.hilbert.utils import batch_shape


@partial(jax.jit, static_argnames=("space", "size", "dtype"))
def u1_random_state_jit(
    space: Any,
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: Any,
) -> jax.Array:
    """Samples unconstrained U(1) states from the scalar local range.

    Args:
        space: U(1) Hilbert space exposing ``local_space`` and ``size``.
        key: JAX pseudo-random key.
        size: Optional leading batch shape.
        dtype: Dtype for generated charge values.

    Returns:
        Random state array with trailing dimension ``space.size``.
    """
    return space.local_space.random_values(
        key,
        (*batch_shape(size), space.size),
        dtype=dtype,
    )


@partial(jax.jit, static_argnames=("space",))
def reimpose_gauge_fixing_jit(space: Any, states: jax.Array) -> jax.Array:
    """JIT wrapper for constructive gauge reconstruction.

    Args:
        space: Gauge-invariant U(1) Hilbert space.
        states: Two-dimensional state batch.

    Returns:
        State batch with slave edges recomputed from free edges.
    """
    return reimpose_gauge_fixing_batch(space, states)
