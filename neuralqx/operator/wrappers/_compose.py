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


"""JAX composition kernels shared by product-like wrappers."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator


def product_max_conn_size(operators: tuple[ComputationalOperator, ...]) -> int:
    """Return the padded connection size for a product of operators."""
    return int(math.prod(int(operator.max_conn_size) for operator in operators))


def apply_product_batch(
    operators: tuple[ComputationalOperator, ...],
    states_2d: jax.Array,
    *,
    coefficient: Any,
    dtype: Any,
) -> tuple[jax.Array, jax.Array]:
    """Apply an algebraic product to a rank-2 state batch.

    ``operators`` are stored left-to-right as written in mathematics. Therefore
    ``(A @ B) |sigma>`` applies ``B`` to the input states first, then ``A`` to
    every branch produced by ``B``.
    """
    states = jnp.asarray(states_2d)
    n_states = states.shape[0]
    hilbert_size = states.shape[1]
    max_conn_size = product_max_conn_size(operators)
    if n_states == 0:
        return (
            jnp.empty((0, max_conn_size, hilbert_size), dtype=states.dtype),
            jnp.empty((0, max_conn_size), dtype=dtype),
        )

    branch_states = states
    branch_mels = jnp.ones((n_states, 1), dtype=dtype)

    for operator in reversed(operators):
        n_branches = branch_states.shape[0] // n_states
        x_primes, mels = operator._get_conn_padded_batch_kernel(branch_states)
        branch_states = x_primes.reshape(
            (n_states, n_branches * int(operator.max_conn_size), hilbert_size)
        )
        branch_mels = (
            branch_mels.reshape((n_states, n_branches, 1))
            * mels.reshape((n_states, n_branches, int(operator.max_conn_size))).astype(
                dtype
            )
        ).reshape((n_states, -1))
        branch_states = branch_states.reshape((-1, hilbert_size))

    x_out = branch_states.reshape((n_states, -1, hilbert_size))
    mels_out = branch_mels * jnp.asarray(coefficient, dtype=dtype)
    return x_out, mels_out


__all__ = ["apply_product_batch", "product_max_conn_size"]
