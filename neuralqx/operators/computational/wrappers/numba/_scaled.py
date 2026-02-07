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

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator
from neuralqx.utils.numbers import promote_constant_for_op_dtype
from neuralqx.utils.numbers import hermitian_flag_for_sum_with_scalar


class Scaled(ComputationalOperator):
    r"""
    Wrapper for scalar multiplication of a ComputationalOperator.

    Given an operator ``O`` and a scalar ``α`` (float/complex, Python/jnp/np),
    this represents the operator ``α·O``:

        (α·O) |σ⟩  ->  same connected states as O, matrix elements scaled by α.

    Notes
    -----
    - Scales all matrix elements (diagonal and off-diagonal).
    - Dtype is promoted with ``promote_constant_for_op_dtype(op.dtype, α)`` and
      then combined via ``jnp.result_type``.
    - Hermiticity: marked Hermitian iff the base operator is Hermitian and α is (numerically) real.
    """

    def __init__(self, op: ComputationalOperator, alpha):
        super().__init__(op.hilbert)
        self._op = op

        # cast alpha to the promoted dtype so results are stable under jit and dtype checks
        target_dtype, alpha_cast = promote_constant_for_op_dtype(op.dtype, alpha)

        # store as JAX scalar with the right dtype
        self._alpha = jnp.asarray(alpha_cast, dtype=target_dtype)
        self._dtype = jnp.result_type(op.dtype, self._alpha.dtype)

        # Hermitian iff base is Hermitian and alpha is real
        self._is_hermitian = bool(
            hermitian_flag_for_sum_with_scalar(op.is_hermitian, self._alpha)
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    def _get_conn_padded_kernel(self, x):
        xp, mels = self._op.get_conn_padded(x)
        # ensure correct dtype then scale all mels
        alpha = jnp.asarray(self._alpha, dtype=self._dtype)
        return xp, (alpha * mels.astype(self._dtype))
