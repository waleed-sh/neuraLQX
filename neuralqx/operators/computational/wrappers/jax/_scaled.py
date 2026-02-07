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

from typing import Tuple

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator import ComputationalJaxOperator
from neuralqx.utils.numbers import promote_constant_for_op_dtype
from neuralqx.utils.numbers import hermitian_flag_for_sum_with_scalar


@jax.jit
def _scale_apply_kernel(
    xp: jnp.ndarray,
    mels: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return xp, (alpha * mels)


@register_pytree_node_class
class ScaledJax(ComputationalJaxOperator):
    r"""
    JAX-native wrapper for scalar multiplication of a ComputationalJaxOperator.

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

    def __init__(self, op: ComputationalJaxOperator, alpha):
        super().__init__(op.hilbert)
        self._op = op

        # promote/attach scalar to a fixed JAX dtype (0-d array to avoid static args)
        target_dtype, alpha_cast = promote_constant_for_op_dtype(op.dtype, alpha)
        self._alpha = jnp.asarray(alpha_cast, dtype=target_dtype)

        # result dtype follows JAX promotion rules between op and alpha
        self._dtype = jnp.result_type(
            getattr(op, "dtype", jnp.float64), self._alpha.dtype
        )

        # Hermitian iff base is Hermitian and alpha is real
        self._is_hermitian = bool(
            hermitian_flag_for_sum_with_scalar(op.is_hermitian, self._alpha)
        )

        # cache connection size once (stable shapes helpful for chunking logic)
        D = int(self.hilbert.size)
        x_probe = jnp.zeros((1, 1, D), dtype=jnp.int64)
        xp_probe, _ = op.get_conn_padded(x_probe)
        self._C = int(xp_probe.shape[2])

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    @property
    def max_conn_size(self) -> int:
        # identical connection topology as the wrapped operator
        return int(self._C)

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int64)

        # delegate to base, then scale mels in a tiny jitted kernel
        xp, mels = self._op.get_conn_padded(x)

        # ensure stable dtype before/after scaling
        mels = mels.astype(self._dtype)
        alpha = self._alpha.astype(self._dtype)

        xp_out, mels_out = _scale_apply_kernel(xp, mels, alpha)
        return xp_out, mels_out

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        # put the wrapped operator and the scalar as leaves, keep small metadata in aux
        leaves = (self._op, self._alpha)
        aux = dict(
            hilbert=self.hilbert,
            dtype_str=str(self._dtype),
            is_hermitian=bool(self._is_hermitian),
            C=int(self._C),
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        op, alpha = leaves
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj._op = op
        obj._alpha = alpha
        obj._dtype = jnp.dtype(aux["dtype_str"])
        obj._is_hermitian = bool(aux["is_hermitian"])
        obj._C = int(aux["C"])
        return obj

    def __repr__(self):
        return (
            f"{type(self).__name__}(op={type(self._op).__name__}, "
            f"alpha={self._alpha}, hilbert={self.hilbert}, dtype={self.dtype})"
        )
