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


"""Scalar multiplication wrapper for computational operators."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator._algebra import is_known_real_scalar
from neuralqx.utils.struct import field

from ._utils import canonical_scalar
from ._utils import result_dtype
from .base import ComputationalWrappedOperator


class ScaledOperator(ComputationalWrappedOperator):
    """A computational operator multiplied by a scalar coefficient."""

    coefficient: Any = field(default=1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "coefficient", canonical_scalar(self.coefficient))

    @property
    def max_conn_size(self) -> int:
        return int(self.operator.max_conn_size)

    @property
    def dtype(self) -> jnp.dtype:
        return result_dtype(self.operator, self.coefficient)

    @property
    def is_hermitian(self) -> bool:
        return bool(
            self.operator.is_hermitian and is_known_real_scalar(self.coefficient)
        )

    @property
    def is_diagonal(self) -> bool:
        return bool(getattr(self.operator, "is_diagonal", False))

    @property
    def adjoint(self) -> ScaledOperator:
        return ScaledOperator(self.operator.adjoint, jnp.conj(self.coefficient))

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_primes, mels = self.operator._get_conn_padded_kernel(state)
        return x_primes, jnp.asarray(self.coefficient, dtype=self.dtype) * mels.astype(
            self.dtype
        )

    def _get_conn_padded_batch_kernel(
        self,
        states_2d: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x_primes, mels = self.operator._get_conn_padded_batch_kernel(states_2d)
        return x_primes, jnp.asarray(self.coefficient, dtype=self.dtype) * mels.astype(
            self.dtype
        )


__all__ = ["ScaledOperator"]
