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


"""Linear-combination wrapper for computational operators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator._algebra import is_known_real_scalar
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field

from ._utils import as_operator_tuple
from ._utils import canonical_coefficients
from ._utils import result_dtype
from ._utils import validate_operator_tuple


def _sum_hilbert(wrapper: SumOperator) -> DiscreteHilbertSpace:
    return wrapper.operators[0].hilbert


class SumOperator(ComputationalOperator):
    """A JAX-native linear combination of computational operators."""

    hilbert: DiscreteHilbertSpace = field(
        static=True,
        init=False,
        derived=_sum_hilbert,
    )
    operators: tuple[ComputationalOperator, ...] | Iterable[ComputationalOperator] = (
        field(
            converter=as_operator_tuple,
        )
    )
    coefficients: Any = field(default=1.0)

    def __post_init__(self) -> None:
        validate_operator_tuple(self.operators)
        object.__setattr__(
            self,
            "coefficients",
            canonical_coefficients(self.coefficients, len(self.operators)),
        )
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        return int(sum(int(operator.max_conn_size) for operator in self.operators))

    @property
    def dtype(self) -> jnp.dtype:
        return result_dtype(*self.operators, self.coefficients)

    @property
    def is_hermitian(self) -> bool:
        return bool(
            all(operator.is_hermitian for operator in self.operators)
            and all(
                is_known_real_scalar(coefficient) for coefficient in self.coefficients
            )
        )

    @property
    def is_diagonal(self) -> bool:
        return all(
            bool(getattr(operator, "is_diagonal", False)) for operator in self.operators
        )

    @property
    def adjoint(self) -> SumOperator:
        return SumOperator(
            tuple(operator.adjoint for operator in self.operators),
            coefficients=jnp.conj(self.coefficients),
        )

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_primes, mels = self._get_conn_padded_batch_kernel(state[None, :])
        return x_primes[0], mels[0]

    def _get_conn_padded_batch_kernel(
        self,
        states_2d: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x_parts = []
        mel_parts = []
        coefficients = self.coefficients.astype(self.dtype)
        for idx, operator in enumerate(self.operators):
            x_primes, mels = operator._get_conn_padded_batch_kernel(states_2d)
            x_parts.append(x_primes)
            mel_parts.append(mels.astype(self.dtype) * coefficients[idx])
        return jnp.concatenate(x_parts, axis=1), jnp.concatenate(mel_parts, axis=1)


__all__ = ["SumOperator"]
