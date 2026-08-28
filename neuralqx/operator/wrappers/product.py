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


"""Matrix-product wrapper for computational operators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator._algebra import is_known_real_scalar
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field

from ._compose import apply_product_batch
from ._compose import product_max_conn_size
from ._utils import as_operator_tuple
from ._utils import canonical_scalar
from ._utils import result_dtype
from ._utils import validate_operator_tuple


def _product_hilbert(wrapper: ProductOperator) -> DiscreteHilbertSpace:
    return wrapper.operators[0].hilbert


class ProductOperator(ComputationalOperator):
    """A JAX-native matrix product of computational operators.

    Operators are stored left-to-right. ``ProductOperator((A, B))`` represents
    ``A @ B`` and therefore applies ``B`` to input states first.
    """

    hilbert: DiscreteHilbertSpace = field(
        static=True,
        init=False,
        derived=_product_hilbert,
    )
    operators: tuple[ComputationalOperator, ...] | Iterable[ComputationalOperator] = (
        field(
            converter=as_operator_tuple,
        )
    )
    coefficient: Any = field(default=1.0)

    def __post_init__(self) -> None:
        validate_operator_tuple(self.operators)
        object.__setattr__(self, "coefficient", canonical_scalar(self.coefficient))
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        return product_max_conn_size(self.operators)

    @property
    def dtype(self) -> jnp.dtype:
        return result_dtype(*self.operators, self.coefficient)

    @property
    def is_hermitian(self) -> bool:
        if len(self.operators) == 1:
            return bool(
                self.operators[0].is_hermitian
                and is_known_real_scalar(self.coefficient)
            )
        return False

    @property
    def is_diagonal(self) -> bool:
        return all(
            bool(getattr(operator, "is_diagonal", False)) for operator in self.operators
        )

    @property
    def adjoint(self) -> ProductOperator:
        return ProductOperator(
            tuple(operator.adjoint for operator in reversed(self.operators)),
            coefficient=jnp.conj(self.coefficient),
        )

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_primes, mels = self._get_conn_padded_batch_kernel(state[None, :])
        return x_primes[0], mels[0]

    def _get_conn_padded_batch_kernel(
        self,
        states_2d: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return apply_product_batch(
            self.operators,
            states_2d,
            coefficient=self.coefficient,
            dtype=self.dtype,
        )


__all__ = ["ProductOperator"]
