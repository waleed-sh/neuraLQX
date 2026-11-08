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


"""Lazy sums of positive products for local quadratic constraints."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator

import jax.numpy as jnp

from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator.abstract import AbstractOperator
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field

from ._utils import as_operator_tuple
from ._utils import result_dtype
from ._utils import validate_operator_tuple
from .squared import Squared


def _sum_hilbert(wrapper: SumOfSquares) -> DiscreteHilbertSpace:
    """Read the common Hilbert space after validating the parent sequence."""
    validate_operator_tuple(wrapper.operators)
    return wrapper.operators[0].hilbert


class SumOfSquares(AbstractOperator):
    r"""The lazy positive operator $Q=\sum_v A_v^\dagger A_v$.

    Each parent is squared separately. There are no products between different
    parents, while a sum inside one parent retains its own cross terms. For a
    normalized state, the expectation is $\sum_v\|A_v\psi\|^2$.

    The expectation and gradient APIs evaluate the individual `Squared` markers
    without constructing product connectivity. Iteration exposes those markers
    in the supplied order. Like `Squared`, this wrapper does not provide matrix
    conversion or padded connectivity.

    To represent $\sum_v H_v H_v^\dagger$, pass the adjoints of the local
    constraints. Passing the local constraints themselves gives
    $\sum_v H_v^\dagger H_v$.

    Attributes:
        operators (tuple[ComputationalOperator, ...]): Nonempty sequence of linear parents
            on a common Hilbert space.
    """

    hilbert: DiscreteHilbertSpace = field(static=True, init=False, derived=_sum_hilbert)
    """Hilbert space shared by all local products."""

    operators: tuple[ComputationalOperator, ...] | Iterable[ComputationalOperator] = (
        field(
            converter=as_operator_tuple,
        )
    )
    """Linear parents, each contributing its own adjoint-times-parent product."""

    def __post_init__(self) -> None:
        """Validate that all parents are computational operators on the same space."""
        validate_operator_tuple(self.operators)
        super().__post_init__()

    def __iter__(self) -> Iterator[Squared]:
        """Yield one lazy positive product per parent for estimator execution."""
        return (Squared(operator) for operator in self.operators)

    def __len__(self) -> int:
        """Number of separately squared parents."""
        return len(self.operators)

    @property
    def dtype(self) -> jnp.dtype:
        """Common matrix-element dtype of the local products."""
        return result_dtype(*self.operators)

    @property
    def is_hermitian(self) -> bool:
        """Every adjoint-times-parent product is Hermitian."""
        return True

    @property
    def is_diagonal(self) -> bool:
        """Whether all parents are known to be diagonal."""
        return all(
            bool(getattr(operator, "is_diagonal", False)) for operator in self.operators
        )

    @property
    def adjoint(self) -> SumOfSquares:
        """Return this operator, since the sum is Hermitian."""
        return self


__all__ = ["SumOfSquares"]
