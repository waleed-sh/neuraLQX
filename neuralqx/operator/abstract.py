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


"""Abstract operator interfaces."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from neuralqx.hilbert import AbstractHilbertSpace
from neuralqx.utils.errors import NeuralqxNotImplementedError
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.struct import field


class AbstractOperator(Struct, metaclass=StructABCMeta):
    r"""An operator tied to a neuraLQX Hilbert space.

    Operators are immutable structs that can participate in JAX transformations.
    Subclasses specify the matrix-element dtype and whether Hermiticity is known.
    The `adjoint` property represents the conjugate transpose, $A^\dagger$.
    Concrete operators must implement it when they are used in products or
    quadratic constraints.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space on which the operator acts.
    """

    hilbert: AbstractHilbertSpace = field(static=True)
    """Hilbert space defining the operator basis."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        if not isinstance(self.hilbert, AbstractHilbertSpace):
            raise NeuralqxTypeError(
                "Operators must act on a neuraLQX Hilbert space, "
                f"got {type(self.hilbert).__name__}."
            )

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype | Any:
        """Matrix-element dtype produced by this operator."""

    @property
    @abc.abstractmethod
    def is_hermitian(self) -> bool:
        """Whether the operator is Hermitian."""

    @property
    def adjoint(self) -> AbstractOperator:
        """Return the Hermitian adjoint operator."""
        raise NeuralqxNotImplementedError(
            f"{type(self).__name__} does not implement the adjoint property."
        )


__all__ = ["AbstractOperator"]
