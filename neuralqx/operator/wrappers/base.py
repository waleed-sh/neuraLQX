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


"""Base classes for operator wrappers."""

from __future__ import annotations

import abc

from neuralqx.hilbert import AbstractHilbertSpace
from neuralqx.operator.abstract import AbstractOperator
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field

from ._utils import validate_computational_operator
from ._utils import validate_wrapped_operator


def _wrapped_hilbert(wrapper: WrappedOperator) -> AbstractHilbertSpace:
    return wrapper.operator.hilbert


def _computational_wrapped_hilbert(
    wrapper: ComputationalWrappedOperator,
) -> AbstractHilbertSpace:
    return wrapper.operator.hilbert


class WrappedOperator(AbstractOperator, metaclass=abc.ABCMeta):
    """Base class for lazy wrappers around a single operator."""

    hilbert: AbstractHilbertSpace = field(
        static=True,
        init=False,
        derived=_wrapped_hilbert,
    )
    operator: AbstractOperator

    def __post_init__(self) -> None:
        validate_wrapped_operator(self.operator)
        super().__post_init__()

    @property
    def parent(self) -> AbstractOperator:
        """Alias for the wrapped operator."""
        return self.operator


class ComputationalWrappedOperator(ComputationalOperator, metaclass=abc.ABCMeta):
    """Base class for computable wrappers around one computational operator."""

    hilbert: AbstractHilbertSpace = field(
        static=True,
        init=False,
        derived=_computational_wrapped_hilbert,
    )
    operator: ComputationalOperator

    def __post_init__(self) -> None:
        validate_computational_operator(self.operator)
        super().__post_init__()

    @property
    def parent(self) -> ComputationalOperator:
        """Alias for the wrapped operator."""
        return self.operator


__all__ = ["ComputationalWrappedOperator", "WrappedOperator"]
