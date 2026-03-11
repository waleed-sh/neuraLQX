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

"""
Abstract base class for Hilbert index/enumerator objects.

This module defines :class:`HilbertStateEnumerator`, the abstract base for
index/enumerator objects attached to Hilbert-space cores.

Concrete enumerator subclasses are attached to cores via ``core.index``.
Standalone dispatch functions then resolve on ``(index_class, core_class)``:

    :func:`~neuralqx.hilbert.utils.index.states_to_numbers`
    :func:`~neuralqx.hilbert.utils.index.numbers_to_states`

to dispatch on the concrete type via ``plum``.  Concrete overloads are
registered in:

* :mod:`neuralqx.hilbert.u1.index`  — for U(1) spaces
"""

import abc

from typing import Any
from typing import Dict
from typing import Generic
from typing import TypeVar

CoreT = TypeVar("CoreT")
StateBatchLike = Any
NumberBatchLike = Any
ExplanationMap = Dict[str, Any]


class HilbertStateEnumerator(abc.ABC, Generic[CoreT]):
    """
    Abstract base for Hilbert index/enumerator objects.

    A state enumerator is an index object that assigns a canonical integer
    label to every basis state and can recover the state from its label. This
    forms the basis for NetKet-compatible indexing and for the plum-based
    dispatch of :func:`states_to_numbers` / :func:`numbers_to_states`.

    Concrete subclasses
    -------------------
    * :class:`~neuralqx.hilbert.u1.index.enumerator.U1UnconstrainedStateEnumerator`
    * :class:`~neuralqx.hilbert.u1.index.enumerator.U1ConstrainedStateEnumerator`

    These classes implement ranking/unranking, metadata capabilities, and
    explainability hooks, and are bound to core instances through
    ``core.index``.
    """

    @property
    @abc.abstractmethod
    def scheme_name(self) -> str:
        """
        Identifier of the enumeration scheme.

        Examples:

        - ``"u1-unconstrained-direct-product"``
        - ``"u1-constrained-reduced-free-c-order"``
        """

    @property
    @abc.abstractmethod
    def ordering_contract(self) -> str:
        """
        Stable textual contract for the rank/unrank ordering.

        This should precisely describe the significance convention and any
        reduced parameterisation details used by the enumerator.
        """

    @property
    @abc.abstractmethod
    def requires_full_precompute(self) -> bool:
        """Whether rank/unrank requires a full precomputed index table."""

    @abc.abstractmethod
    def supports_lazy_mode(self, core: CoreT) -> bool:
        """Whether this enumerator can rank/unrank for ``core`` in lazy mode."""

    @abc.abstractmethod
    def states_to_numbers(
        self,
        core: CoreT,
        states: StateBatchLike,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> NumberBatchLike:
        """
        Convert basis states to sequential integer labels.

        :param core: Hilbert-space core owning this enumerator.
        :param states: State array of shape ``(N,)`` or ``(B, N)``.
        :param validate: If True, perform extra validity checks.
        :param kwargs: Additional keyword arguments forwarded to the concrete implementation (e.g. ``backend``,
           ``return_dtype`` for U(1) spaces).

        :return: Integer label(s) with shape matching the batch dimensions of ``states``.
        """

    @abc.abstractmethod
    def numbers_to_states(
        self,
        core: CoreT,
        numbers: NumberBatchLike,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> StateBatchLike:
        """
        Convert sequential integer labels back to basis states.

        :param core: Hilbert-space core owning this enumerator.
        :param numbers: Integer scalar or array of integers.
        :param validate: If True, check that numbers are within range.
        :param kwargs: Additional keyword arguments forwarded to the concrete implementation.

        :return: State array of shape ``(N,)`` or ``(B, N)``.
        """

    @abc.abstractmethod
    def explain_state_to_number(
        self,
        core: CoreT,
        state: StateBatchLike,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> ExplanationMap:
        """
        Explain how ``state`` is mapped to its integer label.

        Implementations should return a serialisable dictionary containing
        scheme-specific intermediate values (digits, sector indices, offsets,
        mixed-radix contributions, etc.).
        """

    @abc.abstractmethod
    def explain_number_to_state(
        self,
        core: CoreT,
        number: NumberBatchLike,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> ExplanationMap:
        """
        Explain how ``number`` is decoded into a basis state.

        Implementations should return a serialisable dictionary containing
        intermediate decoding information.
        """
