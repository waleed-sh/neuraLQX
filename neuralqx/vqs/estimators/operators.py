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


"""Operator-term normalization for estimator execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from neuralqx.operator import AbstractOperator
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import SumOperator
from neuralqx.operator.wrappers import PenaltyCost


@dataclass(frozen=True, slots=True)
class OperatorTerm:
    """One independent operator term and its scalar coefficient."""

    operator: Any
    coefficient: Any = 1.0


def is_operator_sequence(operator: Any) -> bool:
    """Return whether ``operator`` is a user-facing sequence of operators."""
    return isinstance(operator, Sequence) and not isinstance(
        operator,
        (str, bytes, AbstractOperator),
    )


def as_operator_sequence(operator: Any) -> tuple[Any, ...]:
    """Normalize a user operator or operator sequence."""
    if is_operator_sequence(operator):
        operators = tuple(operator)
        if not operators:
            raise ValueError("Operator sequences must contain at least one operator.")
        return operators
    return (operator,)


def linear_operator_terms(operator: Any) -> tuple[OperatorTerm, ...]:
    """Flatten user operator sequences and ``SumOperator`` wrappers into terms."""
    terms: list[OperatorTerm] = []
    for item in as_operator_sequence(operator):
        terms.extend(_linear_operator_terms(item, coefficient=1.0))
    return tuple(terms)


def discrete_linear_terms(operator: Any) -> tuple[OperatorTerm, ...] | None:
    """Return flattened terms when every term has padded discrete connections."""
    terms = linear_operator_terms(operator)
    if all(isinstance(term.operator, DiscreteOperator) for term in terms):
        return terms
    return None


def sequence_is_hermitian(operator: Any) -> bool:
    """Return whether all terms in an operator sequence are Hermitian."""
    return all(
        bool(getattr(term.operator, "is_hermitian", False))
        for term in linear_operator_terms(operator)
    )


def _linear_operator_terms(
    operator: Any, *, coefficient: Any
) -> tuple[OperatorTerm, ...]:
    if isinstance(operator, SumOperator):
        coefficients = jnp.asarray(operator.coefficients)
        terms: list[OperatorTerm] = []
        for idx, child in enumerate(operator.operators):
            child_coeff = jnp.asarray(coefficient) * coefficients[idx]
            terms.extend(_linear_operator_terms(child, coefficient=child_coeff))
        return tuple(terms)
    if isinstance(operator, PenaltyCost):
        return (OperatorTerm(operator, coefficient),)
    return (OperatorTerm(operator, coefficient),)


__all__ = [
    "OperatorTerm",
    "as_operator_sequence",
    "discrete_linear_terms",
    "is_operator_sequence",
    "linear_operator_terms",
    "sequence_is_hermitian",
]
