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


"""Validation and dtype helpers for operator wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

from neuralqx.operator.abstract import AbstractOperator
from neuralqx.operator.computational import ComputationalOperator


def as_operator_tuple(
    operators: Iterable[ComputationalOperator],
) -> tuple[ComputationalOperator, ...]:
    """Canonicalise an iterable of computational operators to a tuple."""
    return tuple(operators)


def validate_operator_tuple(operators: tuple[ComputationalOperator, ...]) -> None:
    """Validate non-empty, same-Hilbert computational operator tuples."""
    if not operators:
        raise ValueError("At least one operator is required.")
    if not all(isinstance(operator, ComputationalOperator) for operator in operators):
        types = ", ".join(type(operator).__name__ for operator in operators)
        raise TypeError(f"Expected ComputationalOperator instances; got {types}.")
    hilbert = operators[0].hilbert
    if any(operator.hilbert != hilbert for operator in operators[1:]):
        raise ValueError("All wrapped operators must act on the same Hilbert space.")


def validate_computational_operator(operator: ComputationalOperator) -> None:
    """Validate one wrapped computational operator."""
    if not isinstance(operator, ComputationalOperator):
        raise TypeError(
            "Expected a ComputationalOperator; " f"got {type(operator).__name__}."
        )


def validate_wrapped_operator(operator: AbstractOperator) -> None:
    """Validate one wrapped operator."""
    if not isinstance(operator, AbstractOperator):
        raise TypeError(
            "Expected an AbstractOperator; " f"got {type(operator).__name__}."
        )


def canonical_coefficients(coefficients: Any, n_operators: int) -> jnp.ndarray:
    """Return one scalar coefficient per operator."""
    coeffs = jnp.asarray(coefficients)
    if coeffs.ndim == 0:
        return jnp.broadcast_to(coeffs, (int(n_operators),))
    if coeffs.shape != (int(n_operators),):
        raise ValueError(
            f"Expected coefficients with shape {(int(n_operators),)}, "
            f"got {coeffs.shape}."
        )
    return coeffs


def canonical_scalar(value: Any) -> jnp.ndarray:
    """Return a rank-0 JAX scalar."""
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"Expected a scalar coefficient, got shape {scalar.shape}.")
    return scalar


def result_dtype(*values: Any) -> jnp.dtype:
    """Infer a JAX result dtype from operators, arrays, scalars, and dtypes."""
    dtype_inputs = [getattr(value, "dtype", value) for value in values]
    return jnp.dtype(jnp.result_type(*dtype_inputs))


__all__ = [
    "as_operator_tuple",
    "canonical_coefficients",
    "canonical_scalar",
    "result_dtype",
    "validate_computational_operator",
    "validate_operator_tuple",
    "validate_wrapped_operator",
]
