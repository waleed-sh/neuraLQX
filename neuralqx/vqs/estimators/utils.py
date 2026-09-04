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


"""Utility helpers for variational local estimators."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.sharding import place_operator_terms
from neuralqx.operator import ComputationalOperator
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.vqs.estimators.operators import OperatorTerm


def flatten_samples(
    samples: Any, hilbert_size: int
) -> tuple[jax.Array, tuple[int, ...]]:
    """Flatten leading sample axes while preserving the Hilbert axis."""
    arr = jnp.asarray(samples)
    if arr.ndim == 0 or arr.shape[-1] != int(hilbert_size):
        raise ValueError(
            f"Samples must have trailing Hilbert size {hilbert_size}; got shape {arr.shape}."
        )
    return arr.reshape((-1, int(hilbert_size))), arr.shape[:-1]


def restore_sample_values(values: Any, sample_shape: tuple[int, ...]) -> jax.Array:
    """Restore flattened per-sample values to the original sample shape."""
    arr = jnp.asarray(values)
    return arr.reshape(sample_shape)


def bucket_terms_by_conn_size(
    terms: tuple[OperatorTerm, ...],
) -> tuple[tuple[OperatorTerm, ...], ...]:
    """Group operator terms by padded connected-state count."""
    buckets: dict[int, list[OperatorTerm]] = {}
    for term in terms:
        buckets.setdefault(int(term.operator.max_conn_size), []).append(term)
    return tuple(tuple(bucket) for _size, bucket in sorted(buckets.items()))


def host_stacked_operator_term_connections(
    terms: tuple[OperatorTerm, ...],
    states_2d: jax.Array,
    *,
    operator_sharding: bool | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Generate term connections on the host and then place the term axis.

    This is the compatibility path for generic operator terms whose action is
    only exposed through the public ``get_conn_padded`` method. It preserves the
    historical execution model: each term is evaluated by Python on the host,
    the results are stacked, and the leading term axis is optionally placed on
    the operator mesh.
    """
    x_parts = []
    mel_parts = []
    for term in terms:
        x_primes, mels = term.operator.get_conn_padded(
            states_2d,
            use_streaming=operator_streaming,
            chunk_size=operator_streaming_chunk_size,
        )
        dtype = jnp.result_type(mels, term.coefficient)
        x_parts.append(jnp.asarray(x_primes))
        mel_parts.append(
            jnp.asarray(mels, dtype=dtype) * jnp.asarray(term.coefficient, dtype=dtype)
        )

    x_terms = jnp.stack(x_parts, axis=0)
    mel_terms = jnp.stack(mel_parts, axis=0)
    return place_operator_terms(
        x_terms,
        mel_terms,
        enabled=operator_sharding,
    )


def host_operator_term_values(
    apply_variables: Any,
    variables: Any,
    terms: tuple[OperatorTerm, ...],
    states_2d: jax.Array,
    *,
    chunk_size: int | None,
    operator_sharding: bool | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> jax.Array | None:
    """Compute fallback term values with the historical host-dispatch model."""
    if not terms:
        return None

    diagonal_total = None
    offdiagonal_terms = []
    for term in terms:
        if bool(getattr(term.operator, "is_diagonal", False)):
            _x_primes, mels = term.operator.get_conn_padded(
                states_2d,
                use_streaming=operator_streaming,
                chunk_size=operator_streaming_chunk_size,
            )
            dtype = jnp.result_type(mels, term.coefficient)
            values = jnp.sum(
                jnp.asarray(mels, dtype=dtype)
                * jnp.asarray(term.coefficient, dtype=dtype),
                axis=-1,
            )
            diagonal_total = (
                values if diagonal_total is None else diagonal_total + values
            )
        else:
            offdiagonal_terms.append(term)

    values = None
    x_buckets = []
    mel_buckets = []
    for bucket in bucket_terms_by_conn_size(tuple(offdiagonal_terms)):
        x_terms, mel_terms = host_stacked_operator_term_connections(
            bucket,
            states_2d,
            operator_sharding=operator_sharding,
            operator_streaming=operator_streaming,
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        )
        x_buckets.append(x_terms)
        mel_buckets.append(mel_terms)

    if x_buckets:
        from neuralqx.vqs.estimators.kernels import (
            local_values_from_operator_term_buckets,
        )

        values = local_values_from_operator_term_buckets(
            apply_variables,
            variables,
            states_2d,
            tuple(x_buckets),
            tuple(mel_buckets),
            chunk_size=chunk_size,
        )
    if diagonal_total is not None:
        values = diagonal_total if values is None else values + diagonal_total
    return values


def operator_for_hilbert(operator: Any) -> Any:
    """Return the Hilbert-carrying operator behind wrappers."""
    if isinstance(operator, PenaltyCost):
        return operator.cost_operator
    return operator


def validate_operator_hilbert(vstate: Any, operator: Any) -> None:
    """Validate that an operator acts on the variational state's Hilbert space."""
    from neuralqx.vqs.estimators.operators import linear_operator_terms

    for term in linear_operator_terms(operator):
        op = operator_for_hilbert(term.operator)
        if not hasattr(op, "hilbert"):
            continue
        if op.hilbert != vstate.hilbert:
            raise ValueError(
                "Operator and variational state act on different Hilbert spaces."
            )


def is_computational_operator(operator: Any) -> bool:
    """Return whether ``operator`` exposes padded connectivity kernels."""
    return isinstance(operator, ComputationalOperator)


__all__ = [
    "bucket_terms_by_conn_size",
    "flatten_samples",
    "host_operator_term_values",
    "host_stacked_operator_term_connections",
    "is_computational_operator",
    "operator_for_hilbert",
    "restore_sample_values",
    "validate_operator_hilbert",
]
