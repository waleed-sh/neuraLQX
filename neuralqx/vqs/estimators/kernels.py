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


"""JAX kernels for variational local estimators."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.sharding.operators.dispatch import sharded_operator_term_values
from neuralqx.jax.streaming import apply_batched
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.vqs.estimators.operators import OperatorTerm

from .utils import flatten_samples
from .utils import host_operator_term_values
from .utils import is_computational_operator
from .utils import restore_sample_values


@partial(jax.jit, static_argnames=("apply_variables", "chunk_size"))
def local_values_from_connections(
    apply_variables: Any,
    variables: Any,
    samples_2d: jax.Array,
    x_primes: Any,
    mels: Any,
    *,
    chunk_size: int | None,
) -> jax.Array:
    """Compute local estimator values from padded connected states."""
    xp = jnp.asarray(x_primes)
    mel = jnp.asarray(mels)
    if xp.ndim != 3:
        xp = xp.reshape((samples_2d.shape[0], -1, samples_2d.shape[-1]))
        mel = mel.reshape(xp.shape[:-1])
    if xp.shape[0] != samples_2d.shape[0]:
        raise ValueError("Connected-state batch and sample batch disagree.")

    logpsi = apply_batched(
        apply_variables, variables, samples_2d, chunk_size=chunk_size
    )
    connected = xp.reshape((-1, samples_2d.shape[-1]))
    logpsi_connected = apply_batched(
        apply_variables,
        variables,
        connected,
        chunk_size=chunk_size,
    ).reshape(mel.shape)
    ratios = jnp.exp(logpsi_connected - logpsi[:, None])
    contrib_dtype = jnp.result_type(mel, ratios)
    contrib = jnp.where(
        mel != 0,
        mel.astype(contrib_dtype) * ratios.astype(contrib_dtype),
        jnp.zeros_like(ratios, dtype=contrib_dtype),
    )
    return jnp.sum(contrib, axis=-1)


@partial(jax.jit, static_argnames=("apply_variables", "chunk_size"))
def local_values_from_operator_terms(
    apply_variables: Any,
    variables: Any,
    samples_2d: jax.Array,
    x_primes_terms: Any,
    mels_terms: Any,
    *,
    chunk_size: int | None,
) -> jax.Array:
    """Compute summed local values from a stacked operator-term axis."""
    return local_values_from_operator_term_buckets(
        apply_variables,
        variables,
        samples_2d,
        (x_primes_terms,),
        (mels_terms,),
        chunk_size=chunk_size,
    )


@partial(jax.jit, static_argnames=("apply_variables", "chunk_size"))
def local_values_from_operator_term_buckets(
    apply_variables: Any,
    variables: Any,
    samples_2d: jax.Array,
    x_primes_buckets: tuple[Any, ...],
    mels_buckets: tuple[Any, ...],
    *,
    chunk_size: int | None,
) -> jax.Array:
    """Compute local values for same-size term buckets sharing base log-amplitudes."""
    logpsi = apply_batched(
        apply_variables,
        variables,
        samples_2d,
        chunk_size=chunk_size,
    )
    total = None
    for xp, mel in zip(x_primes_buckets, mels_buckets, strict=True):
        values = _local_values_from_operator_terms_with_logpsi(
            apply_variables,
            variables,
            samples_2d,
            logpsi,
            xp,
            mel,
            chunk_size=chunk_size,
        )
        total = values if total is None else total + values
    assert total is not None
    return total


def _local_values_from_operator_terms_with_logpsi(
    apply_variables: Any,
    variables: Any,
    samples_2d: jax.Array,
    logpsi: jax.Array,
    x_primes_terms: Any,
    mels_terms: Any,
    *,
    chunk_size: int | None,
) -> jax.Array:
    """Compute one same-size operator-term bucket with precomputed base values."""
    xp = jnp.asarray(x_primes_terms)
    mel = jnp.asarray(mels_terms)
    if xp.ndim != 4:
        xp = xp.reshape((xp.shape[0], samples_2d.shape[0], -1, samples_2d.shape[-1]))
        mel = mel.reshape(xp.shape[:-1])
    if xp.shape[1] != samples_2d.shape[0]:
        raise ValueError("Operator-term connections and sample batch disagree.")

    connected = xp.reshape((-1, samples_2d.shape[-1]))
    logpsi_connected = apply_batched(
        apply_variables,
        variables,
        connected,
        chunk_size=chunk_size,
    ).reshape(mel.shape)
    ratios = jnp.exp(logpsi_connected - logpsi[None, :, None])
    contrib_dtype = jnp.result_type(mel, ratios)
    contrib = jnp.where(
        mel != 0,
        mel.astype(contrib_dtype) * ratios.astype(contrib_dtype),
        jnp.zeros_like(ratios, dtype=contrib_dtype),
    )
    return jnp.sum(jnp.sum(contrib, axis=-1), axis=0)


def discrete_operator_terms_local_values(
    apply_variables: Any,
    variables: Any,
    samples: Any,
    terms: tuple[OperatorTerm, ...],
    *,
    chunk_size: int | None,
    operator_sharding: bool | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> jax.Array:
    """Local values for a flattened sequence of discrete operator terms."""
    flat, sample_shape = flatten_samples(samples, terms[0].operator.hilbert.size)
    device_total = None
    fallback_terms = []
    computational_terms = tuple(
        term for term in terms if is_computational_operator(term.operator)
    )
    generic_terms = tuple(
        term for term in terms if not is_computational_operator(term.operator)
    )
    if computational_terms:
        device_values = sharded_operator_term_values(
            apply_variables,
            variables,
            computational_terms,
            flat,
            chunk_size=chunk_size,
            enabled=operator_sharding,
            operator_streaming=operator_streaming,
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        )
        if device_values is None:
            fallback_terms.extend(computational_terms)
        else:
            device_total = device_values
    fallback_terms.extend(generic_terms)

    fallback_total = host_operator_term_values(
        apply_variables,
        variables,
        tuple(fallback_terms),
        flat,
        chunk_size=chunk_size,
        operator_sharding=operator_sharding,
        operator_streaming=operator_streaming,
        operator_streaming_chunk_size=operator_streaming_chunk_size,
    )
    values = fallback_total
    if device_total is not None:
        values = device_total if values is None else values + device_total
    assert values is not None
    return restore_sample_values(values, sample_shape)


def discrete_operator_local_values(
    apply_variables: Any,
    variables: Any,
    samples: Any,
    operator: DiscreteOperator,
    *,
    chunk_size: int | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> jax.Array:
    """Local values for a neuraLQX discrete algorithmic operator."""
    flat, sample_shape = flatten_samples(samples, operator.hilbert.size)
    x_primes, mels = operator.get_conn_padded(
        flat,
        use_streaming=operator_streaming,
        chunk_size=operator_streaming_chunk_size,
    )
    values = local_values_from_connections(
        apply_variables,
        variables,
        flat,
        x_primes,
        mels,
        chunk_size=chunk_size,
    )
    return restore_sample_values(values, sample_shape)


def squared_operator_local_values(
    apply_variables: Any,
    variables: Any,
    samples: Any,
    operator: Squared,
    *,
    chunk_size: int | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> jax.Array:
    """Local values for the semantic ``A.adjoint @ A`` wrapper."""
    parent = operator.operator
    if not isinstance(parent, DiscreteOperator):
        raise TypeError("Squared currently requires a discrete parent operator.")
    parent_values = discrete_operator_local_values(
        apply_variables,
        variables,
        samples,
        parent,
        chunk_size=chunk_size,
        operator_streaming=operator_streaming,
        operator_streaming_chunk_size=operator_streaming_chunk_size,
    )
    return jnp.abs(parent_values) ** 2


__all__ = [
    "discrete_operator_local_values",
    "discrete_operator_terms_local_values",
    "local_values_from_connections",
    "local_values_from_operator_term_buckets",
    "local_values_from_operator_terms",
    "squared_operator_local_values",
]
