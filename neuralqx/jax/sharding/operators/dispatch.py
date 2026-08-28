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


"""Operator-axis dispatch for computational local-estimator terms."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from neuralqx.jax.distributed.coordinator import plan_distributed_execution
from neuralqx.jax.shard_map import shard_map
from neuralqx.jax.sharding.samples import place_sample_batch
from neuralqx.jax.streaming import apply_batched
from neuralqx.jax.streaming import stream_flattened_kernel
from neuralqx.operator import ComputationalOperator
from neuralqx.operator._policy import resolve_streaming_policy
from neuralqx.utils.typing import DistributedPhase
from neuralqx.utils.typing import DistributedPlan
from neuralqx.utils.typing import DistributedWorkload


def sharded_operator_term_values(
    apply_variables: Any,
    variables: Any,
    terms: tuple[Any, ...],
    states_2d: jax.Array,
    *,
    chunk_size: int | None,
    enabled: bool | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> jax.Array | None:
    """Compute local values for computational terms on the operator mesh.

    When the operator mesh is active, this path moves computational term
    dispatch and local-estimator evaluation into a single ``shard_map``. Each
    operator-axis shard receives one or more scheduled term indices, computes
    only those terms' connected states, evaluates the model on those connected
    states, and returns that shard's local-value contribution.

    Terms with different ``max_conn_size`` values are deliberately allowed in
    one dispatch. Each branch reduces its connected states to a per-sample local
    value before returning, so branch output shapes agree without padding every
    operator to the largest connection count. The only padding added here is
    zero-work term-slot padding that makes the term axis evenly divisible by
    the operator mesh. Python still traces every branch on the host because JAX
    devices cannot execute Python, but runtime connected-component and
    model-evaluation work is distributed by the operator mesh instead of being
    precomputed term-by-term before placement.

    ``None`` is returned when the fast path is not applicable, allowing callers
    to fall back to a generic host-dispatched implementation.
    """
    if enabled is False:
        return None
    if not terms:
        raise ValueError("Operator-term buckets must contain at least one term.")
    if not all(isinstance(term.operator, ComputationalOperator) for term in terms):
        return None

    states = jnp.asarray(states_2d)
    plan = plan_distributed_execution(
        DistributedPhase.OPERATOR_LOCAL_VALUES,
        DistributedWorkload.operator_local_values(
            operator_costs=tuple(int(term.operator.max_conn_size) for term in terms),
            sample_count=int(states.shape[0]),
            enabled=enabled,
        ),
    )
    if plan is None:
        return None

    term_indices = _place_term_indices(plan)
    common_dtype = _operator_term_bucket_dtype(terms)
    stream, resolved_chunk_size = resolve_streaming_policy(
        use_streaming=operator_streaming,
        chunk_size=operator_streaming_chunk_size,
    )

    branches = tuple(
        _operator_term_branch(
            term,
            apply_variables=apply_variables,
            matrix_element_dtype=common_dtype,
            stream=stream,
            chunk_size=resolved_chunk_size,
            model_chunk_size=chunk_size,
        )
        for term in terms
    )
    branches = (
        *branches,
        _zero_operator_term_branch(),
    )
    logpsi_states = place_sample_batch(states) if plan.uses_sample_axis else states
    if any(not bool(getattr(term.operator, "is_diagonal", False)) for term in terms):
        logpsi = apply_batched(
            apply_variables,
            variables,
            logpsi_states,
            chunk_size=chunk_size,
        )
    else:
        logpsi = jnp.zeros((states.shape[0],), dtype=common_dtype)
    logpsi = _place_logpsi(logpsi, plan)
    states = _place_states(states, plan)

    def local_term_values(
        local_term_indices, local_states, local_logpsi, local_variables
    ):
        def one_term(index):
            return jax.lax.switch(
                index,
                branches,
                local_states,
                local_logpsi,
                local_variables,
            )

        return jax.lax.map(one_term, local_term_indices)

    state_spec, logpsi_spec, out_spec = _dispatch_specs(plan)
    mapped = shard_map(
        local_term_values,
        mesh=plan.mesh,
        in_specs=(
            PartitionSpec(plan.operator_axis),
            state_spec,
            logpsi_spec,
            PartitionSpec(),
        ),
        out_specs=out_spec,
    )
    term_values = mapped(term_indices, states, logpsi, variables)
    return jnp.sum(term_values, axis=0)


def _scheduled_operator_term_indices(
    terms: tuple[Any, ...],
    partitions: int,
) -> jax.Array:
    """Compatibility wrapper for the operator-term scheduler."""
    from neuralqx.jax.distributed.scheduling import schedule_balanced_slots

    costs = tuple(int(term.operator.max_conn_size) for term in terms)
    return schedule_balanced_slots(costs, partitions)


def _place_term_indices(plan: DistributedPlan) -> jax.Array:
    sharding = NamedSharding(plan.mesh, PartitionSpec(plan.operator_axis))
    return jax.device_put(plan.term_indices, sharding)


def _place_states(states: jax.Array, plan: DistributedPlan) -> jax.Array:
    if not plan.uses_sample_axis:
        return states
    assert plan.sample_axis is not None
    sharding = NamedSharding(plan.mesh, PartitionSpec(plan.sample_axis, None))
    return jax.device_put(states, sharding)


def _place_logpsi(logpsi: jax.Array, plan: DistributedPlan) -> jax.Array:
    if not plan.uses_sample_axis:
        return logpsi
    assert plan.sample_axis is not None
    sharding = NamedSharding(plan.mesh, PartitionSpec(plan.sample_axis))
    return jax.device_put(logpsi, sharding)


def _dispatch_specs(
    plan: DistributedPlan,
) -> tuple[PartitionSpec, PartitionSpec, PartitionSpec]:
    if not plan.uses_sample_axis:
        return (
            PartitionSpec(None, None),
            PartitionSpec(None),
            PartitionSpec(plan.operator_axis, None),
        )
    assert plan.sample_axis is not None
    return (
        PartitionSpec(plan.sample_axis, None),
        PartitionSpec(plan.sample_axis),
        PartitionSpec(plan.operator_axis, plan.sample_axis),
    )


def _operator_term_branch(
    term: Any,
    *,
    apply_variables: Any,
    matrix_element_dtype: jnp.dtype,
    stream: bool,
    chunk_size: int | None,
    model_chunk_size: int | None,
):
    """Build one same-shape operator branch for sharded term dispatch."""
    operator = term.operator
    coefficient = term.coefficient
    max_conn_size = int(operator.max_conn_size)

    def branch(states_2d, logpsi, variables):
        if stream:
            x_primes, mels = stream_flattened_kernel(
                lambda chunk: operator._get_conn_padded_batch_kernel(chunk),
                states_2d,
                chunk_size=chunk_size,
                max_conn_size=max_conn_size,
                matrix_element_dtype=operator.dtype,
            )
        else:
            x_primes, mels = operator._get_conn_padded_batch_kernel(states_2d)
        mels = jnp.asarray(mels, dtype=matrix_element_dtype) * jnp.asarray(
            coefficient, dtype=matrix_element_dtype
        )
        if bool(getattr(operator, "is_diagonal", False)):
            return jnp.sum(mels, axis=-1)
        connected = x_primes.reshape((-1, states_2d.shape[-1]))
        logpsi_connected = apply_batched(
            apply_variables,
            variables,
            connected,
            chunk_size=model_chunk_size,
        ).reshape(mels.shape)
        ratios = jnp.exp(logpsi_connected - logpsi[:, None])
        contrib_dtype = jnp.result_type(mels, ratios)
        contrib = jnp.where(
            mels != 0,
            mels.astype(contrib_dtype) * ratios.astype(contrib_dtype),
            jnp.zeros_like(ratios, dtype=contrib_dtype),
        )
        return jnp.sum(contrib, axis=-1)

    return branch


def _zero_operator_term_branch():
    """Return a padding branch with zero local-value contribution."""

    def branch(states_2d, logpsi, variables):
        del states_2d, variables
        return jnp.zeros_like(logpsi)

    return branch


def _operator_term_bucket_dtype(terms: tuple[Any, ...]) -> jnp.dtype:
    dtype = jnp.result_type(terms[0].operator.dtype, terms[0].coefficient)
    for term in terms[1:]:
        dtype = jnp.result_type(dtype, term.operator.dtype, term.coefficient)
    return jnp.dtype(dtype)


__all__ = ["sharded_operator_term_values"]
