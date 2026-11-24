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

"""Device dispatch for independent squared operator actions.

The scheduling unit is one complete parent A in A.adjoint @ A. Its local
amplitude is summed before taking the absolute square, preserving interference
between connections inside that parent. Only the resulting squares are added
across the operator axis. Tensor axes remain compiler managed.
"""

from functools import partial
from types import SimpleNamespace

import jax.numpy as jnp

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from neuralqx.jax.distributed import plan_distributed_execution
from neuralqx.jax.shard_map import shard_map
from neuralqx.jax.shard_map import varying_over
from neuralqx.jax.sharding.arrays import place_global_array
from neuralqx.jax.streaming import apply_batched
from neuralqx.operator import ComputationalOperator
from neuralqx.operator._policy import resolve_streaming_policy
from neuralqx.profile._instrumentation import instrument
from neuralqx.utils.typing import DistributedPhase
from neuralqx.utils.typing import DistributedWorkload

from .dispatch import _operator_term_branch


def squared_term_plan(parents, sample_count, *, enabled):
    """Schedule whole computational parents, or retain the existing path.

    A single parent has no independent square to distribute. Generic discrete
    operators retain their existing eager connectivity implementation.
    """
    if len(parents) < 2 or not all(
        isinstance(op, ComputationalOperator) for op in parents
    ):
        return None
    return plan_distributed_execution(
        DistributedPhase.OPERATOR_LOCAL_VALUES,
        DistributedWorkload.operator_local_values(
            operator_costs=tuple(op.max_conn_size for op in parents),
            sample_count=sample_count,
            enabled=enabled,
        ),
    )


def execution_arguments(
    plan, *, chunk_size, operator_streaming, operator_streaming_chunk_size
):
    """Separate dynamic slot indices from hashable compilation settings."""
    indices = place_global_array(
        plan.term_indices, NamedSharding(plan.mesh, P(plan.operator_axis))
    )
    stream, connection_chunk_size = resolve_streaming_policy(
        use_streaming=operator_streaming,
        chunk_size=operator_streaming_chunk_size,
    )
    return indices, dict(
        mesh=plan.mesh,
        operator_axis=plan.operator_axis,
        sample_axis=plan.sample_axis if plan.uses_sample_axis else None,
        chunk_size=chunk_size,
        stream=stream,
        connection_chunk_size=connection_chunk_size,
    )


def parent_branch(
    parent, index, apply_variables, dtype, *, chunk_size, stream, connection_chunk_size
):
    """Reuse linear dispatch for one whole amplitude, before taking its square."""
    return _operator_term_branch(
        SimpleNamespace(operator=parent, coefficient=1.0),
        term_index=index,
        apply_variables=apply_variables,
        matrix_element_dtype=dtype,
        model_chunk_size=chunk_size,
        stream=stream,
        chunk_size=connection_chunk_size,
    )


def amplitude_dtype(logpsi, parents):
    """Use one dtype for every switch branch, including diagonal parents."""
    return jnp.result_type(logpsi.dtype, *(op.dtype for op in parents))


@instrument(
    "operators.sharded_squares", detail="detailed", category="neuralqx.operator"
)
def sharded_squared_values(
    apply_variables, variables, samples, parents, plan, **options
):
    """Return the per-sample sum of independent squared actions on a work mesh."""
    indices, settings = execution_arguments(plan, **options)
    return _squared_values(
        apply_variables, variables, samples, parents, indices, **settings
    )


@partial(
    jax.jit,
    static_argnames=(
        "apply_variables",
        "mesh",
        "operator_axis",
        "sample_axis",
        "chunk_size",
        "stream",
        "connection_chunk_size",
    ),
)
def _squared_values(
    apply_variables,
    variables,
    samples,
    parents,
    indices,
    *,
    mesh,
    operator_axis,
    sample_axis,
    chunk_size,
    stream,
    connection_chunk_size,
):
    # Each source amplitude is evaluated once, before replication to operator workers.
    logpsi = apply_batched(apply_variables, variables, samples, chunk_size=chunk_size)
    dtype = amplitude_dtype(logpsi, parents)
    logpsi = logpsi.astype(dtype)

    def local(local_variables, batch, logs, operators, slots):
        local_variables, batch, logs = varying_over(
            (local_variables, batch, logs), operator_axis
        )

        def make_branch(parent, index):
            evaluate = parent_branch(
                parent,
                index,
                apply_variables,
                dtype,
                chunk_size=chunk_size,
                stream=stream,
                connection_chunk_size=connection_chunk_size,
            )
            return lambda variables: jnp.abs(evaluate(batch, logs, variables)) ** 2

        zero = jnp.zeros_like(logs.real)
        branches = (
            *(make_branch(op, i) for i, op in enumerate(operators)),
            lambda _: zero,
        )

        def step(total, index):
            values = jax.lax.switch(index, branches, local_variables)
            return total + values, None

        total, _ = jax.lax.scan(step, zero, slots)
        return jax.lax.psum(total, operator_axis)

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(sample_axis), P(sample_axis), P(), P(operator_axis)),
        out_specs=P(sample_axis),
    )(variables, samples, logpsi, parents, indices)
