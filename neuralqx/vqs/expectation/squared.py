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

"""Memory-bounded operator-parallel gradients of sums of squared actions.

This is a separate execution path. The existing serial and sample-parallel
specialized squared-gradient kernels remain the fallback.
"""

from functools import partial

import jax.numpy as jnp

import jax
from jax.sharding import PartitionSpec as P
from neuralqx.jax.shard_map import shard_map
from neuralqx.jax.shard_map import varying_over
from neuralqx.jax.sharding.operators.squared import amplitude_dtype
from neuralqx.jax.sharding.operators.squared import execution_arguments
from neuralqx.jax.sharding.operators.squared import parent_branch
from neuralqx.jax.sharding.operators.squared import squared_term_plan
from neuralqx.jax.streaming import apply_batched
from neuralqx.profile._instrumentation import instrument
from neuralqx.vqs.estimators.operators import squared_operator_parents
from neuralqx.vqs.estimators.statistics import statistics


@instrument(
    "gradient.operator_squares", detail="detailed", category="neuralqx.gradient"
)
def operator_squared_expect_and_grad(vstate, operator, samples, sample_shape, options):
    """Dispatch complete squared parents across operators, or return None.

    Independent terms are summed on the operator axis. Gradients are averaged
    only over distinct samples. The tensor axis belongs to the compiler and
    never participates in Monte Carlo normalization.
    """
    parents = squared_operator_parents(operator)
    if parents is None:
        return None
    plan = squared_term_plan(parents, len(samples), enabled=options.operator_sharding)
    if plan is None:
        return None
    indices, settings = execution_arguments(
        plan,
        chunk_size=options.chunk_size,
        operator_streaming=options.operator_streaming,
        operator_streaming_chunk_size=options.operator_streaming_chunk_size,
    )
    values, gradient = _squared_values_and_grad(
        vstate.apply_variables_kernel,
        vstate.merge_variables_kernel,
        vstate.parameters,
        vstate.model_state,
        samples,
        parents,
        indices,
        vstate.machine_pow,
        **settings,
    )
    return statistics(values.reshape(sample_shape)), gradient


@partial(
    jax.jit,
    static_argnames=(
        "apply_variables",
        "merge_variables",
        "mesh",
        "operator_axis",
        "sample_axis",
        "chunk_size",
        "stream",
        "connection_chunk_size",
    ),
)
def _squared_values_and_grad(
    apply_variables,
    merge_variables,
    parameters,
    model_state,
    samples,
    parents,
    indices,
    machine_pow,
    *,
    mesh,
    operator_axis,
    sample_axis,
    chunk_size,
    stream,
    connection_chunk_size,
):
    def source_logs(params):
        return apply_batched(
            apply_variables,
            merge_variables(params, model_state),
            samples,
            chunk_size=chunk_size,
        )

    # One source forward/pullback, shared by every vertex. The targets below
    # differentiate source-feature reuse inside apply_connected independently.
    logs, source_pullback = jax.vjp(source_logs, parameters)
    dtype = amplitude_dtype(logs, parents)
    axes = (operator_axis,) if sample_axis is None else (operator_axis, sample_axis)

    def local(params, state, batch, source, operators, slots):
        # Establish local coordinates before differentiation so pcast's
        # transpose cannot duplicate our explicit work-axis reductions.
        params, state = varying_over((params, state), axes)
        batch, source = varying_over((batch, source), operator_axis)
        zero_values = jnp.zeros_like(source.real)
        zero_gradient = jax.tree.map(jnp.zeros_like, params)

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
            diagonal = bool(parent.is_diagonal)

            def branch(local_params):
                def target_loss(p):
                    variables = merge_variables(p, state)
                    values = jnp.abs(evaluate(batch, source, variables)) ** 2
                    return jnp.mean(values), values

                (_, values), gradient = jax.value_and_grad(target_loss, has_aux=True)(
                    local_params
                )
                # A diagonal branch bypasses the network altogether. It has
                # neither a target derivative nor a denominator derivative.
                denominator = zero_values if diagonal else values
                return values, denominator, gradient

            return branch

        zero = (zero_values, zero_values, zero_gradient)
        branches = (
            *(make_branch(op, i) for i, op in enumerate(operators)),
            lambda _: zero,
        )

        def step(carry, index):
            # Differentiate within the scan, not through the whole scan. Only
            # one vertex's activations and one gradient accumulator stay live.
            contribution = jax.lax.switch(index, branches, params)
            return jax.tree.map(jnp.add, carry, contribution), None

        (values, denominator, gradient), _ = jax.lax.scan(step, zero, slots)
        values, denominator, gradient = jax.lax.psum(
            (values, denominator, gradient), operator_axis
        )
        if sample_axis is not None:
            gradient = jax.lax.pmean(gradient, sample_axis)
        return values, denominator, gradient

    values, denominator, gradient = shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P(sample_axis), P(sample_axis), P(), P(operator_axis)),
        out_specs=(P(sample_axis), P(sample_axis), P()),
    )(parameters, model_state, samples, logs.astype(dtype), parents, indices)

    # Differentiate both the squared estimator and the normalized sampling
    # measure. In the ordinary non-diagonal case this is exactly
    # ((machine_pow - 2) * values - machine_pow * mean(values)) / n_samples.
    source_weight = (
        machine_pow * (values - jnp.mean(values)) - 2 * denominator
    ) / values.size
    source_gradient = source_pullback(source_weight.astype(logs.dtype))[0]
    gradient = jax.tree.map(
        lambda target, source: jnp.conj(target + source), gradient, source_gradient
    )
    return values, gradient
