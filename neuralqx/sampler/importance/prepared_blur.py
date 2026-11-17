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


"""Sparse blur evaluation that reuses eager operator preparation and local caches."""

from functools import partial

import jax.numpy as jnp

import jax
from neuralqx.jax.streaming import apply_batched
from neuralqx.vqs.estimators.connectivity import prepare_connections
from neuralqx.vqs.estimators.kernels import _prepare_local_connections


def _support(operator, states):
    targets, mels, owners = _prepare_local_connections(
        operator, states, operator_streaming=False, operator_streaming_chunk_size=None
    )
    if owners is None:
        owners = jnp.repeat(jnp.arange(len(states)), targets.shape[1])
        targets, mels = targets.reshape((-1, states.shape[-1])), mels.reshape(-1)
    plan = prepare_connections(operator.hilbert, states, targets, mels, owners)
    degree = (
        jnp.zeros(len(states), jnp.int32)
        .at[plan.owners]
        .add((plan.mels != 0).astype(jnp.int32))
    )
    return plan, degree


@jax.jit
def _draw(states, targets, ids, owners, active, degree, key, probability):
    if not len(targets):
        return states
    choice_key, move_key = jax.random.split(key)
    # Group active destinations by source, then draw one integer rank per
    # row. Padding is sorted last and cannot become an accepted destination.
    order = jnp.argsort(jnp.where(active, owners, len(states)), stable=True)
    starts = jnp.cumsum(degree) - degree
    rank = jax.random.randint(choice_key, degree.shape, 0, jnp.maximum(degree, 1))
    positions = jnp.minimum(starts + rank, len(owners) - 1)
    slots = ids[order[positions]]
    move = (degree > 0) & (jax.random.uniform(move_key, (len(states),)) < probability)
    return jnp.where(move[:, None], targets[slots], states)


@partial(jax.jit, static_argnames=("n",))
def _incoming(logs, degrees, ids, owners, active, exponent, probability, *, n):
    masses = (
        exponent * logs[ids]
        - jnp.log(jnp.maximum(degrees[ids], 1))
        + jnp.log(probability)
    )
    masses = jnp.where(active, masses, -jnp.inf)
    maximum = jax.ops.segment_max(masses, owners, n)
    safe = jnp.where(jnp.isfinite(maximum), maximum, 0)
    weights = jnp.exp(masses - safe[owners])
    totals = jax.ops.segment_sum(weights, owners, n)
    scores = jax.ops.segment_sum(
        weights * jnp.where(jnp.isfinite(logs[ids]), logs[ids], 0), owners, n
    )
    return safe + jnp.log(totals), scores / jnp.where(totals > 0, totals, 1)


def _local(spec, operators, apply, variables, samples, exponent, key, model_chunk_size):
    outputs, densities, scores = [], [], []
    for start in range(0, len(samples), spec.chunk_size):
        original = samples[start : start + spec.chunk_size]
        block_key = jax.random.fold_in(key, start)
        term_key, move_key = jax.random.split(block_key)
        chosen = jax.random.randint(term_key, (len(original),), 0, len(operators))
        result = original
        for i, (_, outgoing) in enumerate(operators):
            plan, degree = _support(outgoing, original)
            moved = _draw(
                original,
                plan.states,
                plan.target_ids,
                plan.owners,
                plan.mels != 0,
                degree,
                jax.random.fold_in(move_key, i),
                spec.probability,
            )
            result = jnp.where((chosen == i)[:, None], moved, result)
        logs = apply_batched(apply, variables, result, chunk_size=model_chunk_size).real
        dtype = jnp.result_type(logs, exponent)
        terms, derivatives = [], []
        for incoming, outgoing in operators:
            _, degree = _support(outgoing, result)
            self_log = exponent * logs + jnp.log(
                jnp.where(degree > 0, 1 - spec.probability, 1.0)
            )
            plan, _ = _support(incoming, result)
            if plan.n_unique:
                # Evaluate each distinct reverse source once. Padding uses a
                # valid state and never contributes to the density.
                degrees = []
                for begin in range(0, len(plan.states), spec.chunk_size):
                    _, local_degree = _support(
                        outgoing, plan.states[begin : begin + spec.chunk_size]
                    )
                    degrees.append(local_degree)
                reverse_degree = jnp.concatenate(degrees)
                neighbor_logs = apply_batched(
                    apply, variables, plan.states, chunk_size=model_chunk_size
                ).real
                incoming_log, incoming_score = _incoming(
                    neighbor_logs,
                    reverse_degree,
                    plan.target_ids,
                    plan.owners,
                    plan.mels != 0,
                    exponent,
                    spec.probability,
                    n=len(result),
                )
            else:
                incoming_log, incoming_score = (
                    jnp.full(len(result), -jnp.inf),
                    jnp.zeros(len(result)),
                )
            combined = jnp.logaddexp(self_log, incoming_log)
            safe = jnp.where(jnp.isfinite(combined), combined, 0)
            derivative = (
                jnp.exp(self_log - safe) * jnp.where(jnp.isfinite(logs), logs, 0)
                + jnp.exp(incoming_log - safe) * incoming_score
            )
            terms.append(combined)
            derivatives.append(derivative)
        stacked = jnp.stack(terms)
        total = jax.scipy.special.logsumexp(stacked, axis=0)
        derivative = jnp.sum(jnp.exp(stacked - total) * jnp.stack(derivatives), axis=0)
        outputs.append(result)
        densities.append((total - jnp.log(len(operators))).astype(dtype))
        scores.append(derivative.astype(dtype))
    return jnp.concatenate(outputs), jnp.concatenate(densities), jnp.concatenate(scores)


def prepared_blur(
    spec, operators, apply, variables, samples, exponent, key, model_chunk_size
):
    """Reuse compact operators independently on every addressable sample shard.

    Only sparse counts used for capacity buckets reach the host. Configurations,
    amplitudes and importance weights remain device arrays. Existing operator
    caches retain their configured budgets. No new persistent cache is created.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from neuralqx.vqs.estimators.sharded import _device_executor
    from neuralqx.vqs.estimators.sharded import _on_device
    from neuralqx.vqs.estimators.sharded import _parallel_preparation
    from neuralqx.vqs.estimators.sharded import local_model_layout

    layout = local_model_layout(samples, variables)
    if layout is None:
        return _local(
            spec, operators, apply, variables, samples, exponent, key, model_chunk_size
        )

    def local_value(value, device):
        if not isinstance(value, jax.Array):
            return value
        for shard in value.addressable_shards:
            if shard.device == device:
                return shard.data
        return jax.device_put(value, device)

    def local(shard):
        device = shard.device
        with jax.default_device(device):
            local_ops = tuple(
                (_on_device(a, device), _on_device(b, device)) for a, b in operators
            )
            local_vars = jax.tree.map(lambda x: local_value(x, device), variables)
            local_key = jax.random.fold_in(
                local_value(key, device), shard.index[0].start or 0
            )
            return _local(
                spec,
                local_ops,
                apply,
                local_vars,
                shard.data,
                local_value(exponent, device),
                local_key,
                model_chunk_size,
            )

    shards = samples.addressable_shards
    results = (
        list(_device_executor(len(shards)).map(local, shards))
        if _parallel_preparation(shards)
        else [local(s) for s in shards]
    )
    return tuple(
        jax.make_array_from_single_device_arrays(
            (len(samples), *group[0].shape[1:]),
            NamedSharding(layout.mesh, P(layout.spec[0])),
            list(group),
        )
        for group in zip(*results, strict=True)
    )
