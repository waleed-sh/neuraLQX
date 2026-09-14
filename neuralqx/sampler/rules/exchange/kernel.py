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


"""JAX kernels for exchange transition rules."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax


def exchange_batch(
    *,
    site_i: jax.Array,
    site_j: jax.Array,
    incident_clusters: jax.Array,
    incident_valid: jax.Array,
    probabilities: Any | None,
    key: jax.Array,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vectorized exchange proposals for a local chain batch."""
    site_i = jnp.asarray(site_i, dtype=jnp.int32)
    site_j = jnp.asarray(site_j, dtype=jnp.int32)
    incident_clusters = jnp.asarray(incident_clusters, dtype=jnp.int32)
    incident_valid = jnp.asarray(incident_valid, dtype=jnp.bool_)
    if probabilities is not None:
        probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    weights = cluster_weights(site_i, site_j, sigma, probabilities)
    weight_sums = jnp.sum(weights, axis=-1)
    clusters = sample_weighted_indices(key, weights, weight_sums)
    return jax.vmap(
        lambda row, row_weights, row_weight_sum, row_cluster: exchange_one(
            site_i=site_i,
            site_j=site_j,
            incident_clusters=incident_clusters,
            incident_valid=incident_valid,
            probabilities=probabilities,
            sigma=row,
            weights=row_weights,
            weight_sum=row_weight_sum,
            cluster=row_cluster,
        )
    )(sigma, weights, weight_sums, clusters)


def exchange_batch_spmd(
    *,
    site_i: jax.Array,
    site_j: jax.Array,
    incident_clusters: jax.Array,
    incident_valid: jax.Array,
    probabilities: Any | None,
    key: jax.Array,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vectorized exchange proposals using SPMD-safe metadata gathers."""
    site_i = jnp.asarray(site_i, dtype=jnp.int32)
    site_j = jnp.asarray(site_j, dtype=jnp.int32)
    incident_clusters = jnp.asarray(incident_clusters, dtype=jnp.int32)
    incident_valid = jnp.asarray(incident_valid, dtype=jnp.bool_)
    if probabilities is not None:
        probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    weights = cluster_weights_spmd(site_i, site_j, sigma, probabilities)
    weight_sums = jnp.sum(weights, axis=-1)
    clusters = sample_weighted_indices(key, weights, weight_sums)
    return jax.vmap(
        lambda row, row_weights, row_weight_sum, row_cluster: exchange_one_spmd(
            site_i=site_i,
            site_j=site_j,
            incident_clusters=incident_clusters,
            incident_valid=incident_valid,
            probabilities=probabilities,
            sigma=row,
            weights=row_weights,
            weight_sum=row_weight_sum,
            cluster=row_cluster,
        )
    )(sigma, weights, weight_sums, clusters)


def exchange_one(
    *,
    site_i: jax.Array,
    site_j: jax.Array,
    incident_clusters: jax.Array,
    incident_valid: jax.Array,
    probabilities: Any | None,
    sigma: jax.Array,
    weights: jax.Array,
    weight_sum: jax.Array,
    cluster: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exchange one valid cluster in a single chain state."""
    has_move = weight_sum > 0
    left = site_i[cluster]
    right = site_j[cluster]
    left_value = sigma[left]
    right_value = sigma[right]
    proposed = sigma.at[left].set(right_value)
    proposed = proposed.at[right].set(left_value)
    proposed = jnp.where(has_move, proposed, sigma)
    proposed_sum = updated_weight_sum(
        site_i=site_i,
        site_j=site_j,
        incident_clusters=incident_clusters,
        incident_valid=incident_valid,
        probabilities=probabilities,
        sigma=proposed,
        weights=weights,
        weight_sum=weight_sum,
        left=left,
        right=right,
    )
    log_correction = jnp.where(
        has_move & (proposed_sum > 0),
        jnp.log(weight_sum) - jnp.log(proposed_sum),
        jnp.zeros_like(weight_sum),
    )
    return proposed, log_correction


def exchange_one_spmd(
    *,
    site_i: jax.Array,
    site_j: jax.Array,
    incident_clusters: jax.Array,
    incident_valid: jax.Array,
    probabilities: Any | None,
    sigma: jax.Array,
    weights: jax.Array,
    weight_sum: jax.Array,
    cluster: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exchange one valid cluster using SPMD-safe metadata gathers."""
    has_move = weight_sum > 0
    left = static_take(site_i, cluster)
    right = static_take(site_j, cluster)
    left_value = gather_single_site(sigma, left)
    right_value = gather_single_site(sigma, right)
    proposed = sigma.at[left].set(right_value)
    proposed = proposed.at[right].set(left_value)
    proposed = jnp.where(has_move, proposed, sigma)
    proposed_sum = updated_weight_sum_spmd(
        site_i=site_i,
        site_j=site_j,
        incident_clusters=incident_clusters,
        incident_valid=incident_valid,
        probabilities=probabilities,
        sigma=proposed,
        weights=weights,
        weight_sum=weight_sum,
        left=left,
        right=right,
    )
    log_correction = jnp.where(
        has_move & (proposed_sum > 0),
        jnp.log(weight_sum) - jnp.log(proposed_sum),
        jnp.zeros_like(weight_sum),
    )
    return proposed, log_correction


def updated_weight_sum(
    *,
    site_i: jax.Array,
    site_j: jax.Array,
    incident_clusters: jax.Array,
    incident_valid: jax.Array,
    probabilities: Any | None,
    sigma: jax.Array,
    weights: jax.Array,
    weight_sum: jax.Array,
    left: jax.Array,
    right: jax.Array,
) -> jax.Array:
    """Update the proposal normalizer from incident-cluster deltas."""
    incident = jnp.concatenate(
        (
            incident_clusters[left],
            incident_clusters[right],
        ),
        axis=0,
    )
    valid = jnp.concatenate(
        (
            incident_valid[left],
            incident_valid[right],
        ),
        axis=0,
    )
    old_weights = jnp.where(valid, weights[incident], 0.0)
    changed_i = site_i[incident]
    changed_j = site_j[incident]
    changed_mask = values_differ(sigma[changed_i], sigma[changed_j])
    new_weights = changed_mask.astype(jnp.float32)
    if probabilities is not None:
        new_weights = new_weights * probabilities[incident]
    new_weights = jnp.where(valid, new_weights, 0.0)
    return weight_sum + jnp.sum(new_weights - old_weights)


def updated_weight_sum_spmd(
    *,
    site_i: jax.Array,
    site_j: jax.Array,
    incident_clusters: jax.Array,
    incident_valid: jax.Array,
    probabilities: Any | None,
    sigma: jax.Array,
    weights: jax.Array,
    weight_sum: jax.Array,
    left: jax.Array,
    right: jax.Array,
) -> jax.Array:
    """Update the proposal normalizer with SPMD-safe metadata gathers."""
    incident = jnp.concatenate(
        (
            static_take_row(incident_clusters, left),
            static_take_row(incident_clusters, right),
        ),
        axis=0,
    )
    valid = jnp.concatenate(
        (
            static_take_row(incident_valid.astype(jnp.int32), left).astype(jnp.bool_),
            static_take_row(incident_valid.astype(jnp.int32), right).astype(jnp.bool_),
        ),
        axis=0,
    )
    old_weights = jnp.where(valid, weights[incident], 0.0)
    changed_i = static_take(site_i, incident)
    changed_j = static_take(site_j, incident)
    changed_mask = values_differ(
        gather_sites(sigma, changed_i),
        gather_sites(sigma, changed_j),
    )
    new_weights = changed_mask.astype(jnp.float32)
    if probabilities is not None:
        new_weights = new_weights * static_take(probabilities, incident)
    new_weights = jnp.where(valid, new_weights, 0.0)
    return weight_sum + jnp.sum(new_weights - old_weights)


def cluster_weights(
    site_i: jax.Array,
    site_j: jax.Array,
    sigma: jax.Array,
    probabilities: Any | None,
) -> jax.Array:
    """Return proposal weights for currently hoppable exchange clusters."""
    weights = cluster_mask(site_i, site_j, sigma).astype(jnp.float32)
    if probabilities is not None:
        weights = weights * jnp.asarray(probabilities, dtype=weights.dtype)
    return weights


def cluster_weights_spmd(
    site_i: jax.Array,
    site_j: jax.Array,
    sigma: jax.Array,
    probabilities: Any | None,
) -> jax.Array:
    """Return SPMD-safe proposal weights for exchange clusters."""
    weights = cluster_mask_spmd(site_i, site_j, sigma).astype(jnp.float32)
    if probabilities is not None:
        weights = weights * jnp.asarray(probabilities, dtype=weights.dtype)
    return weights


def sample_weighted_indices(
    key: jax.Array,
    weights: jax.Array,
    weight_sums: jax.Array,
) -> jax.Array:
    """Sample one index per row from non-negative unnormalized weights."""
    has_move = weight_sums > 0
    safe_weights = jnp.where(has_move[:, None], weights, jnp.ones_like(weights))
    safe_sums = jnp.where(
        has_move,
        weight_sums,
        jnp.asarray(weights.shape[-1], dtype=weights.dtype),
    )
    thresholds = jax.random.uniform(key, weight_sums.shape, dtype=weights.dtype)
    thresholds = thresholds * safe_sums
    cdf = jnp.cumsum(safe_weights, axis=-1)
    clusters = jnp.sum(cdf <= thresholds[:, None], axis=-1, dtype=jnp.int32)
    return jnp.minimum(
        clusters,
        jnp.asarray(weights.shape[-1] - 1, dtype=clusters.dtype),
    )


def cluster_mask(site_i: jax.Array, site_j: jax.Array, sigma: jax.Array) -> jax.Array:
    """Return whether each cluster swaps unequal values."""
    return values_differ(sigma[..., site_i], sigma[..., site_j])


def cluster_mask_spmd(
    site_i: jax.Array,
    site_j: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Return SPMD-safe exchange mask for each cluster."""
    return values_differ(gather_sites(sigma, site_i), gather_sites(sigma, site_j))


def values_differ(left: jax.Array, right: jax.Array) -> jax.Array:
    """Compare values by dtype, using tolerance for floating spaces."""
    if jnp.issubdtype(left.dtype, jnp.integer) or jnp.issubdtype(left.dtype, jnp.bool_):
        return left != right
    return ~jnp.isclose(left, right)


def gather_sites(values: jax.Array, sites: jax.Array) -> jax.Array:
    """Variance-safe gather of one or more site values from state arrays."""
    one_hot = jax.nn.one_hot(sites, values.shape[-1], dtype=values.dtype)
    return jnp.sum(values[..., None, :] * one_hot, axis=-1, dtype=values.dtype)


def gather_single_site(values: jax.Array, site: jax.Array) -> jax.Array:
    """Variance-safe gather of one site value from one state vector."""
    one_hot = jax.nn.one_hot(site, values.shape[-1], dtype=values.dtype)
    return jnp.sum(values * one_hot, axis=-1, dtype=values.dtype)


def static_take(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Variance-safe gather from static rule metadata inside ``shard_map``."""
    one_hot = jax.nn.one_hot(indices, values.shape[0], dtype=values.dtype)
    return jnp.sum(one_hot * values, axis=-1, dtype=values.dtype)


def static_take_row(values: jax.Array, index: jax.Array) -> jax.Array:
    """Variance-safe row gather from static rule metadata inside ``shard_map``."""
    one_hot = jax.nn.one_hot(index, values.shape[0], dtype=values.dtype)
    return jnp.sum(one_hot[:, None] * values, axis=0, dtype=values.dtype)


__all__ = [
    "cluster_mask",
    "cluster_mask_spmd",
    "cluster_weights",
    "cluster_weights_spmd",
    "exchange_batch",
    "exchange_batch_spmd",
    "exchange_one",
    "exchange_one_spmd",
    "gather_single_site",
    "gather_sites",
    "sample_weighted_indices",
    "static_take",
    "static_take_row",
    "updated_weight_sum",
    "updated_weight_sum_spmd",
    "values_differ",
]
