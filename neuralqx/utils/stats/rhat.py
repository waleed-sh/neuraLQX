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


"""Gelman-Rubin convergence diagnostics for chain-major Monte Carlo data."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def split_r_hat(matrix: jax.Array, variance: jax.Array) -> jax.Array:
    """Return the split-chain R-hat diagnostic for ``matrix``.

    ``matrix`` is interpreted as ``(n_chains, chain_length)``. The diagnostic is
    computed from the variance of the split-chain means and normalised by the
    total estimator variance. This is the numerically stable convention used for
    local-estimator streams: when the local estimator is almost deterministic,
    tiny within-chain roundoff must not turn into an unbounded diagnostic.
    """
    n_chains, chain_length = matrix.shape
    dtype = _real_dtype(matrix)
    if n_chains < 2 or chain_length < 2:
        return _nan(dtype)

    split_length = chain_length // 2
    even_length = 2 * split_length
    split = matrix[:, :even_length].reshape((2 * n_chains, split_length))
    split_mean_variance = chain_mean_variance(split)
    return r_hat_from_mean_variance(
        split_mean_variance,
        variance,
        chain_length=split_length,
        valid=True,
    )


def r_hat_from_mean_variance(
    mean_variance: jax.Array,
    variance: jax.Array,
    *,
    chain_length: int | jax.Array,
    valid: bool | jax.Array,
) -> jax.Array:
    """Combine chain-mean variance and total variance into an R-hat value."""
    dtype = jnp.result_type(mean_variance, variance, jnp.float32)
    n = jnp.asarray(chain_length, dtype=dtype)
    var = jnp.asarray(variance, dtype=dtype)
    mean_var = jnp.asarray(mean_variance, dtype=dtype)
    finite_variance = var > 0
    r_hat_squared = (n - 1.0) / n + mean_var / var
    r_hat = jnp.sqrt(jnp.maximum(r_hat_squared, 0.0))
    return jnp.where(
        jnp.asarray(valid) & (n > 1) & finite_variance,
        r_hat,
        _nan(dtype),
    )


def chain_mean_variance(
    matrix: jax.Array, *, valid: jax.Array | None = None
) -> jax.Array:
    """Return the population variance of chain means."""
    means = jnp.mean(matrix, axis=1)
    if valid is None:
        centered = means - jnp.mean(means)
        return jnp.mean(jnp.abs(centered) ** 2).real

    weights = valid.astype(_real_dtype(means))
    n_valid = jnp.sum(weights)
    mean = jnp.sum(weights * means) / jnp.maximum(n_valid, 1.0)
    centered = jnp.where(valid, means - mean, 0.0)
    return jnp.sum(jnp.abs(centered) ** 2).real / jnp.maximum(n_valid, 1.0)


def _real_dtype(value: jax.Array) -> jnp.dtype:
    return jnp.result_type(jnp.real(value), jnp.float32)


def _nan(dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(jnp.nan, dtype=dtype)


__all__ = ["chain_mean_variance", "r_hat_from_mean_variance", "split_r_hat"]
