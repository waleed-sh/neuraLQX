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


"""Online Monte Carlo statistics based on parallel Welford updates."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.utils.struct import Struct

from .core import Stats
from .rhat import r_hat_from_mean_variance
from .shape import as_chain_matrix


class OnlineStats(Struct):
    """Streaming scalar statistics for chain-major Monte Carlo data.

    The accumulator keeps per-chain counts, means, and M2 sums. Updates merge a
    full batch per chain using the parallel Welford formula, avoiding repeated
    passes over historical data. The resulting ``R̂`` is the online chain
    diagnostic from accumulated chain means and within-chain variances. Exact
    split ``R̂`` is available in the one-shot :func:`statistics` path where raw
    chain histories are still present.
    """

    chain_count: Any
    chain_mean: Any
    chain_m2: Any
    n_samples: Any

    @classmethod
    def empty(cls, n_chains: int, dtype: Any = jnp.float64) -> OnlineStats:
        """Create an empty accumulator for ``n_chains`` independent chains."""
        n_chains = int(n_chains)
        if n_chains <= 0:
            raise ValueError("n_chains must be positive.")
        return cls(
            chain_count=jnp.zeros((n_chains,), dtype=jnp.float64),
            chain_mean=jnp.zeros((n_chains,), dtype=dtype),
            chain_m2=jnp.zeros((n_chains,), dtype=jnp.float64),
            n_samples=jnp.asarray(0, dtype=jnp.int64),
        )

    @classmethod
    def from_data(cls, values: Any, *, chain_axis: int | None = 0) -> OnlineStats:
        """Create an accumulator initialized from ``values``."""
        matrix = as_chain_matrix(values, chain_axis=chain_axis)
        return cls.empty(matrix.shape[0], matrix.dtype).update(matrix, chain_axis=0)

    @property
    def n_chains(self) -> int:
        """Number of independent chains."""
        return int(self.chain_count.shape[0])

    def update(self, values: Any, *, chain_axis: int | None = 0) -> OnlineStats:
        """Return a new accumulator with ``values`` merged in."""
        matrix = as_chain_matrix(values, chain_axis=chain_axis)
        if matrix.shape[0] != self.n_chains:
            raise ValueError(
                f"Number of chains changed: expected {self.n_chains}, "
                f"got {matrix.shape[0]}."
            )
        return _online_update(self, matrix)

    def to_stats(self) -> Stats:
        """Convert accumulated state to a scalar ``Stats`` object."""
        return _online_to_stats(self)

    def __repr__(self) -> str:
        return repr(self.to_stats())


@jax.jit
def _online_update(accumulator: OnlineStats, matrix: jax.Array) -> OnlineStats:
    n_batch = matrix.shape[1]
    batch_count = jnp.full_like(
        accumulator.chain_count,
        n_batch,
        dtype=accumulator.chain_count.dtype,
    )
    batch_mean = jnp.mean(matrix, axis=1)
    batch_m2 = jnp.sum(jnp.abs(matrix - batch_mean[:, None]) ** 2, axis=1).real

    old_count = accumulator.chain_count
    new_count = old_count + batch_count
    safe_count = jnp.maximum(new_count, 1.0)
    delta = batch_mean - accumulator.chain_mean
    chain_mean = accumulator.chain_mean + delta * (batch_count / safe_count)
    chain_m2 = (
        accumulator.chain_m2
        + batch_m2
        + jnp.abs(delta) ** 2 * (old_count * batch_count / safe_count)
    )
    return accumulator.replace(
        chain_count=new_count,
        chain_mean=chain_mean.astype(accumulator.chain_mean.dtype),
        chain_m2=chain_m2,
        n_samples=accumulator.n_samples + matrix.size,
    )


@jax.jit
def _online_to_stats(accumulator: OnlineStats) -> Stats:
    total = jnp.sum(accumulator.chain_count)
    safe_total = jnp.maximum(total, 1.0)
    mean = jnp.sum(accumulator.chain_count * accumulator.chain_mean) / safe_total
    global_m2 = jnp.sum(accumulator.chain_m2) + jnp.sum(
        accumulator.chain_count * jnp.abs(accumulator.chain_mean - mean) ** 2
    )
    variance = global_m2 / safe_total
    n_valid = jnp.sum(accumulator.chain_count > 0)

    single_chain_error = jnp.sqrt(variance / safe_total)
    chain_mean_var = jnp.sum(
        jnp.where(
            accumulator.chain_count > 0,
            jnp.abs(accumulator.chain_mean - mean) ** 2,
            0.0,
        )
    ) / jnp.maximum(n_valid, 1)
    chain_error = jnp.sqrt(chain_mean_var / jnp.maximum(n_valid, 1))
    error = jnp.where(n_valid > 1, chain_error, single_chain_error)

    n_per_chain = safe_total / jnp.maximum(n_valid, 1)
    r_hat = r_hat_from_mean_variance(
        chain_mean_var,
        variance,
        chain_length=n_per_chain,
        valid=n_valid > 1,
    )

    mean_nan = jnp.asarray(jnp.nan, dtype=mean.dtype)
    real_nan = jnp.asarray(jnp.nan, dtype=variance.dtype)
    return Stats(
        mean=jnp.where(total > 0, mean, mean_nan),
        variance=jnp.where(total > 0, variance, real_nan),
        error_of_mean=jnp.where(total > 0, error, real_nan),
        n_samples=accumulator.n_samples,
        r_hat=r_hat,
    )


def online_statistics(
    values: Any,
    accumulator: OnlineStats | None = None,
    *,
    chain_axis: int | None = 0,
) -> OnlineStats:
    """Accumulate ``values`` into an ``OnlineStats`` object."""
    if accumulator is None:
        return OnlineStats.from_data(values, chain_axis=chain_axis)
    return accumulator.update(values, chain_axis=chain_axis)


__all__ = ["OnlineStats", "online_statistics"]
