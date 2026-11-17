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


"""Sampling measures and stable ratio statistics shared by VQS and SR."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.utils.stats import statistics
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field


@dataclass(frozen=True)
class TemperedApply:
    """Adapt a log-amplitude callable to the sampler's target exponent."""

    apply: Any
    """Original model evaluator, kept static across parameter updates."""

    def __call__(self, payload, states):
        """Evaluate the tempered log amplitude without changing physical variables."""
        variables, exponent = payload
        return self.apply(variables, states) * (exponent / 2)


class SamplingMeasure(Struct):
    """Frozen density information for one generated sample batch.

    ``log_density`` is unnormalized and may have an arbitrary common additive
    constant. It is held fixed when differentiating a physical expectation.
    The score is its exponent derivative minus the base normalizer derivative.
    All arrays use the same sample partitioning as their configurations.
    """

    log_density: Any
    """Log sampling density at each retained configuration."""

    exponent_score: Any
    """Derivative of the normalized sampling log density with respect to exponent."""

    exponent: Any
    """Exponent used to generate this batch, before any adaptation update."""

    batch_id: int = field(static=True, default=0)
    """Identifier preventing repeated adaptation on the same cached batch."""


class WeightedLocalEstimators(Struct):
    r"""Numerators and normalization samples for an importance-weighted mean.

    The mean is ``sum(numerator) / sum(denominator)``. Error bars use the ratio
    estimator's influence values, preserving chain boundaries for correlation
    diagnostics. ``variance`` describes these influence values, not the
    quantum variance of the operator. Ordinary finite-sample ratio bias remains.
    """

    numerator: Any
    """Per-configuration operator contribution divided by the frozen sampling density."""

    denominator: Any
    """Born density divided by the frozen sampling density, with the same common scale."""

    @property
    def mean(self):
        """Self-normalized expectation across all sample partitions."""
        return jnp.sum(self.numerator) / jnp.sum(self.denominator)

    @property
    def weights(self):
        """Nonnegative Born weights normalized over the global batch."""
        return self.denominator / jnp.sum(self.denominator)

    @property
    def data(self):
        """Weighted contributions whose ordinary mean equals the ratio estimate."""
        return self.numerator / jnp.mean(self.denominator)

    @property
    def ess(self):
        """Importance-weight effective sample size, before chain correlations."""
        return jnp.sum(self.denominator) ** 2 / jnp.sum(self.denominator**2)

    def to_stats(self):
        """Estimate error bars from centered ratio influence values."""
        mean = self.mean
        influence = (self.numerator - mean * self.denominator) / jnp.mean(
            self.denominator
        )
        return statistics(influence).replace(mean=mean)


def normalized_weights(log_values, log_density):
    """Normalize Born-to-sampling weights with a global log-sum-exp reduction."""
    logw = 2 * jnp.real(log_values) - log_density
    weights = jnp.exp(logw - jnp.max(logw))
    return jax.lax.stop_gradient(weights / jnp.sum(weights))
