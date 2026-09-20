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


"""Shared helpers for expectation-level penalty wrappers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.operator.wrappers import penalty_local_value_coefficients

from .statistics import Stats


def penalty_stats_transform_inputs(operator: PenaltyCost, parent_stats: Stats):
    """Return ``(parent_stats, scale, shift)`` for a penalty wrapper."""
    scale, shift = penalty_local_value_coefficients(operator, parent_stats.mean)
    return parent_stats, scale, shift


def transform_stats(parent_stats: Stats, scale: Any, shift: Any) -> Stats:
    """Apply an affine local-estimator transform to summary statistics."""
    scale_arr = jnp.asarray(scale)
    shift_arr = jnp.asarray(shift)
    mean = scale_arr * parent_stats.mean + shift_arr
    variance = jnp.abs(scale_arr) ** 2 * parent_stats.variance
    error = jnp.abs(scale_arr) * parent_stats.error_of_mean
    return Stats(mean, variance, error, parent_stats.n_samples, parent_stats.r_hat)


__all__ = ["penalty_stats_transform_inputs", "transform_stats"]
