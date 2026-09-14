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


"""Validation helpers for Metropolis samplers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neuralqx.sampler.rules import AbstractTransitionRule
from neuralqx.sampler.state import AbstractSamplerState
from neuralqx.sampler.state import MetropolisSamplerState


def ensure_metropolis_state(state: AbstractSamplerState) -> MetropolisSamplerState:
    """Return ``state`` as a Metropolis state or raise a helpful error."""
    if not isinstance(state, MetropolisSamplerState):
        raise TypeError(
            "MetropolisSampler expects MetropolisSamplerState; "
            f"got {type(state).__name__}."
        )
    return state


def validate_rule_for_hilbert(rule: AbstractTransitionRule, hilbert_size: int) -> None:
    """Validate static rule data against the flat Hilbert size."""
    clusters = getattr(rule, "clusters", None)
    if clusters is None:
        return
    arr = jnp.asarray(clusters)
    if arr.size == 0:
        return
    if bool(jnp.any(arr < 0)) or bool(jnp.any(arr >= int(hilbert_size))):
        raise ValueError(
            "Exchange clusters must reference flat Hilbert sites in "
            f"[0, {int(hilbert_size)})."
        )


def assert_sample_shape(samples: Any, n_batches: int, hilbert_size: int) -> None:
    """Validate the rank-2 chain batch shape produced by a transition rule."""
    arr = jnp.asarray(samples)
    expected = (int(n_batches), int(hilbert_size))
    if arr.shape != expected:
        raise ValueError(
            f"sample shape {arr.shape} does not match expected {expected}."
        )


__all__ = [
    "assert_sample_shape",
    "ensure_metropolis_state",
    "validate_rule_for_hilbert",
]
