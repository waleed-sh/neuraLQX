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


"""Convenience constructors for Metropolis samplers."""

from __future__ import annotations

from typing import Any

from neuralqx.sampler.metropolis.sampler import MetropolisSampler
from neuralqx.sampler.rules import ExchangeRule
from neuralqx.sampler.rules import LocalRule
from neuralqx.sampler.rules import NeighborRule


def MetropolisLocal(hilbert, **kwargs: Any) -> MetropolisSampler:
    """Construct a Metropolis sampler using ``LocalRule``."""
    return MetropolisSampler(hilbert=hilbert, rule=LocalRule(), **kwargs)


def MetropolisNeighbor(hilbert, **kwargs: Any) -> MetropolisSampler:
    """Construct a Metropolis sampler using ``NeighborRule``."""
    return MetropolisSampler(hilbert=hilbert, rule=NeighborRule(), **kwargs)


def MetropolisExchange(hilbert, **kwargs: Any) -> MetropolisSampler:
    """Construct a Metropolis sampler using ``ExchangeRule`` arguments."""
    rule_keys = {"clusters", "graph", "d_max", "entity", "probabilities"}
    rule_kwargs = {key: value for key, value in kwargs.items() if key in rule_keys}
    sampler_kwargs = {
        key: value for key, value in kwargs.items() if key not in rule_keys
    }
    return MetropolisSampler(
        hilbert=hilbert,
        rule=ExchangeRule(**rule_kwargs),
        **sampler_kwargs,
    )


__all__ = ["MetropolisExchange", "MetropolisLocal", "MetropolisNeighbor"]
