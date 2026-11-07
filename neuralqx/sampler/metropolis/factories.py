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
from neuralqx.sampler.rules import SU2MetropolisRule


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


def MetropolisSU2(
    hilbert,
    *,
    weights=(0.45, 0.2, 0.25, 0.1),
    stay_probability=0.05,
    coherent_loop_probability=0.0,
    **kwargs: Any,
) -> MetropolisSampler:
    """SU(2) sampler with corrected channel proposals and full-support refreshes.

    Weights select loop, same-parity edge, vertex-channel, and global moves.
    The global weight and stay probability must be positive. ``sweep_size``
    counts attempted proposals per stored sample, not accepted state changes.
    ``coherent_loop_probability`` mixes common-sign loop shifts into the
    independent-sign proposal. Positive values help long cycles leave vacuum.
    """
    from neuralqx.hilbert.su2 import SU2GaugeInvariantHilbert

    if not isinstance(hilbert, SU2GaugeInvariantHilbert):
        raise TypeError("MetropolisSU2 requires SU2GaugeInvariantHilbert.")
    return MetropolisSampler(
        hilbert=hilbert,
        rule=SU2MetropolisRule(
            weights=weights,
            stay_probability=stay_probability,
            coherent_loop_probability=coherent_loop_probability,
        ),
        **kwargs,
    )


__all__ = [
    "MetropolisExchange",
    "MetropolisLocal",
    "MetropolisNeighbor",
    "MetropolisSU2",
]
