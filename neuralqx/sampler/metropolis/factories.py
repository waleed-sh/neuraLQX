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

from neuralqx.hilbert.u1 import U1GaugeInvariantHilbert
from neuralqx.sampler.metropolis.sampler import MetropolisSampler
from neuralqx.sampler.rules import ExchangeRule
from neuralqx.sampler.rules import LocalRule
from neuralqx.sampler.rules import NeighborRule
from neuralqx.sampler.rules import SU2MetropolisRule
from neuralqx.sampler.rules import U1GaugeRule


def MetropolisLocal(hilbert, **kwargs: Any) -> MetropolisSampler:
    """Construct a Metropolis sampler using ``LocalRule``."""
    return MetropolisSampler(hilbert=hilbert, rule=LocalRule(), **kwargs)


def MetropolisNeighbor(hilbert, **kwargs: Any) -> MetropolisSampler:
    """Construct a Metropolis sampler using ``NeighborRule``."""
    return MetropolisSampler(hilbert=hilbert, rule=NeighborRule(), **kwargs)


def MetropolisU1Gauge(
    hilbert: U1GaugeInvariantHilbert,
    *,
    n_edges: int = 1,
    adjacency: bool = False,
    **kwargs: Any,
) -> MetropolisSampler:
    r"""Construct a gauge-preserving sampler for a U(1)^N Hilbert space.

    The default proposal replaces one independent charge selected from all
    gauge copies and reconstructs the dependent edges. It uses the constructive
    relations of ``U1GaugeInvariantHilbert`` and needs no proposal-probability
    correction. The Metropolis acceptance targets $|\psi(\sigma)|^p$, where
    ``machine_pow`` supplies $p$ and defaults to two.

    Args:
        hilbert: A ``U1GaugeInvariantHilbert`` with resolved gauge relations.
        n_edges: Number of distinct free charge coordinates to update per move.
        adjacency: Take modular nearest-neighbor steps instead of uniformly
            replacing the selected charges.
        **kwargs: Settings passed to ``MetropolisSampler``, including
            ``n_chains``, ``sweep_size``, and ``reset_chains``. A sweep counts
            attempted proposals per stored sample.

    Returns:
        A ``MetropolisSampler`` using ``U1GaugeRule``.

    Raises:
        TypeError: The Hilbert space or proposal settings have the wrong type.
        ValueError: The update count is nonpositive or exceeds the number of
            available free coordinates in a nontrivial space.
    """
    return MetropolisSampler(
        hilbert=hilbert,
        rule=U1GaugeRule(n_edges=n_edges, adjacency=adjacency),
        **kwargs,
    )


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
    "MetropolisU1Gauge",
]
