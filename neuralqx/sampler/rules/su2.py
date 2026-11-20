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


"""Reversible, full-support Metropolis proposals for SU(2) spin networks.

The proposal core is shared with Hilbert random-state initialization.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from neuralqx.hilbert.su2.proposals.metropolis import DEFAULT_STAY_PROBABILITY
from neuralqx.hilbert.su2.proposals.metropolis import DEFAULT_WEIGHTS
from neuralqx.hilbert.su2.proposals.metropolis import _cycle_geometry
from neuralqx.hilbert.su2.proposals.metropolis import _edge_log_prob
from neuralqx.hilbert.su2.proposals.metropolis import _vertex_channels
from neuralqx.hilbert.su2.proposals.metropolis import _vertex_groups as _vertex_groups
from neuralqx.hilbert.su2.proposals.metropolis import propose_metropolis
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.struct import field

from .base import AbstractTransitionRule


class SU2MetropolisRule(AbstractTransitionRule):
    r"""SU(2) MH mixture with detailed balance and full finite-cutoff support.

    ``weights`` selects (loop, edge, vertex, global) moves; weights are
    constant in the state. A separate ``stay_probability`` makes the kernel
    lazy. The global weight must be positive to guarantee irreducibility.

    Loop moves independently shift each selected edge by +/-1. With probability
    ``coherent_loop_probability`` they instead use one common sign for the
    selected loop, which helps long cycles leave the vacuum. Both choices have
    symmetric reverse shifts. The default zero preserves independent signs.
    Edge moves redraw a spin of the same parity; vertex moves redraw channels.
    Changed endpoints draw admissible channels with sequential feasible draws.
    Every branch returns log q(reverse) - log q(forward). Invalid attempts
    stay put, with no retry-until-valid conditioning.

    Global moves choose a uniform cycle-space parity vector, then uniform
    edge labels of those parities and feasible channels. Every admissible
    state has positive proposal probability. Together with laziness, MH is
    ergodic on the positive support of |psi|^machine_pow at fixed parameters.
    This is a support/convergence guarantee, not a bound on mixing time.

    Initialization uses the Hilbert space's random valid-state generator.
    A cached exact basis permits uniform draws. Otherwise independent,
    finite uniform-target warmups produce gauge-invariant starting states
    without enumerating the basis. This initialization is independent of
    the wavefunction and of this rule's training proposal weights. It does
    not replace thermalization against the current wavefunction. Explicit
    Hilbert initialization settings, including ``random_init="vacuum"``,
    remain available when that starting configuration is intentional.
    """

    weights: tuple[float, float, float, float] = field(
        static=True, default=DEFAULT_WEIGHTS
    )
    stay_probability: float = field(static=True, default=DEFAULT_STAY_PROBABILITY)
    coherent_loop_probability: float = field(static=True, default=0.0)

    def __post_init__(self):
        weights = tuple(float(w) for w in self.weights)
        if (
            len(weights) != 4
            or not all(np.isfinite(w) and w >= 0 for w in weights)
            or weights[3] <= 0
        ):
            raise NeuralqxValueError(
                "weights must be four finite nonnegative values with positive global weight."
            )
        total = sum(weights)
        if not np.isfinite(total):
            raise NeuralqxValueError("The sum of weights must be finite.")
        if not np.isfinite(self.stay_probability) or not 0 < self.stay_probability < 1:
            raise NeuralqxValueError(
                "stay_probability must be strictly between zero and one."
            )
        if (
            not np.isfinite(self.coherent_loop_probability)
            or not 0 <= self.coherent_loop_probability <= 1
        ):
            raise NeuralqxValueError(
                "coherent_loop_probability must be between zero and one."
            )
        object.__setattr__(self, "weights", tuple(w / total for w in weights))

    def __hash__(self):
        return hash(
            (
                type(self),
                self.weights,
                self.stay_probability,
                self.coherent_loop_probability,
            )
        )

    def init_state(self, sampler, apply_fn, parameters, key):
        _cycle_geometry(sampler.hilbert)
        return None

    def random_state_local(self, sampler, apply_fn, parameters, sampler_state, key):
        """Draw valid initial configurations using the Hilbert initialization policy.

        The default policy randomizes spins and intertwiner channels before
        the configurations enter the variational chains. Each sample shard
        supplies its own key and local batch size through the common rule
        interface. The vacuum remains an allowed random outcome and is the
        only outcome when the gauge-invariant space is trivial.
        """
        _cycle_geometry(sampler.hilbert)
        return super().random_state_local(
            sampler, apply_fn, parameters, sampler_state, key
        )

    def global_log_proposal(self, hilbert, states):
        """Log mass for a *valid* global draw, before rejection of invalid draws.

        Invalid states have mass zero. Missing mass is a rejected proposal,
        not a normalization constant to divide out. This method is useful
        for auditing the Hastings correction without enumerating states.
        """
        basis, _ = _cycle_geometry(hilbert)
        states = jnp.asarray(states)
        shape = states.shape[:-1]
        flat = states.reshape((-1, hilbert.size))
        log_p = _edge_log_prob(hilbert, flat[:, : hilbert.n_edges])
        if hilbert.two_j_max > 0:
            log_p -= len(basis) * jnp.log(2.0)
        for edges, sites in zip(
            hilbert.metadata.incident_edges_by_vertex,
            hilbert.layout.vertex_channel_sites,
            strict=True,
        ):
            _, p = _vertex_channels(
                flat[:, jnp.asarray(edges, dtype=jnp.int32)],
                channels=flat[:, jnp.asarray(sites, dtype=jnp.int32)],
            )
            log_p += p
        return jnp.where(hilbert.is_valid(flat), log_p, -jnp.inf).reshape(shape)

    def transition(self, sampler, apply_fn, parameters, sampler_state, key, sigma):
        del apply_fn, parameters, sampler_state
        return propose_metropolis(
            sampler.hilbert,
            key,
            sigma,
            weights=self.weights,
            stay_probability=self.stay_probability,
            coherent_loop_probability=self.coherent_loop_probability,
        )


__all__ = ["SU2MetropolisRule"]
