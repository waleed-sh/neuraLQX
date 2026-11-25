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
from neuralqx.hilbert.su2.proposals.metropolis import exchange_geometry
from neuralqx.hilbert.su2.proposals.metropolis import parity_sector
from neuralqx.hilbert.su2.proposals.metropolis import propose_metropolis
from neuralqx.hilbert.su2.proposals.metropolis import random_sector_warmup
from neuralqx.hilbert.su2.recoupling import left_comb_valid_py
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.struct import field

from .base import AbstractTransitionRule


class SU2MetropolisRule(AbstractTransitionRule):
    r"""SU(2) MH mixture with detailed balance and full finite-cutoff support.

    ``weights`` selects (loop, edge, vertex, global, exchange) moves. The
    original four-entry form remains supported and disables exchange. Weights are
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

    Without a fixed parity, initialization uses the Hilbert space's random
    valid-state generator.
    A cached exact basis permits uniform draws. Otherwise independent,
    finite uniform-target warmups produce gauge invariant starting states
    without enumerating the basis. This initialization is independent of
    the wavefunction and of this rule's training proposal weights. It does
    not replace thermalization against the current wavefunction. Explicit
    Hilbert initialization settings, including ``random_init="vacuum"``,
    remain available when that starting configuration is intentional.
    """

    weights: tuple[float, ...] = field(static=True, default=DEFAULT_WEIGHTS)
    stay_probability: float = field(static=True, default=DEFAULT_STAY_PROBABILITY)
    coherent_loop_probability: float = field(static=True, default=0.0)
    edge_parity: int | tuple[int, ...] | None = field(static=True, default=None)
    """Optional fixed parity on each edge. None retains the full Hilbert space.

    A binary scalar applies to every edge. A tuple supplies arbitrary edge
    parities, with even odd-edge incidence at each vertex. In this mode loop
    shifts are +/-2, globals refresh within the sector, and exchange rejects
    swaps violating the assigned parities. This changes the sampled space
    and must match the model's intended support. It is not inferred from the
    cutoff or Hamiltonian.
    """
    exchange_pairs: tuple[tuple[int, int], ...] | None = field(
        static=True, default=None
    )
    """Optional fixed pairs of edge indices. None draws distinct edges uniformly."""
    initial_state: tuple[int, ...] | None = field(static=True, default=None)
    """Optional valid initial configuration, for example a nonzero model coefficient.

    Otherwise sector initialization constructs minimal binary spins and runs
    the Hilbert warmup length within that sector. A supplied state is
    broadcast without warmup, so model thermalization is still needed.
    """

    def __post_init__(self):
        weights = tuple(float(w) for w in self.weights)
        if (
            len(weights) not in (4, 5)
            or not all(np.isfinite(w) and w >= 0 for w in weights)
            or sum(weights) <= 0
            or (self._requires_global() and weights[3] <= 0)
        ):
            raise NeuralqxValueError(
                "weights must have four or five finite nonnegative values, with positive global weight for SU2MetropolisRule."
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
        if self.edge_parity is not None:
            parity = np.asarray(self.edge_parity)
            if parity.ndim > 1 or not np.isin(parity, (0, 1)).all():
                raise NeuralqxValueError(
                    "edge_parity must be a binary scalar or sequence."
                )
            value = int(parity) if parity.ndim == 0 else tuple(int(p) for p in parity)
            object.__setattr__(self, "edge_parity", value)
        if self.exchange_pairs is not None:
            pairs = np.asarray(self.exchange_pairs)
            if (
                pairs.ndim != 2
                or pairs.shape[1] != 2
                or not np.equal(pairs, np.round(pairs)).all()
            ):
                raise NeuralqxValueError(
                    "exchange_pairs must be integer edge-index pairs."
                )
            object.__setattr__(
                self, "exchange_pairs", tuple(tuple(int(i) for i in p) for p in pairs)
            )
        if self.initial_state is not None:
            initial = np.asarray(self.initial_state)
            if (
                initial.ndim != 1
                or not np.isfinite(initial).all()
                or not np.equal(initial, np.round(initial)).all()
            ):
                raise NeuralqxValueError(
                    "initial_state must be a one-dimensional integer configuration."
                )
            object.__setattr__(self, "initial_state", tuple(int(i) for i in initial))

    def _requires_global(self):
        return True

    def _validate_geometry(self, space):
        sector = parity_sector(space, self.edge_parity)
        exchange_geometry(space, self.exchange_pairs)
        initial = self.initial_state
        if initial is not None:
            if len(initial) != space.size or any(
                t < 0 or t > space.two_j_max for t in initial[: space.n_edges]
            ):
                raise NeuralqxValueError(
                    "initial_state has an incorrect size or out-of-range spins."
                )
            if sector is not None and any(
                t % 2 != p
                for t, p in zip(initial[: space.n_edges], sector, strict=True)
            ):
                raise NeuralqxValueError(
                    "initial_state does not belong to edge_parity."
                )
            if not all(
                left_comb_valid_py(
                    [initial[e] for e in edges], [initial[s] for s in sites]
                )
                for edges, sites in zip(
                    space.metadata.incident_edges_by_vertex,
                    space.layout.vertex_channel_sites,
                    strict=True,
                )
            ):
                raise NeuralqxValueError("initial_state is not gauge invariant.")
        return sector

    def __hash__(self):
        return hash(
            (
                type(self),
                self.weights,
                self.stay_probability,
                self.coherent_loop_probability,
                self.edge_parity,
                self.exchange_pairs,
                self.initial_state,
            )
        )

    def init_state(self, sampler, apply_fn, parameters, key):
        self._validate_geometry(sampler.hilbert)
        return None

    def random_state_local(self, sampler, apply_fn, parameters, sampler_state, key):
        """Draw valid initial configurations using the Hilbert initialization policy.

        The default policy randomizes spins and intertwiner channels before
        the configurations enter the variational chains. Each sample shard
        supplies its own key and local batch size through the common rule
        interface. The vacuum remains an allowed random outcome and is the
        only outcome when the gauge invariant space is trivial.
        """
        sector = self._validate_geometry(sampler.hilbert)
        if self.initial_state is not None:
            return jnp.broadcast_to(
                jnp.asarray(self.initial_state, dtype=sampler.dtype),
                (sampler.n_batches, sampler.hilbert.size),
            )
        if sector is not None:
            steps = sampler.hilbert.init_warmup
            steps = max(32, 4 * sampler.hilbert.n_edges) if steps is None else steps
            return random_sector_warmup(
                sampler.hilbert, key, sampler.n_batches, steps, sector
            ).astype(sampler.dtype)
        return super().random_state_local(
            sampler, apply_fn, parameters, sampler_state, key
        )

    def global_log_proposal(self, hilbert, states):
        """Log mass for a *valid* global draw, before rejection of invalid draws.

        Invalid states have mass zero. Missing mass is a rejected proposal,
        not a normalization constant to divide out. This method is useful
        for auditing the Hastings correction without enumerating states.
        """
        sector = self._validate_geometry(hilbert)
        states = jnp.asarray(states)
        shape = states.shape[:-1]
        flat = states.reshape((-1, hilbert.size))
        log_p = _edge_log_prob(hilbert, flat[:, : hilbert.n_edges])
        if hilbert.two_j_max > 0 and sector is None:
            basis, _ = _cycle_geometry(hilbert)
            log_p -= len(basis) * jnp.log(2.0)
        if sector is not None:
            log_p = jnp.where(
                jnp.all(flat[:, : hilbert.n_edges] % 2 == jnp.asarray(sector), axis=-1),
                log_p,
                -jnp.inf,
            )
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
            edge_parity=self.edge_parity,
            exchange_pairs=self.exchange_pairs,
        )


class SU2ExchangeRule(SU2MetropolisRule):
    """Swap edge spins and redraw affected intertwiners with the exact MH ratio.

    Unlike a flat-site exchange, this preserves gauge admissibility. Invalid
    attempted swaps stay put. Pair selection is symmetric and independent of
    the state. Channel resampling contributes log q(old)-log q(new).

    Exchange alone preserves the edge-spin histogram and cannot sample the
    entire Hilbert space or parity sector. Prefer the fifth mixture weight
    of SU2MetropolisRule to include spin-changing and global refresh moves.
    """

    weights: tuple[float, ...] = field(static=True, default=(0, 0, 0, 0, 1))

    def _requires_global(self):
        return False

    def __post_init__(self):
        super().__post_init__()
        if self.weights != (0, 0, 0, 0, 1):
            raise NeuralqxValueError(
                "SU2ExchangeRule only accepts pure exchange weights."
            )


__all__ = ["SU2ExchangeRule", "SU2MetropolisRule"]
