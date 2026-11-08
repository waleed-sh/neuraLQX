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


"""Gauge-preserving Metropolis proposals for constructive U(1)^N spaces."""

from __future__ import annotations

from collections.abc import Callable
from operator import index
from typing import TYPE_CHECKING
from typing import Any

import jax
from neuralqx.hilbert.u1 import FreeEdgeFlipSingleGauge
from neuralqx.hilbert.u1 import U1GaugeInvariantHilbert
from neuralqx.utils.struct import field

from .base import AbstractTransitionRule

if TYPE_CHECKING:
    from neuralqx.sampler.metropolis import MetropolisSampler
    from neuralqx.sampler.state import MetropolisSamplerState


class U1GaugeRule(AbstractTransitionRule):
    r"""Update independent U(1) charges and reconstruct the dependent edges.

    A proposal selects ``n_edges`` distinct free charge coordinates uniformly
    from all gauge copies. By default each selected charge is replaced by a
    uniformly drawn value from the Hilbert space's local range, including its
    current value. ``adjacency=True`` instead takes a random modular step of
    one charge unit in either direction. Reconstruction then imposes the
    Hilbert space's gauge relations in every copy.

    The free coordinates determine a unique valid configuration. Both update
    rules are symmetric in those coordinates, so
    $q(\sigma'\mid\sigma)=q(\sigma\mid\sigma')$. No Hastings correction is
    needed. With the default one-coordinate replacement, every configuration
    in the constructive gauge-fixed space can be reached by successive moves.
    The existing Hilbert proposal kernel and the Metropolis chain are compiled
    by JAX without enumerating the basis.

    A space with no free coordinates has only one valid configuration. Its
    proposals leave that configuration unchanged.

    Attributes:
        n_edges (int): Number of free charge coordinates to update across all copies.
        adjacency (bool): Use modular nearest-neighbor steps instead of replacement.
    """

    n_edges: int = field(static=True, default=1)
    """Number of distinct free charge coordinates selected per proposal."""

    adjacency: bool = field(static=True, default=False)
    """Whether selected charges take random modular steps of one charge unit."""

    def __post_init__(self) -> None:
        """Require a positive integer update count and a boolean move choice."""
        if isinstance(self.n_edges, bool):
            raise TypeError("n_edges must be a positive integer.")
        try:
            n_edges = index(self.n_edges)
        except TypeError as exc:
            raise TypeError("n_edges must be a positive integer.") from exc
        if n_edges <= 0:
            raise ValueError("n_edges must be positive.")
        if not isinstance(self.adjacency, bool):
            raise TypeError("adjacency must be a boolean.")
        object.__setattr__(self, "n_edges", n_edges)

    def validate_hilbert(self, hilbert: Any) -> None:
        """Check that the space supports constructive gauge-preserving moves.

        Args:
            hilbert: Hilbert space supplied to the Metropolis sampler.

        Raises:
            TypeError: The space is not a ``U1GaugeInvariantHilbert``.
            ValueError: A nonempty free-coordinate set is smaller than
                ``n_edges``.
        """
        if not isinstance(hilbert, U1GaugeInvariantHilbert):
            raise TypeError("U1GaugeRule requires U1GaugeInvariantHilbert.")
        n_free = hilbert.gauge_dimensions * hilbert.topology.n_free
        if n_free and self.n_edges > n_free:
            raise ValueError(
                f"Cannot update {self.n_edges} free U(1) charges from {n_free}."
            )

    def transition(
        self,
        sampler: MetropolisSampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state: MetropolisSamplerState,
        key: jax.Array,
        sigma: jax.Array,
    ) -> tuple[jax.Array, None]:
        """Propose a valid configuration for each chain with zero log correction.

        Args:
            sampler: Metropolis sampler supplying the gauge-invariant space.
            apply_fn: Wavefunction evaluator, unused by this symmetric proposal.
            parameters: Network parameters, unused by the proposal.
            sampler_state: Current chain metadata, unused by the proposal.
            key: JAX random key for the batched update.
            sigma: Current configurations of shape ``(n_chains, hilbert.size)``.

        Returns:
            Proposed configurations and ``None`` for the Hastings correction.
        """
        del apply_fn, parameters, sampler_state
        if not sampler.hilbert.topology.free:
            return sigma, None
        # Keep the key convention of the upstream gauge sampler and bench.py.
        _, proposal_key = jax.random.split(key)
        move = FreeEdgeFlipSingleGauge(n_edges=self.n_edges, adjacency=self.adjacency)
        return sampler.hilbert.propose(move, proposal_key, sigma), None

    def __hash__(self) -> int:
        """Hash the static update settings for compiled sampler reuse."""
        return hash((type(self), self.n_edges, self.adjacency))


__all__ = ["U1GaugeRule"]
