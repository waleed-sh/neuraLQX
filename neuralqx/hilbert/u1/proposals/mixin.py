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


"""Proposal routing API for U(1) Hilbert spaces.

U(1) spaces support both generic flat-site proposals and U(1)-specific moves
such as free-edge flips and plaquette flips. This mixin routes move descriptors
to the appropriate proposal kernel.
"""

from __future__ import annotations

from typing import Any

import jax

from neuralqx.hilbert.proposal.moves import AbstractProposalMove
from neuralqx.hilbert.u1.proposals import AbstractU1Move
from neuralqx.hilbert.u1.proposals import FreeEdgeFlipAllGauge
from neuralqx.hilbert.u1.proposals import FreeEdgeFlipSingleGauge
from neuralqx.hilbert.u1.proposals import PlaquetteFlipAllGauge
from neuralqx.hilbert.u1.proposals import PlaquetteFlipSingleGauge
from neuralqx.hilbert.u1.proposals import free_edge_move
from neuralqx.hilbert.u1.proposals import propose_free_all
from neuralqx.hilbert.u1.proposals import propose_free_single
from neuralqx.hilbert.u1.proposals import propose_plaquette_all
from neuralqx.hilbert.u1.proposals import propose_plaquette_single


class U1ProposalMixin:
    """Proposal API shared by unconstrained and gauge-fixed U(1) spaces.

    The mixin accepts both U(1)-specific move descriptors and generic
    flat-site proposal descriptors. U(1) moves operate in graph terms, such as
    free base-edge updates and plaquette flips, while generic moves are routed
    through the package-level proposal dispatcher.

    Concrete spaces may override :attr:`_free_base_edges` and
    :meth:`_project_after_update` to restrict updates and restore constraints.
    ``U1Hilbert`` keeps all base edges free and returns updates unchanged,
    whereas ``U1GaugeInvariantHilbert`` exposes only independently sampled
    edges and reconstructs slave edges after each proposal.
    """

    def propose(self, move: Any, key: jax.Array, states: Any) -> jax.Array:
        """Applies a proposal move to U(1) states.

        Args:
            move: U(1)-specific or generic proposal move descriptor.
            key: JAX pseudo-random key.
            states: Input state array.

        Returns:
            Proposed state array with the same shape as ``states``.
        """
        if isinstance(move, AbstractU1Move):
            return self._propose_u1(move, key, states)
        if isinstance(move, AbstractProposalMove):
            from neuralqx.hilbert.proposal import propose as generic_propose

            return generic_propose(self, move, key, states)
        raise TypeError(f"Unsupported U(1) proposal move {type(move).__name__}.")

    def flip_state(
        self,
        states: Any,
        key: jax.Array,
        number_of_edges: int = 1,
        *,
        adjacency: bool = False,
        scope: str = "single",
    ) -> jax.Array:
        """Applies a free-edge flip proposal.

        Args:
            states: Input state array.
            key: JAX pseudo-random key.
            number_of_edges: Number of free base edges to update.
            adjacency: Whether updates should use adjacent local values.
            scope: Gauge-component update scope.

        Returns:
            Proposed state array.
        """
        move = free_edge_move(number_of_edges, adjacency=adjacency, scope=scope)
        return self.propose(move, key, states)

    def plaquette_flip(
        self,
        states: Any,
        key: jax.Array,
        *,
        all_gauge: bool = True,
    ) -> jax.Array:
        """Applies a plaquette flip proposal.

        Args:
            states: Input state array.
            key: JAX pseudo-random key.
            all_gauge: Whether to update all gauge components at once.

        Returns:
            Proposed state array.
        """
        move = PlaquetteFlipAllGauge() if all_gauge else PlaquetteFlipSingleGauge()
        return self.propose(move, key, states)

    @property
    def _free_base_edges(self) -> tuple[int, ...]:
        """Base graph edges eligible for free-edge proposals."""
        return tuple(range(self.tiny_size))

    def _project_after_update(self, states: jax.Array) -> jax.Array:
        """Projects updated states back into the valid U(1) subspace."""
        return states

    def _propose_u1(
        self, move: AbstractU1Move, key: jax.Array, states: Any
    ) -> jax.Array:
        """Routes a U(1)-specific proposal move to its kernel."""
        if isinstance(move, FreeEdgeFlipSingleGauge):
            return propose_free_single(self, move, key, states, self._free_base_edges)
        if isinstance(move, FreeEdgeFlipAllGauge):
            return propose_free_all(self, move, key, states, self._free_base_edges)
        if isinstance(move, PlaquetteFlipSingleGauge):
            return propose_plaquette_single(self, key, states)
        if isinstance(move, PlaquetteFlipAllGauge):
            return propose_plaquette_all(self, key, states)
        raise TypeError(f"Unsupported U(1) proposal move {type(move).__name__}.")
