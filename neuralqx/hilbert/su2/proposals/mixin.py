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


"""Proposal routing for SU(2) Hilbert spaces.

SU(2) spaces support loop, vertex-recouping, edge heatbath, and mixed proposal
moves. Generic flat-site proposal descriptors are delegated back to the generic
Hilbert proposal dispatcher.
"""

from __future__ import annotations

from typing import Any

import jax

from neuralqx.hilbert.proposal.moves import AbstractProposalMove
from neuralqx.hilbert.su2.proposals import AbstractSU2Move
from neuralqx.hilbert.su2.proposals import EdgeSpinHeatbathMove
from neuralqx.hilbert.su2.proposals import LoopFundamentalMove
from neuralqx.hilbert.su2.proposals import MixedSU2Move
from neuralqx.hilbert.su2.proposals import VertexRecouplingMove
from neuralqx.hilbert.su2.proposals import propose_edge_heatbath
from neuralqx.hilbert.su2.proposals import propose_loop
from neuralqx.hilbert.su2.proposals import propose_mixed_su2
from neuralqx.hilbert.su2.proposals import propose_vertex_recoupling
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.hilbert.utils import restore_trailing


class SU2ProposalMixin:
    """Proposal API for gauge-invariant SU(2) spin-network states.

    The mixin routes SU(2)-specific move descriptors to kernels that preserve
    or restore vertex admissibility. Loop moves shift doubled spins around graph
    loops, vertex moves resample recoupling channels, edge heatbath moves try
    new edge sectors and complete channels, and mixed moves choose among those
    families.

    Generic flat-site move descriptors are delegated to the common Hilbert
    proposal dispatcher. Concrete SU(2) spaces must provide graph metadata,
    layout metadata, state size, and the validation helpers required by the
    proposal kernels.
    """

    def propose(self, move: Any, key: jax.Array, states: Any) -> jax.Array:
        """Applies a proposal move to SU(2) spin-network states.

        Args:
            move: SU(2)-specific or generic proposal move descriptor.
            key: JAX pseudo-random key.
            states: Input state array with trailing dimension ``self.size``.

        Returns:
            Proposed state array with the same shape as ``states``.

        Raises:
            TypeError: If ``move`` is neither an SU(2) descriptor nor a generic
            Hilbert proposal descriptor.
        """
        if isinstance(move, AbstractSU2Move):
            return self._propose_su2(move, key, states)
        if isinstance(move, AbstractProposalMove):
            from neuralqx.hilbert.proposal import propose as generic_propose

            return generic_propose(self, move, key, states)
        raise TypeError(f"Unsupported SU(2) proposal move {type(move).__name__}.")

    def flip_state(
        self,
        states: Any,
        key: jax.Array,
        number_of_edges: int = 1,
        *,
        adjacency: bool = False,
        scope: str = "loop",
    ) -> jax.Array:
        """Applies the default SU(2) loop-fundamental proposal.

        Args:
            states: Input spin-network state array.
            key: JAX pseudo-random key.
            number_of_edges: Accepted for compatibility with U(1)-style
                callers and ignored.
            adjacency: Accepted for compatibility and ignored.
            scope: Accepted for compatibility and ignored.

        Returns:
            Proposed state array from :class:`LoopFundamentalMove`.
        """
        del number_of_edges, adjacency, scope
        return self.propose(LoopFundamentalMove(), key, states)

    def _propose_su2(
        self, move: AbstractSU2Move, key: jax.Array, states: Any
    ) -> jax.Array:
        """Routes a typed SU(2) move descriptor to its proposal kernel."""
        if isinstance(move, MixedSU2Move):
            arr, single, batch = ensure_2d_trailing(states, self.size)
            proposed = propose_mixed_su2(self, arr, key, move)
            return restore_trailing(proposed, single, batch)
        if isinstance(move, LoopFundamentalMove):
            arr, single, batch = ensure_2d_trailing(states, self.size)
            proposed = propose_loop(self, arr, key, move)
            return restore_trailing(proposed, single, batch)
        if isinstance(move, VertexRecouplingMove):
            arr, single, batch = ensure_2d_trailing(states, self.size)
            proposed = propose_vertex_recoupling(self, arr, key, move)
            return restore_trailing(proposed, single, batch)
        if isinstance(move, EdgeSpinHeatbathMove):
            arr, single, batch = ensure_2d_trailing(states, self.size)
            proposed = propose_edge_heatbath(self, arr, key, move)
            return restore_trailing(proposed, single, batch)
        raise TypeError(f"Unsupported SU(2) proposal move {type(move).__name__}.")


__all__ = ["SU2ProposalMixin"]
