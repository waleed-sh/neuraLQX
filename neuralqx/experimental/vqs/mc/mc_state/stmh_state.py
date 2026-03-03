#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from typing import Sequence

from neuralqx.experimental.vqs.mc.mc_state.utils import same_treedef
from neuralqx.vqs import MCState


class STMultiMCState:
    """
    A Container for multiple head-specific ``MCState`` objects that share one parameter pytree.

    This class is intentionally small and driver-oriented. It assumes you already created one
    ``MCState`` per head (typically from ``STMHHeadView`` models) with identical parameter tree
    structure, and then synchronizes their parameters so they act as different *views* of the same
    ST-MH model.


    The current ``MultiMCState`` returns a list of parameter pytrees (one per state), which is
    exactly right for MT-MH (independent networks) but is not the ST-MH (shared trunk + heads in one
    joint parameter set) Ansatz. Here we expose a single parameter pytree to the base VMC driver so
    optimizer state and updates are computed only once.


    :param states:
        List of head-specific ``MCState`` objects. They must share the same Hilbert space and the same
        parameter-tree structure.
    :param canonical_state:
        Index of the state whose parameters are treated as the canonical source of truth.
    :param sync_model_state:
        If ``True``, ``broadcast_from_canonical()`` also copies ``model_state`` from the canonical
        state to all other states. Keep this ``False`` unless you really need mutable model-state
        collections and understand the implications.
    """

    def __init__(
        self,
        states: list[MCState],
        *,
        canonical_state: int = 0,
        sync_model_state: bool = False,
    ):
        if len(states) == 0:
            raise ValueError("STMultiMCState requires at least one MCState.")

        self.states = list(states)
        self._canonical_state = int(canonical_state)
        self._sync_model_state = bool(sync_model_state)

        # Hilbert space compatibility
        h0 = self.states[0].hilbert
        for i, st in enumerate(self.states[1:], start=1):
            if st.hilbert != h0:
                raise ValueError(
                    f"All head MCStates must share the same hilbert space. Index 0 and {i} mismatch."
                )
        self._hilbert = h0

        # Parameter tree compatibility
        p0 = self.states[self._canonical_state].parameters
        for i, st in enumerate(self.states):
            if not same_treedef(p0, st.parameters):
                raise ValueError(
                    f"All head MCStates must have identical parameter tree structure. "
                    f"State {i} mismatches canonical state {self._canonical_state}."
                )

        # Make them truly shared from the start.
        self.broadcast_from_canonical()

    @property
    def parameters(self):
        return self.states[self._canonical_state].parameters

    @parameters.setter
    def parameters(self, pars) -> None:
        for st in self.states:
            st.parameters = pars
        if self._sync_model_state:
            self._broadcast_model_state_from(self._canonical_state)

    @property
    def hilbert(self):
        return self._hilbert

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def canonical_state(self) -> int:
        return self._canonical_state

    @property
    def model(self):
        return [st.model for st in self.states]

    @property
    def sampler(self):
        return [st.sampler for st in self.states]

    @property
    def n_samples(self):
        return [st.n_samples for st in self.states]

    @n_samples.setter
    def n_samples(self, value: int):
        for st in self.states:
            st.n_samples = value

    @property
    def n_samples_per_rank(self):
        return [st.n_samples_per_rank for st in self.states]

    @n_samples_per_rank.setter
    def n_samples_per_rank(self, value: int):
        for st in self.states:
            st.n_samples_per_rank = value

    @property
    def chain_length(self):
        return [st.chain_length for st in self.states]

    @chain_length.setter
    def chain_length(self, value: int):
        for st in self.states:
            st.chain_length = value

    @property
    def n_discard_per_chain(self):
        return [st.n_discard_per_chain for st in self.states]

    @n_discard_per_chain.setter
    def n_discard_per_chain(self, value: int):
        for st in self.states:
            st.n_discard_per_chain = value

    @property
    def samples(self):
        return [st.samples for st in self.states]

    def to_array(self, normalize: bool = True):
        return [st.to_array(normalize) for st in self.states]

    def _broadcast_model_state_from(self, idx: int) -> None:
        ref = self.states[idx].model_state
        for st in self.states:
            st.model_state = ref

    def broadcast_from_canonical(self) -> None:
        ref_idx = self._canonical_state
        ref_params = self.states[ref_idx].parameters
        for st in self.states:
            st.parameters = ref_params
        if self._sync_model_state:
            self._broadcast_model_state_from(ref_idx)

    def reset(self) -> None:
        # Ensure all heads evaluate with identical shared parameters before (re-)sampling.
        self.broadcast_from_canonical()
        for st in self.states:
            st.reset()

    def sample(self, **kwargs):
        self.broadcast_from_canonical()
        for st in self.states:
            st.sample(**kwargs)

    def expect(self, O):
        self.broadcast_from_canonical()
        return [st.expect(O) for st in self.states]

    def __repr__(self) -> str:
        return (
            f"STMultiMCState(n_states={self.n_states}, "
            f"canonical_state={self.canonical_state}, hilbert={self.hilbert})"
        )


def make_shared_stmh_state(
    head_states: Sequence[MCState],
    *,
    canonical_state: int = 0,
    sync_model_state: bool = False,
) -> STMultiMCState:
    """
    Build a shared-parameter ST-MH container from already-created head ``MCState`` objects.

    Usage:
        1. Build one ST-MH base Flax model and ``K`` head views using ``STMHHeadView``.
        2. Create ``K`` MCState objects exactly as you do today (one per head view).
        3. Call ``make_shared_stmh_state(head_states)``.
        4. Pass the result to ``SingleTrunkMultiHeadVMC``.
    """
    return STMultiMCState(
        list(head_states),
        canonical_state=canonical_state,
        sync_model_state=sync_model_state,
    )
