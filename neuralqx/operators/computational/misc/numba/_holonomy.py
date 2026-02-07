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


from __future__ import annotations

from typing import Tuple
from typing import Union

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


def _wrap_add(
    vals: jnp.ndarray,
    delta: jnp.ndarray,
    mod_span: Union[int, float],
    state_min: Union[int, float],
) -> jnp.ndarray:
    """
    Modular wrapping of quantum numbers into [state_min, state_max].

    Works for both symmetric ([-N,N]) and positive ([0, 2N-1]) Hilbert spaces.
    """
    # return (vals + delta + cutoff) % mod_span - cutoff
    return ((vals + delta - state_min) % mod_span) + state_min


class U1Holonomy(ComputationalOperator):
    r"""
    A pure U(1) holonomy on a single edge index `edge_index`:
      - Emits one connected configuration σ' where that edge is shifted by ±1 (with wrap-mod).
      - Matrix element (weight) is identically 1.
    """

    def __init__(self, H, edge_index: int, *, adjoint: bool = False):

        super().__init__(H.hilbert)

        # ensure gauge dims are 1
        if int(H.core.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )

        # record host/invariants

        # Hilbert object
        self._H = H

        # number of DoFs
        self.D = int(H.size)

        # the cutoff and a span for the modded sums
        self.cutoff = int(H.cutoff)
        # self._mod_span = int(2 * self.cutoff + 1)

        # get the allowed quantum numbers
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )
        self._state_max = jnp.asarray(
            int(
                self._state_min
                + (self._mod_span - 1) * int(H.allowed_basis_states.step)
            ),
            dtype=jnp.int64,
        )

        # guard against malformed edge indices
        if not (isinstance(edge_index, int) and 0 <= edge_index < self.D):
            raise ValueError(
                f"`edge_index` must be in [0, {self.D-1}], got {edge_index} instead."
            )

        # all clear, store it
        self.edge_index = int(edge_index)

        # adjoint check
        if adjoint:
            # adjoints are lowering
            self.direction = -1
        else:
            # non-adjoints are raising
            self.direction = 1

        # single off-diagonal
        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # normalise input shapes
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        BN = B * N
        sig = σ.reshape(BN, D)

        # shift the selected edge by ±1 with wrap

        # get the egde indices as an array
        comps = jnp.asarray([self.edge_index], jnp.int32)

        # create the deltas with appropriate shape
        deltas = jnp.full((BN, 1), self.direction, jnp.int32)

        # slice the inputs to only get the relevant components to be updated
        cur = sig[:, comps]

        # update them with modded addition
        upd = _wrap_add(cur, deltas, self._mod_span, self._state_min).astype(sig.dtype)

        # update back the configurations
        sig_prime = sig.at[:, comps].set(upd)

        # emit σp and unit matrix element
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, 1, D)).copy()
        σp = σp.at[:, :, 0, :].set(sig_prime.reshape(B, N, D))

        mels = jnp.ones((B, N, 1), dtype=self.dtype)

        # if the input was single, return it as so
        if single:
            return σp[0], mels[0]

        return σp, mels


class NDU1Holonomy(ComputationalOperator):
    r"""
    N-D (K-copy) holonomy on a single base edge index `edge_index`:

    Decomposed action:
        h_1 ⊗ I ⊗ ... + I ⊗ h_2 ⊗ ... + ... + I ⊗ ... ⊗ h_K

    - For each gauge copy k ∈ {0, ..., K-1}, produces one connected configuration σ'_k
      where only the k-th copy's edge is shifted by ±1
    - Matrix elements are all 1.
    - `adjoint=True` flips the direction to lowering (-1), otherwise raising (+1).
    """

    def __init__(self, H, edge_index: int, *, adjoint: bool = False):
        super().__init__(H.hilbert)

        # host invariants
        self._H = H

        # number of gauge copies
        self.K = int(H.core.gauge_dimensions)
        if self.K < 1:
            raise ValueError("`gauge_dimensions` must be ≥ 1.")

        self.D = int(H.size)
        # number of edges per gauge copy (use hi.size for per-copy width if available)
        self.n_edges_total = int(H.tiny_size)

        # validate base edge index (within a single copy)
        if not (isinstance(edge_index, int) and 0 <= edge_index < self.n_edges_total):
            raise ValueError(
                f"`edge_index` must be in [0, {self.n_edges_total-1}] within a single gauge copy, "
                f"got {edge_index}."
            )
        self.edge_index = int(edge_index)

        # wrap params from allowed local states
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )

        # direction (+1 or -1)
        self.direction = -1 if adjoint else +1

        # number of connections equals number of gauge copies
        self._nconn = self.K

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # normalise input shapes
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        if D != self.D:
            raise ValueError(f"Input last dim D={D} mismatches operator D={self.D}.")

        BN = B * N
        sig = σ.reshape(BN, D)

        # prepare K connected copies
        # σp: (B, N, K, D), start as K repeats of the original σ
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D)).copy()

        # for each gauge copy k, shift only that copy's component of the base edge
        # global index for copy k: base_edge_index + k * n_edges_total
        base = self.edge_index
        copy_offsets = jnp.arange(self.K, dtype=jnp.int32) * self.n_edges_total
        comp_indices = base + copy_offsets

        # build deltas for all BN batch elements, per connection k
        deltas = jnp.full((BN, 1), self.direction, dtype=jnp.int32)

        # apply per-connection update with wrap-mod
        sig_flat = sig
        for k in range(self.K):
            idx = int(comp_indices[k])

            # extract current values and wrap-add
            cur = sig_flat[:, [idx]]
            upd = _wrap_add(cur, deltas, self._mod_span, self._state_min).astype(
                sig_flat.dtype
            )

            # write back into the k-th connection
            # reshape view of kth connection to (BN, D), update index, then set into σp
            sig_k = σp[:, :, k, :].reshape(BN, D)
            sig_k = sig_k.at[:, [idx]].set(upd)
            σp = σp.at[:, :, k, :].set(sig_k.reshape(B, N, D))

        # matrix elements (all ones), shape (B, N, K)
        mels = jnp.ones((B, N, self._nconn), dtype=self.dtype)

        # if input was (N, D), return (K, D) / (K,) forms
        if single:
            return σp[0], mels[0]

        return σp, mels
