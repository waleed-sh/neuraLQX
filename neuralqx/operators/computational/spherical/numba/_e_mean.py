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

from neuralqx.graph import HalfLadderGraph
from neuralqx.operators.types.computational_operator import ComputationalOperator


class SphericalEx(ComputationalOperator):
    r"""
    Diagonal E^x at a single vertex:

        E^x_v(σ) = K(v) / 2

    with K(v) = k_-(v) + k_+(v). For endpoints, any missing k-edge is treated as 0 so that K(v) sums
    only the available incident k’s.

    Note:
        - we do not mod_add the sum K(v)
        - we use an Immirzi parameter of 1.0 by default
    """

    def __init__(self, H, vertex: int, *, gamma: float = 1.0):
        super().__init__(H.hilbert)

        # HalfLadderGraph type check
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires the graph to be an instance of "
                f"`{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}` instead."
            )

        #
        #
        # record host/invariants

        # Hilbert space object
        self._H = H

        # number of DoFs
        self.D = int(H.size)

        # the cutoff
        self.cutoff = int(H.cutoff)

        # create a span from it
        # self._mod_span = int(2 * self.cutoff + 1)

        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )
        self._state_step = jnp.asarray(
            int(H.allowed_basis_states.step), dtype=jnp.int64
        )
        self._state_max = jnp.asarray(
            int(self._state_min + (self._mod_span - 1) * self._state_step),
            dtype=jnp.int64,
        )

        # store target vertex
        self.v = int(vertex)

        # store Immirzi
        self.immirzi = gamma

        # resolve incident edges at v
        mu_list, k_list = H.graph.get_edges_at_k_vertex(self.v)

        # sanity check/guard against custom malformed graph implementations
        if len(mu_list) != 1:
            raise RuntimeError(
                f"{type(self).__name__}: expected exactly one μ-edge at v={self.v}, got {mu_list}"
            )

        #
        #
        # identify left/right k-edges if present
        #
        # in principle, we do this to allow for the operator to act also on non-"inner" vertices
        # note that this treatment assumes the default orientation implemented in the
        # HalfLadderGraph class and WILL break/give wrong results if changed

        # (v-1, v, *)
        left_candidates = [e for e in k_list if e[1] == self.v]

        # (v, v+1, *)
        right_candidates = [e for e in k_list if e[0] == self.v]

        # check if both k_minus and k_plus are there
        self._has_km = len(left_candidates) == 1
        self._has_kp = len(right_candidates) == 1

        # if so, map the edges to indices
        self._idx_km = (
            jnp.int32(self._H.graph.edge_to_index(left_candidates[0]))
            if self._has_km
            else jnp.int32(-1)
        )
        self._idx_kp = (
            jnp.int32(self._H.graph.edge_to_index(right_candidates[0]))
            if self._has_kp
            else jnp.int32(-1)
        )

        # one diagonal connection per input configuration
        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    @staticmethod
    def _wrap_add(
        vals: jnp.ndarray,
        mod_span: Union[int, float],
        state_min: Union[int, float],
    ) -> jnp.ndarray:
        """
        Modular wrapping of quantum numbers into [state_min, state_max].

        Works for both symmetric ([-N,N]) and positive ([0, 2N-1]) Hilbert spaces.
        """
        return ((vals - state_min) % mod_span) + state_min

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # normalise input shape to (B, N, D)
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        BN = B * N
        sig = σ.reshape(BN, D)

        # read k_-(v) and/or k_+(v), use 0 if missing
        k_m = (
            sig[:, int(self._idx_km)]
            if self._has_km
            else jnp.zeros((BN,), dtype=sig.dtype)
        )
        k_p = (
            sig[:, int(self._idx_kp)]
            if self._has_kp
            else jnp.zeros((BN,), dtype=sig.dtype)
        )

        # K(v) = k_- + k_+
        K = (k_m + k_p).astype(jnp.int32)

        #
        #
        # optional K wrapping
        #
        # if you want to wrap K to [-cutoff, cutoff] BEFORE forming E^x = K/2,
        # just uncomment the next line:
        # K = self._wrap_add(K, self._mod_span, self._state_min)

        Ex = 0.5 * self.immirzi * K.astype(jnp.float64)

        # assemble outputs (diagonal only)
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D))
        mels = Ex.reshape(B, N, 1).astype(self.dtype)

        # if the output was single, return the correct expected shapes
        if single:
            return σp[0], mels[0]

        return σp, mels
