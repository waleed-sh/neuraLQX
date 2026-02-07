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


from typing import List
from typing import Any
from typing import Tuple
from typing import Union

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


def _type_to_step(t: Any) -> int:
    """
    Map the dressed 'type' field to an integer step (+1 or -1)
    """
    if isinstance(t, str):
        s = t.strip().lower()
        # right now we use creation and annihilation, but just for future proofing
        if s in ("creation", "+", "up", "plus"):
            return +1
        if s in ("annihilation", "-", "down", "minus"):
            return -1
    if isinstance(t, (int, jnp.integer)):
        return +1 if int(t) > 0 else -1
    # fallback: assume raising
    return +1


def _wrap_add(
    vals: jnp.ndarray,
    delta: jnp.ndarray,
    mod_span: Union[int, float],
    state_min: Union[int, float],
) -> jnp.ndarray:
    # Wrap onto [state_min, state_min + mod_span - 1]
    return ((vals + delta - state_min) % mod_span) + state_min


class NDMinimalLoopHolonomy(ComputationalOperator):
    """
    N-D minimal loop holonomy as a light-weight ComputationalOperator.

    This implements the decomposed action
        h_1 ⊗ I ⊗ ... + I ⊗ h_2 ⊗ ... + ... + I ⊗ ... ⊗ h_N

    Each h_k acts on a single gauge copy (k-th) by shifting the charges along the provided dressed
    minimal loop. The direction (+1 or -1) for each edge in the loop is deduced from the "type"
    entry in the dressed data.
    """

    def __init__(self, H, minimal_loop: List, *, adjoint: int = 0):

        super().__init__(H.hilbert)

        # get the gauge dimensions
        self.gauge_dim = int(H.gauge_dimensions)

        # number of edges per gauge copy
        self.n_edges_total = int(H.size // self.gauge_dim)

        # capture allowed local states and wrap parameters once
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )

        # parse the dressed loop once: edges -> flat indices on base copy, steps -> +/-1
        base_edge_idx = []
        steps = []

        for elem in minimal_loop:
            # elem is ((u, v, key), {"type": ...}) per the graph API of v0.0.9
            e_tuple = elem[0]
            t_info = elem[1]
            e_idx = H.graph.edge_to_index((e_tuple[0], e_tuple[1], e_tuple[2]))
            base_edge_idx.append(e_idx)
            steps.append(_type_to_step(t_info.get("type", "raise")))

        base_edge_idx = jnp.array(base_edge_idx, dtype=jnp.int32)
        steps = jnp.array(steps, dtype=jnp.int32)

        # if adjoint: invert all steps
        if int(adjoint) == 1:
            steps = -steps

        # loop length
        self._L = int(base_edge_idx.size)
        self._steps = steps

        # precompute for each gauge copy k, the flattened indices inside σ that must be shifted
        # for gauge copy k: indices = base_edge_idx + k * n_edges_total
        # store as an array of shape (gauge_dim, L)
        if self._L > 0:
            offsets = (
                jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total
            ).reshape(self.gauge_dim, 1)
            self._indices_by_gauge = offsets + base_edge_idx[None, :]
        else:
            self._indices_by_gauge = jnp.zeros((self.gauge_dim, 0), dtype=jnp.int32)

        # fixed number of connections: one per gauge copy in the decomposed sum
        self._K = self.gauge_dim

    @property
    def is_hermitian(self) -> bool:
        # in general, a holonomy shift operator is not Hermitian, so return False.
        return False

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        leading_shape = σ.shape[:-1]
        D = σ.shape[-1]

        # handle trivial loop: zero operator -> return padded K=1 with zero mels
        if self._L == 0:
            σp = jnp.expand_dims(σ, axis=-2)
            zeros = jnp.zeros(leading_shape, dtype=self.dtype)
            mels = jnp.expand_dims(zeros, axis=-1)
            return σp, mels

        # flatten leading dims -> (M, D)
        M = int(jnp.prod(jnp.array(leading_shape))) if leading_shape else 1
        sig = σ.reshape((M, D))

        # start from K copies of σ
        sigp = jnp.repeat(sig[:, None, :], repeats=self._K, axis=1)

        # for each gauge copy k, add +/-1 on the loop's edge components for that copy
        # _indices_by_gauge: (K, L), _steps: (L,)
        for k in range(self._K):
            idx_k = self._indices_by_gauge[k]
            cur = sigp[:, k, idx_k]
            deltas = jnp.broadcast_to(self._steps, (M, self._L))
            upd = _wrap_add(cur, deltas, self._mod_span, self._state_min).astype(
                sigp.dtype
            )
            sigp = sigp.at[:, k, idx_k].set(upd)

        # matrix elements: all 1 (complex)
        mels = jnp.ones((M, self._K), dtype=self.dtype)

        # restore leading shape
        new_shape = (*leading_shape, self._K, D)
        σp = sigp.reshape(new_shape)
        mels = mels.reshape((*leading_shape, self._K))

        return σp, mels


class MinimalLoopHolonomy(ComputationalOperator):
    r"""
    Minimal loop holonomy on a single gauge copy.

    Given a dressed minimal loop:
        [ ((u, v, key), {"type": ...}), ... ]

    this operator applies the total shift of all loop edges on one chosen gauge copy
    (with wrap-mod), emitting a single connected configuration and unit matrix element
    """

    def __init__(self, H, minimal_loop: List, *, copy: int = 0, adjoint: bool = False):
        super().__init__(H.hilbert)

        # invariants
        self._H = H
        self.K = int(H.core.gauge_dimensions)

        # width of one gauge copy
        self.n_edges_total = int(H.tiny_size)
        self.D = int(H.size)

        if not (0 <= copy < self.K):
            raise ValueError(f"`copy` must be in [0, {self.K-1}], got {copy}.")

        self._copy = int(copy)

        # wrap-mod parameters from allowed local_states
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )

        # parse loop: base edge indices + steps
        base_edge_idx = []
        steps = []
        for elem in minimal_loop:
            u, v, key = elem[0]
            t_info = elem[1]
            e_idx = H.graph.edge_to_index((u, v, key))
            base_edge_idx.append(e_idx)
            steps.append(_type_to_step(t_info.get("type", "creation")))

        self._L = len(base_edge_idx)
        self._steps = jnp.array(
            [-s for s in steps] if adjoint else steps, dtype=jnp.int32
        )

        if self._L > 0:
            base_edge_idx = jnp.array(base_edge_idx, dtype=jnp.int32)
            offset = jnp.int32(self._copy * self.n_edges_total)

            # indices of the loop edges inside the chosen gauge copy
            self._indices = base_edge_idx + offset
        else:
            self._indices = jnp.zeros((0,), dtype=jnp.int32)

        # single connection
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

        if self._L == 0:
            # zero op: return one diagonal with zero mel
            σp = jnp.expand_dims(σ, axis=2)
            mels = jnp.zeros((B, N, 1), dtype=self.dtype)
            return (σp[0], mels[0]) if single else (σp, mels)

        BN = B * N
        sig = σ.reshape(BN, D)

        # apply all loop steps on the chosen copy with wrap-mod
        comps = self._indices
        cur = sig[:, comps]
        deltas = jnp.broadcast_to(self._steps, (BN, self._L))
        upd = _wrap_add(cur, deltas, self._mod_span, self._state_min).astype(sig.dtype)
        sig_prime = sig.at[:, comps].set(upd)

        # emit σp and unit matrix element
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, 1, D)).copy()
        σp = σp.at[:, :, 0, :].set(sig_prime.reshape(B, N, D))
        mels = jnp.ones((B, N, 1), dtype=self.dtype)

        if single:
            return σp[0], mels[0]
        return σp, mels


class OneMinusHalfHolonomy(ComputationalOperator):
    r"""
    Operator:  I - 0.5 * ( h_{a,k} + h_{a,k}^\dagger )

    Acts on a single gauge copy `k`:
      - Connection 0: diagonal, σ (weight +1)
      - Connection 1: forward minimal-loop holonomy h_{a,k} (weight -1/2)
      - Connection 2: adjoint/backward minimal-loop holonomy h_{a,k}^\dagger (weight -1/2)

    If the dressed minimal loop is empty, this reduces to identity (one diagonal connection).
    """

    def __init__(self, H, minimal_loop: List, *, copy: int = 0):
        super().__init__(H.hilbert)

        # invariants
        self._H = H
        self.K = int(H.core.gauge_dimensions)

        # width per gauge copy
        self.n_edges_total = int(H.tiny_size)
        self.D = int(H.size)

        if not (0 <= copy < self.K):
            raise ValueError(f"`copy` must be in [0, {self.K-1}], got {copy}.")
        self._copy = int(copy)

        # wrap-mod parameters
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )

        # parse loop: base edge indices + steps
        base_edge_idx = []
        steps = []
        for elem in minimal_loop:
            u, v, key = elem[0]
            t_info = elem[1]
            e_idx = H.graph.edge_to_index((u, v, key))
            base_edge_idx.append(e_idx)
            steps.append(_type_to_step(t_info.get("type", "creation")))

        self._L = len(base_edge_idx)

        if self._L > 0:
            base_edge_idx = jnp.array(base_edge_idx, dtype=jnp.int32)

            # forward (+)
            self._steps = jnp.array(steps, dtype=jnp.int32)

            # adjoint (−)
            self._steps_adj = -self._steps
            offset = jnp.int32(self._copy * self.n_edges_total)
            self._indices = base_edge_idx + offset

            # diag + forward + adjoint
            self._nconn = 3
        else:
            # empty loop -> identity
            self._indices = jnp.zeros((0,), dtype=jnp.int32)
            self._steps = jnp.zeros((0,), dtype=jnp.int32)
            self._steps_adj = jnp.zeros((0,), dtype=jnp.int32)
            self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return True

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

        if self._L == 0:
            # identity: single diagonal connection with weight 1
            σp = jnp.expand_dims(σ, axis=2)
            mels = jnp.ones((B, N, 1), dtype=self.dtype)
            if single:
                return σp[0], mels[0]
            return σp, mels

        BN = B * N
        sig = σ.reshape(BN, D)

        # allocate 3 connections: diag, forward, adjoint
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, 3, D)).copy()

        #
        #
        # connection 0: diagonal (σ itself), already set by broadcast

        #
        #
        # connections 1 and 2: apply ± loop steps with wrap-mod on chosen copy

        comps = self._indices
        cur = sig[:, comps]

        deltas_fwd = jnp.broadcast_to(self._steps, (BN, self._L))
        deltas_adj = jnp.broadcast_to(self._steps_adj, (BN, self._L))

        upd_fwd = _wrap_add(cur, deltas_fwd, self._mod_span, self._state_min).astype(
            sig.dtype
        )
        upd_adj = _wrap_add(cur, deltas_adj, self._mod_span, self._state_min).astype(
            sig.dtype
        )

        sig_fwd = sig.at[:, comps].set(upd_fwd)
        sig_adj = sig.at[:, comps].set(upd_adj)

        σp = σp.at[:, :, 1, :].set(sig_fwd.reshape(B, N, D))
        σp = σp.at[:, :, 2, :].set(sig_adj.reshape(B, N, D))

        # matrix elements: [ +1, -1/2, -1/2 ] broadcast
        weights = jnp.array([1.0, -0.5, -0.5], dtype=self.dtype)
        mels = jnp.broadcast_to(weights, (B, N, 3))

        if single:
            return σp[0], mels[0]

        return σp, mels
