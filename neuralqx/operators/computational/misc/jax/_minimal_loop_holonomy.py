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
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


def _type_to_step(t) -> int:
    if isinstance(t, str):
        s = t.strip().lower()
        if s in ("creation", "+", "up", "plus", "raise", "raising"):
            return +1
        if s in ("annihilation", "-", "down", "minus", "lower", "lowering"):
            return -1
    try:
        return +1 if int(t) > 0 else -1
    except Exception:
        return +1


def _wrap_add_step(
    vals_i32: jnp.ndarray,
    delta_i32: jnp.ndarray,
    state_min: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_step: jnp.ndarray,
) -> jnp.ndarray:
    idx_vals = (vals_i32 - state_min) // state_step
    idx_delta = delta_i32 // state_step
    idx_sum = (idx_vals + idx_delta) % mod_span
    return idx_sum * state_step + state_min


@jax.jit
def _nd_minloop_kernel(
    sigma: jnp.ndarray,
    indices_by_gauge: jnp.ndarray,
    steps: jnp.ndarray,
    state_min: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_step: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N

    sig = sigma.reshape(BN, D).astype(jnp.int64)

    K, L = indices_by_gauge.shape
    if L == 0:
        # zero loop -> zero operator, return one diagonal zero connection
        zero = jnp.zeros((B, N, 1), dtype=jnp.float64)
        return sigma[:, :, None, :], zero

    # broadcast base to (K, BN, D) so we can vmap per copy
    base = jnp.broadcast_to(sig[None, ...], (K, BN, D))

    def apply_one(k, sbase):
        idx = indices_by_gauge[k]
        cur = sbase[:, idx]
        deltas = jnp.broadcast_to(steps[None, :], (BN, L))
        upd = _wrap_add_step(cur, deltas, state_min, mod_span, state_step).astype(
            sbase.dtype
        )
        return sbase.at[:, idx].set(upd)

    ks = jnp.arange(K, dtype=jnp.int64)
    sig_all = jax.vmap(apply_one, in_axes=(0, 0), out_axes=0)(ks, base)

    # (K, BN, D) -> (B, N, K, D)
    sigma_p = sig_all.transpose(1, 0, 2).reshape(B, N, K, D)
    mels = jnp.ones((B, N, K), dtype=jnp.float64)
    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class NDMinimalLoopHolonomyJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return self.K

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, minimal_loop: List, *, adjoint: int = 0):
        super().__init__(H.hilbert_netket)

        self.K = int(H.hilbert.gauge_dimensions)
        self.D = int(H.size)
        self.n_edges_total = int(H.tiny_size)

        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)

        # parse loop: base indices and steps
        base_edge_idx = []
        steps = []
        for elem in minimal_loop:
            u, v, key = elem[0]
            t_info = elem[1]
            e_idx = int(H.graph.edge_to_index((u, v, key)))
            base_edge_idx.append(e_idx)
            steps.append(_type_to_step(t_info.get("type", "creation")))

        self._L = len(base_edge_idx)
        if self._L > 0:
            base_edge_idx = jnp.asarray(base_edge_idx, dtype=jnp.int64)
            steps = jnp.asarray(steps, dtype=jnp.int64)
            if int(adjoint) == 1:
                steps = -steps

            # indices per copy: base + k * n_edges_total
            offsets = (jnp.arange(self.K, dtype=jnp.int64) * self.n_edges_total)[
                :, None
            ]
            self._indices_by_gauge = offsets + base_edge_idx[None, :]
            self._steps = steps
        else:
            self._indices_by_gauge = jnp.zeros((self.K, 0), dtype=jnp.int64)
            self._steps = jnp.zeros((0,), dtype=jnp.int64)

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (self._indices_by_gauge, self._steps)
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            K=self.K,
            n_edges_total=self.n_edges_total,
            state_min=self._state_min,
            mod_span=self._mod_span,
            state_step=self._state_step,
            L=self._L,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        indices_by_gauge, steps = leaves
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj.K = int(struct["K"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._state_min = int(struct["state_min"])
        obj._mod_span = int(struct["mod_span"])
        obj._state_step = int(struct["state_step"])
        obj._L = int(struct["L"])
        obj._indices_by_gauge = indices_by_gauge
        obj._steps = steps
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)
        return _nd_minloop_kernel(
            sigma=x,
            indices_by_gauge=self._indices_by_gauge,
            steps=self._steps,
            state_min=jnp.asarray([self._state_min], dtype=jnp.int64),
            mod_span=jnp.asarray([self._mod_span], dtype=jnp.int64),
            state_step=jnp.asarray([self._state_step], dtype=jnp.int64),
        )


@jax.jit
def _minloop_kernel(
    sigma: jnp.ndarray,
    indices: jnp.ndarray,
    steps: jnp.ndarray,
    state_min: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_step: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N

    if indices.shape[0] == 0:
        # zero loop, return diagonal zero
        return sigma[:, :, None, :], jnp.ones((B, N, 1), dtype=jnp.float64)

    sig = sigma.reshape(BN, D).astype(jnp.int64)

    cur = sig[:, indices]
    deltas = jnp.broadcast_to(steps, (BN, indices.size))
    upd = _wrap_add_step(cur, deltas, state_min, mod_span, state_step).astype(sig.dtype)

    sigp = sig.at[:, indices].set(upd)
    sigma_p = sigp.reshape(B, N, D)[:, :, None, :]
    mels = jnp.ones((B, N, 1), dtype=jnp.float64)
    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class MinimalLoopHolonomyJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, minimal_loop: List, *, copy: int = 0, adjoint: bool = False):
        super().__init__(H.hilbert_netket)

        self.K = int(H.hilbert.gauge_dimensions)
        self.D = int(H.size)
        self.n_edges_total = int(H.tiny_size)

        if not (0 <= int(copy) < self.K):
            raise ValueError(f"`copy` must be in [0, {self.K-1}], got {copy}.")

        self._copy = int(copy)

        # lattice parameters
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)

        # parse loop
        base_edge_idx = []
        steps = []
        for elem in minimal_loop:
            u, v, key = elem[0]
            t_info = elem[1]
            e_idx = int(H.graph.edge_to_index((u, v, key)))
            base_edge_idx.append(e_idx)
            steps.append(_type_to_step(t_info.get("type", "creation")))
        if adjoint:
            steps = [-s for s in steps]

        self._L = len(base_edge_idx)
        if self._L > 0:
            base_edge_idx = jnp.asarray(base_edge_idx, dtype=jnp.int64)
            offset = jnp.int64(self._copy * self.n_edges_total)
            self._indices = base_edge_idx + offset
            self._steps = jnp.asarray(steps, dtype=jnp.int64)
        else:
            self._indices = jnp.zeros((0,), dtype=jnp.int64)
            self._steps = jnp.zeros((0,), dtype=jnp.int64)

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (self._indices, self._steps)
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            K=self.K,
            n_edges_total=self.n_edges_total,
            copy=self._copy,
            state_min=self._state_min,
            mod_span=self._mod_span,
            state_step=self._state_step,
            L=self._L,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        indices, steps = leaves
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj.K = int(struct["K"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._copy = int(struct["copy"])
        obj._state_min = int(struct["state_min"])
        obj._mod_span = int(struct["mod_span"])
        obj._state_step = int(struct["state_step"])
        obj._L = int(struct["L"])
        obj._indices = indices
        obj._steps = steps
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):

        x = jnp.asarray(x, dtype=jnp.int64)
        return _minloop_kernel(
            sigma=x,
            indices=self._indices,
            steps=self._steps,
            state_min=jnp.asarray([self._state_min], dtype=jnp.int64),
            mod_span=jnp.asarray([self._mod_span], dtype=jnp.int64),
            state_step=jnp.asarray([self._state_step], dtype=jnp.int64),
        )


@jax.jit
def _one_minus_half_kernel(
    sigma: jnp.ndarray,
    indices: jnp.ndarray,
    steps: jnp.ndarray,
    state_min: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_step: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N

    if indices.shape[0] == 0:
        # empty loop -> identity only
        return sigma[:, :, None, :], jnp.ones((B, N, 1), dtype=jnp.float64)

    sig = sigma.reshape(BN, D).astype(jnp.int64)

    # diagonal (σ)
    sigma_out = jnp.broadcast_to(sigma[:, :, None, :], (B, N, 3, D)).copy()

    cur = sig[:, indices]

    deltas_fwd = jnp.broadcast_to(steps, (BN, indices.size))
    deltas_adj = -deltas_fwd

    upd_fwd = _wrap_add_step(cur, deltas_fwd, state_min, mod_span, state_step).astype(
        sig.dtype
    )
    upd_adj = _wrap_add_step(cur, deltas_adj, state_min, mod_span, state_step).astype(
        sig.dtype
    )

    sig_fwd = sig.at[:, indices].set(upd_fwd)
    sig_adj = sig.at[:, indices].set(upd_adj)

    sigma_out = sigma_out.at[:, :, 1, :].set(sig_fwd.reshape(B, N, D))
    sigma_out = sigma_out.at[:, :, 2, :].set(sig_adj.reshape(B, N, D))

    weights = jnp.asarray([1.0, -0.5, -0.5], dtype=jnp.float64)
    mels = jnp.broadcast_to(weights, (B, N, 3))

    return (sigma_out[0], mels[0]) if single else (sigma_out, mels)


@register_pytree_node_class
class OneMinusHalfHolonomyJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1 if self._L == 0 else 3

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, minimal_loop: List, *, copy: int = 0):
        super().__init__(H.hilbert_netket)

        self.K = int(H.hilbert.gauge_dimensions)
        self.D = int(H.size)
        self.n_edges_total = int(H.tiny_size)

        if not (0 <= int(copy) < self.K):
            raise ValueError(f"`copy` must be in [0, {self.K-1}], got {copy}.")

        self._copy = int(copy)

        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)

        base_edge_idx = []
        steps = []
        for elem in minimal_loop:
            u, v, key = elem[0]
            t_info = elem[1]
            e_idx = int(H.graph.edge_to_index((u, v, key)))
            base_edge_idx.append(e_idx)
            steps.append(_type_to_step(t_info.get("type", "creation")))

        self._L = len(base_edge_idx)
        if self._L > 0:
            base_edge_idx = jnp.asarray(base_edge_idx, dtype=jnp.int64)
            offset = jnp.int64(self._copy * self.n_edges_total)
            self._indices = base_edge_idx + offset
            self._steps = jnp.asarray(steps, dtype=jnp.int64)
        else:
            self._indices = jnp.zeros((0,), dtype=jnp.int64)
            self._steps = jnp.zeros((0,), dtype=jnp.int64)

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (self._indices, self._steps)
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            K=self.K,
            n_edges_total=self.n_edges_total,
            copy=self._copy,
            state_min=self._state_min,
            mod_span=self._mod_span,
            state_step=self._state_step,
            L=self._L,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        indices, steps = leaves
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj.K = int(struct["K"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._copy = int(struct["copy"])
        obj._state_min = int(struct["state_min"])
        obj._mod_span = int(struct["mod_span"])
        obj._state_step = int(struct["state_step"])
        obj._L = int(struct["L"])
        obj._indices = indices
        obj._steps = steps
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)

        # fast host-side short-circuit for empty loop (avoid compiling extra branches)

        if self._L == 0:
            if x.ndim == 2:
                x = x[None, :, :]
                out = x[:, :, None, :]
                m = jnp.ones((1, x.shape[1], 1), dtype=jnp.float64)
                return out[0], m[0]
            return x[:, :, None, :], jnp.ones(x.shape[:2] + (1,), dtype=jnp.float64)

        return _one_minus_half_kernel(
            sigma=x,
            indices=self._indices,
            steps=self._steps,
            state_min=jnp.asarray([self._state_min], dtype=jnp.int64),
            mod_span=jnp.asarray([self._mod_span], dtype=jnp.int64),
            state_step=jnp.asarray([self._state_step], dtype=jnp.int64),
        )
