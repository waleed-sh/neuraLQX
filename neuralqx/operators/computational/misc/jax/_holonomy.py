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


from typing import Tuple
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


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
def _u1_holonomy_kernel(
    sigma: jnp.ndarray,
    comp_index: jnp.ndarray,
    delta: jnp.ndarray,
    state_min: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_step: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Off-diagonal single connection: shift one global component by ±step with wrap.
    """

    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape

    BN = B * N
    sig = sigma.reshape(BN, D).astype(jnp.int64)

    # gather, wrap-add, scatter back
    idx = comp_index[0]

    cur = jax.lax.dynamic_slice_in_dim(
        sig,
        start_index=idx,
        slice_size=1,
        axis=1,
    )

    upd = _u1_wrap_add(cur, delta, state_min, mod_span, state_step)

    sigp = jax.lax.dynamic_update_slice_in_dim(
        sig,
        upd,
        start_index=idx,
        axis=1,
    )

    sigma_p = sigp.reshape(B, N, D)

    # pack as one connection
    sigma_p = sigma_p[:, :, None, :]
    mels = jnp.ones((B, N, 1), dtype=jnp.float64)

    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


def _u1_wrap_add(cur, delta, state_min, mod_span, state_step):
    # helper so the JAXPR stays tidy
    return _wrap_add_step(cur, delta, state_min, mod_span, state_step).astype(cur.dtype)


@register_pytree_node_class
class U1HolonomyJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, edge_index: int, *, adjoint: bool = False):
        super().__init__(H.hilbert_netket)

        # require U(1)
        if int(H.hilbert.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) (gauge_dimensions = 1)."
            )

        self.D = int(H.size)
        if not (isinstance(edge_index, int) and 0 <= edge_index < self.D):
            raise ValueError(
                f"`edge_index` must be in [0, {self.D-1}], got {edge_index}."
            )

        # lattice parameters from allowed states
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)

        self._edge_index = int(edge_index)

        # lowering if adjoint, else raising
        self._direction = -1 if adjoint else +1

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = tuple()

        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            edge_index=self._edge_index,
            direction=self._direction,
            state_min=self._state_min,
            mod_span=self._mod_span,
            state_step=self._state_step,
        )

        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj._edge_index = int(struct["edge_index"])
        obj._direction = int(struct["direction"])
        obj._state_min = int(struct["state_min"])
        obj._mod_span = int(struct["mod_span"])
        obj._state_step = int(struct["state_step"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)

        sigma_p, mels = _u1_holonomy_kernel(
            sigma=x,
            comp_index=jnp.asarray([self._edge_index], dtype=jnp.int64),
            delta=jnp.asarray([self._direction * self._state_step], dtype=jnp.int64),
            state_min=jnp.asarray([self._state_min], dtype=jnp.int64),
            mod_span=jnp.asarray([self._mod_span], dtype=jnp.int64),
            state_step=jnp.asarray([self._state_step], dtype=jnp.int64),
        )
        return sigma_p, mels


@jax.jit
def _nd_u1_holonomy_kernel(
    sigma: jnp.ndarray,
    comp_indices: jnp.ndarray,
    delta: jnp.ndarray,
    state_min: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_step: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N

    sig0 = sigma.reshape(BN, D).astype(jnp.int64)
    K = comp_indices.shape[0]

    # prepare a (K, BN, D) stack by updating a single column per k using dynamic slices
    base = jnp.broadcast_to(sig0, (K, BN, D)).copy()

    def upd_one(k):
        idx = comp_indices[k]
        cur = jax.lax.dynamic_slice_in_dim(base[k], idx, slice_size=1, axis=1)
        upd = _wrap_add_step(
            cur, jnp.broadcast_to(delta, (BN, 1)), state_min, mod_span, state_step
        ).astype(base.dtype)
        return jax.lax.dynamic_update_slice_in_dim(base[k], upd, idx, axis=1)

    ks = jnp.arange(K, dtype=jnp.int64)
    # only k is batched, the base array is captured from closure
    sig_all = jax.vmap(lambda k: upd_one(k), in_axes=0, out_axes=0)(ks)

    # (K, BN, D) -> (B, N, K, D)
    sigma_p = sig_all.transpose(1, 0, 2).reshape(B, N, K, D)
    mels = jnp.ones((B, N, K), dtype=jnp.float64)
    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class NDU1HolonomyJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return self.K

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, edge_index: int, *, adjoint: bool = False):
        super().__init__(H.hilbert_netket)

        self.K = int(H.hilbert.gauge_dimensions)
        if self.K < 1:
            raise ValueError("`gauge_dimensions` must be ≥ 1.")

        self.D = int(H.size)
        self.n_edges_total = int(H.tiny_size)

        if not (isinstance(edge_index, int) and 0 <= edge_index < self.n_edges_total):
            raise ValueError(
                f"`edge_index` must be in [0, {self.n_edges_total-1}] within a single copy."
            )

        base = int(edge_index)
        copy_offsets = jnp.arange(self.K, dtype=jnp.int64) * self.n_edges_total
        self._comp_indices = (base + copy_offsets).astype(jnp.int64)

        # lattice params
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)

        self._direction = -1 if adjoint else +1

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (self._comp_indices,)
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            K=self.K,
            n_edges_total=self.n_edges_total,
            direction=self._direction,
            state_min=self._state_min,
            mod_span=self._mod_span,
            state_step=self._state_step,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        (comp_indices,) = leaves
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj.K = int(struct["K"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._direction = int(struct["direction"])
        obj._state_min = int(struct["state_min"])
        obj._mod_span = int(struct["mod_span"])
        obj._state_step = int(struct["state_step"])
        obj._comp_indices = comp_indices
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)

        sigma_p, mels = _nd_u1_holonomy_kernel(
            sigma=x,
            comp_indices=self._comp_indices,
            delta=jnp.asarray([self._direction * self._state_step], dtype=jnp.int64),
            state_min=jnp.asarray([self._state_min], dtype=jnp.int64),
            mod_span=jnp.asarray([self._mod_span], dtype=jnp.int64),
            state_step=jnp.asarray([self._state_step], dtype=jnp.int64),
        )

        return sigma_p, mels
