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


@jax.jit
def _creation_kernel(
    sigma: jnp.ndarray,
    site: jnp.ndarray,
    delta: jnp.ndarray,
    state_min: jnp.ndarray,
    state_max: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N
    sig = sigma.reshape(BN, D)

    s = jnp.asarray(site, dtype=jnp.int32)
    cur = jax.lax.dynamic_slice_in_dim(sig, s, 1, axis=1)  # (BN, 1)
    new = cur + delta

    lo = jnp.asarray(state_min)
    hi = jnp.asarray(state_max)
    mask = (new >= lo) & (new <= hi)

    # flatten to (BN,)
    cur_flat = cur.reshape(BN)
    new_flat = new.reshape(BN)
    mask_flat = mask.reshape(BN)

    # select new or old
    updated = jnp.where(mask_flat, new_flat, cur_flat).reshape(BN, 1)

    # use dynamic_update_slice_in_dim instead of .at
    sigp = jax.lax.dynamic_update_slice_in_dim(sig, updated, s, axis=1)

    sigma_p = sigp.reshape(B, N, D)
    sigma_p = sigma_p[:, :, None, :]
    mels = mask_flat.reshape(B, N, 1).astype(jnp.float32)
    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class QRCreationJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float32

    def __init__(self, H, *, site: int, n: int = 1):
        super().__init__(H.hilbert)

        self._H = H
        self.D = int(H.size)

        if not (isinstance(site, int) and 0 <= site < self.D):
            raise ValueError(f"`site` must be in [0, {self.D-1}], got {site} instead.")

        self._site = int(site)
        self._n = int(n)

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

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        leaves = tuple()
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            site=self._site,
            n=self._n,
            state_min=self._state_min,
            state_max=self._state_max,
            state_step=self._state_step,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj._site = int(struct["site"])
        obj._n = int(struct["n"])
        obj._state_min = int(struct["state_min"])
        obj._state_max = int(struct["state_max"])
        obj._state_step = int(struct["state_step"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)
        delta = jnp.asarray(self._n * self._state_step, dtype=jnp.int64)

        sigma_p, mels = _creation_kernel(
            sigma=x,
            site=jnp.asarray(self._site, dtype=jnp.int64),
            delta=delta,
            state_min=jnp.asarray(self._state_min, dtype=jnp.int64),
            state_max=jnp.asarray(self._state_max, dtype=jnp.int64),
        )
        return sigma_p, mels
