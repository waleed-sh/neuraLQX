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

from neuralqx.graph import HalfLadderGraph
from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


@jax.jit
def _spherical_volume_kernel(
    sigma: jnp.ndarray,
    idx_mu: jnp.ndarray,
    idx_km: jnp.ndarray,
    idx_kp: jnp.ndarray,
    shift_arr: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    D = sigma.shape[-1]
    sig_flat = sigma.reshape((-1, D))

    # μ(v)
    mu = jnp.take(sig_flat, idx_mu, axis=-1)
    if mu.ndim == 2:
        mu = jnp.squeeze(mu, axis=-1)

    # safe gather k_-(v)
    m_mask = idx_km >= 0
    m_idx = jnp.where(m_mask, idx_km, jnp.int32(0))
    k_m = jnp.take(sig_flat, m_idx, axis=-1)

    if k_m.ndim == 2:
        k_m = jnp.squeeze(k_m, axis=-1)
    k_m = jnp.where(m_mask, k_m, jnp.int32(0))

    # safe gather k_+(v)
    p_mask = idx_kp >= 0
    p_idx = jnp.where(p_mask, idx_kp, jnp.int32(0))
    k_p = jnp.take(sig_flat, p_idx, axis=-1)

    if k_p.ndim == 2:
        k_p = jnp.squeeze(k_p, axis=-1)

    k_p = jnp.where(p_mask, k_p, jnp.int32(0))

    K = (k_m + k_p).astype(jnp.float64)
    vol_vals = jnp.abs(mu).astype(jnp.float64) * jnp.sqrt(jnp.abs(K + shift_arr))

    vals = vol_vals.reshape(sigma.shape[:-1]).astype(jnp.float64)
    return sigma[..., None, :], vals[..., None]


@register_pytree_node_class
class SphericalVolumeJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, vertex: int, *, shift: float = 0.0):
        super().__init__(H.hilbert_netket)

        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"{type(self).__name__} requires `{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}`."
            )

        self.D = int(H.size)
        self.v = int(vertex)

        # Graph API contract guarantees one μ-edge and up to two k-edges at vertex v
        mu_list, k_list = H.graph.get_edges_at_k_vertex(self.v)

        if len(mu_list) != 1:
            raise RuntimeError(
                f"{type(self).__name__}: expected exactly one μ-edge at v={self.v}, got {mu_list}"
            )

        mu_here = mu_list[0]

        # (v-1, v, *) -> k_-(v) , (v, v+1, *) -> k_+(v)
        left_candidates = [e for e in k_list if e[1] == self.v]
        right_candidates = [e for e in k_list if e[0] == self.v]

        idx_mu = int(H.graph.edge_to_index(mu_here))
        idx_km = -1
        if len(left_candidates) == 1:
            idx_km = int(H.graph.edge_to_index(left_candidates[0]))
        idx_kp = -1
        if len(right_candidates) == 1:
            idx_kp = int(H.graph.edge_to_index(right_candidates[0]))

        # store as array-scalars to avoid static-arg recompiles
        self._idx_mu_arr = jnp.asarray(idx_mu, dtype=jnp.int32)
        self._idx_km_arr = jnp.asarray(idx_km, dtype=jnp.int32)
        self._idx_kp_arr = jnp.asarray(idx_kp, dtype=jnp.int32)
        self._shift_arr = jnp.asarray(shift, dtype=jnp.float64)

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (self._idx_mu_arr, self._idx_km_arr, self._idx_kp_arr, self._shift_arr)
        aux = {
            "hilbert": self.hilbert,
            "D": self.D,
            "v": self.v,
        }
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        idx_mu_arr, idx_km_arr, idx_kp_arr, shift_arr = leaves
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.D = int(aux["D"])
        obj.v = int(aux["v"])
        obj._idx_mu_arr = idx_mu_arr
        obj._idx_km_arr = idx_km_arr
        obj._idx_kp_arr = idx_kp_arr
        obj._shift_arr = shift_arr
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = jnp.asarray(x, dtype=jnp.int32)

        sigma_p, mels = _spherical_volume_kernel(
            sigma=x,
            idx_mu=self._idx_mu_arr,
            idx_km=self._idx_km_arr,
            idx_kp=self._idx_kp_arr,
            shift_arr=self._shift_arr,
        )

        return sigma_p, mels
