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

from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


@jax.jit
def _area_kernel(
    sigma: jnp.ndarray,
    comps_flat: jnp.ndarray,
    squared_flag: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Diagonal area operator:
      sigma_out = sigma[..., None, :]
      mels      = ( sum_{e in surface} ||m_e||_2 or ||m_e||_2^2 )[..., None]
    Args are all arrays to avoid static-arg issues under jit.
    Assumes U(1)^3 (gauge_dim = 3).
    """

    # flatten leading dims for a single gather
    D = sigma.shape[-1]
    sig_flat = sigma.reshape((-1, D))

    # if no components to gather, produce zeros with correct shape
    if comps_flat.shape[0] == 0:
        zeros = jnp.zeros(sigma.shape[:-1] + (1,), dtype=jnp.float64)
        return sigma[..., None, :], zeros

    # gather all requested components in a single take -> (M, E*3)
    gathered = jnp.take(sig_flat, comps_flat, axis=-1)

    # U(1)^3: reshape to (M, E, 3)
    gdim = 3
    E = comps_flat.shape[0] // gdim
    gathered = gathered.reshape((-1, E, gdim))

    # per-edge squared norms -> (M, E)
    norms_sq = jnp.sum(gathered * gathered, axis=-1)

    # either sqrt or keep squared based on flag (array scalar)
    # Note: squared_flag is a 0-dim JAX bool array to avoid static args
    per_edge = jnp.where(squared_flag, norms_sq, jnp.sqrt(norms_sq))

    # sum over edges -> (M,)
    areas_flat = jnp.sum(per_edge, axis=-1)

    # restore leading shape and cast to float64
    areas = areas_flat.reshape(sigma.shape[:-1]).astype(jnp.float64)

    # diagonal operator: one connection = itself
    return sigma[..., None, :], areas[..., None]


@register_pytree_node_class
class AreaOperatorJax(ComputationalJaxOperator):
    """
    JAX-compatible (diagonal) area operator for a given surface S (list of edges).
    For σ[..., D] with D = n_edges_total * 3 (U(1)^3):
      σp   = σ[..., None, :]
      mels = (Σ_e ||m_e||_2) or (Σ_e ||m_e||_2^2)  depending on `squared`.
    """

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H: AbstractHilbertInterface, edges, *, squared: bool = False):

        super().__init__(H.hilbert_netket)

        self.gauge_dim = int(H.gauge_dimensions)

        # total edges per copy
        self.n_edges_total = int(H.size // self.gauge_dim)

        # map edges -> indices and build flat component indices for all three copies
        edges_idx = jnp.asarray(
            [H.graph.edge_to_index(e) for e in edges], dtype=jnp.int64
        )
        self._E = int(edges_idx.size)

        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total
        if self._E > 0:
            comps = edges_idx[:, None] + offsets[None, :]
            self._comps_flat = comps.reshape(-1)
        else:
            self._comps_flat = jnp.zeros((0,), dtype=jnp.int64)

        # store squared flag as Python bool, we will convert to a JAX bool array at call
        self._squared = bool(squared)

    #
    #
    #   pytree support
    def tree_flatten(self):
        # arrays as leaves, metadata in the struct
        leaves = (self._comps_flat,)
        struct = {
            "hilbert": self.hilbert,
            "gauge_dim": self.gauge_dim,
            "n_edges_total": self.n_edges_total,
            "E": self._E,
            "squared": self._squared,
        }
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)

        # re-init base with hilbert only
        super(cls, obj).__init__(struct["hilbert"])

        # restore metadata
        obj.gauge_dim = int(struct["gauge_dim"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._E = int(struct["E"])
        obj._squared = bool(struct["squared"])

        # restore leaves
        (obj._comps_flat,) = leaves
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = jnp.asarray(x, dtype=jnp.int64)

        # fast early-exit (avoid compiling kernel at all) when no edges were provided
        if self._E == 0:
            return x[..., None, :], jnp.zeros(x.shape[:-1] + (1,), dtype=jnp.float64)

        σp, mels = _area_kernel(
            sigma=x,
            comps_flat=self._comps_flat,
            squared_flag=jnp.asarray(self._squared, dtype=bool),
        )

        return σp, mels
