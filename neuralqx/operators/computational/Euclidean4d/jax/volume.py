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
from neuralqx.utils.misc.graph import get_true_edge_triplet


@jax.jit
def _volume_kernel(
    sigma: jnp.ndarray,
    comps1: jnp.ndarray,
    comps2: jnp.ndarray,
    comps3: jnp.ndarray,
    signs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Diagonal volume: returns (sigma[..., None, :], volume[..., None]).
    """

    # if no triplets: early diagonal zero (handle shape 0 cleanly)
    T = signs.shape[0]
    if T == 0:
        return sigma[..., None, :], jnp.zeros(
            sigma.shape[:-1] + (1,), dtype=jnp.float64
        )

    # collapse leading dims for a single gather
    D = sigma.shape[-1]
    sig_flat = sigma.reshape((-1, D))

    # gather 3-vectors per triplet at all copies
    m1 = jnp.take(sig_flat, comps1, axis=-1)
    m2 = jnp.take(sig_flat, comps2, axis=-1)
    m3 = jnp.take(sig_flat, comps3, axis=-1)

    # cross and dot -> (M, T)
    cr = jnp.cross(m2, m3, axis=-1)
    dots = jnp.einsum("mtc,mtc->mt", m1, cr)

    # signed sum over triplets, then sqrt|.|
    Vr = jnp.sum(dots * signs[None, :], axis=1)
    vol_M = jnp.sqrt(jnp.abs(Vr)).astype(jnp.float64)

    # restore leading shape
    vol = vol_M.reshape(sigma.shape[:-1])

    # one diagonal connection
    return sigma[..., None, :], vol[..., None]


@register_pytree_node_class
class VolumeOperatorJax(ComputationalJaxOperator):
    """
    JAX-compatible (diagonal) volume operator at a single vertex v for U(1)^3.
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

    def __init__(self, H, vertex):

        super().__init__(H.hilbert_netket)

        # small static metadata
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex
        self.gauge_dim = int(H.gauge_dimensions)
        assert (
            self.gauge_dim == 3
        ), f"{type(self).__name__} assumes gauge_dim == 3 (U(1)^3)."

        self.n_edges_total = int(H.size // self.gauge_dim)

        # graph triplets and signs
        triplets = H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = H.graph.signs[str(self.vertex)]

        # build gather indices for all 3 copies
        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total

        comps1, comps2, comps3, signs = [], [], [], []
        for triple in triplets:
            e1, e2, e3 = map(H.graph.edge_to_index, triple)

            c1 = (jnp.int64(e1) + offsets).astype(jnp.int64)
            c2 = (jnp.int64(e2) + offsets).astype(jnp.int64)
            c3 = (jnp.int64(e3) + offsets).astype(jnp.int64)

            comps1.append(c1)
            comps2.append(c2)
            comps3.append(c3)

            signs.append(int(signs_dict[get_true_edge_triplet(triple, H.graph)]))

        if len(comps1) > 0:
            self._comps1 = jnp.stack(comps1, axis=0).astype(jnp.int64)
            self._comps2 = jnp.stack(comps2, axis=0).astype(jnp.int64)
            self._comps3 = jnp.stack(comps3, axis=0).astype(jnp.int64)
            self._signs = jnp.asarray(signs, dtype=jnp.int64)
        else:
            self._comps1 = jnp.zeros((0, 3), dtype=jnp.int64)
            self._comps2 = jnp.zeros((0, 3), dtype=jnp.int64)
            self._comps3 = jnp.zeros((0, 3), dtype=jnp.int64)
            self._signs = jnp.zeros((0,), dtype=jnp.int64)

        self._T = int(self._comps1.shape[0])

    #
    #
    #   pytree support

    def tree_flatten(self):
        # arrays as leaves, small metadata as struct
        leaves = (self._comps1, self._comps2, self._comps3, self._signs)
        struct = {
            "hilbert": self.hilbert,
            "vertex": self.vertex,
            "gauge_dim": self.gauge_dim,
            "n_edges_total": self.n_edges_total,
        }
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)

        # re-run base init for hilbert only
        super(cls, obj).__init__(struct["hilbert"])

        # restore static fields
        obj.vertex = int(struct["vertex"])
        obj.gauge_dim = int(struct["gauge_dim"])
        obj.n_edges_total = int(struct["n_edges_total"])

        # restore leaves
        obj._comps1, obj._comps2, obj._comps3, obj._signs = leaves
        obj._T = int(obj._comps1.shape[0])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = jnp.asarray(x, dtype=jnp.int64)
        # early exit outside jit for T == 0 to avoid useless tracing/launches

        if self._T == 0:
            return x[..., None, :], jnp.zeros(x.shape[:-1] + (1,), dtype=jnp.float64)

        σp, mels = _volume_kernel(
            sigma=x,
            comps1=self._comps1,
            comps2=self._comps2,
            comps3=self._comps3,
            signs=self._signs,
        )
        return σp, mels
