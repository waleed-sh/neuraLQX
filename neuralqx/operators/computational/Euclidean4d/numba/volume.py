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


"""
An implementation for the volume oeprator of the Euclidean WCL 4-d model using a
ComputationalOperator rather than a LocalOperator
"""

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator
from neuralqx.utils.misc.graph import get_true_edge_triplet


class VolumeOperator(ComputationalOperator):
    r"""
    Returns the (diagonal) volume operator at a single vertex ``v`` in the U(1)^3 model.

    The operator acts on the charge basis :math:`|\vec{m}\rangle` by summing, over all
    oriented edge triplets that meet at ``v``, the signed scalar triple products of the
    corresponding 3-component U(1) charge vectors, and then taking the absolute value
    and square root:

    .. math::

        \hat{V}_v \,|\vec{m}\rangle
        \;=\;
        \sqrt{\left|
            \sum_{(e_1,e_2,e_3)\ni v}
            \epsilon_v(e_1,e_2,e_3)\;
            \vec m_{e_1}\!\cdot\!\big(\vec m_{e_2}\times \vec m_{e_3}\big)
        \right|}\;|\vec{m}\rangle,

    where each edge charge vector is

    .. math::

        \vec m_{e} = \big( m_e^{(1)}, \, m_e^{(2)} , \, m_e^{(3)} \big),

    and :math:`\epsilon_v(e_1, e_2, e_3) \in \{+1,-1\}` is the orientation sign of the ordered
    triple at the vertex (right-hand rule with the graph's local orientation). The operator
    is Hermitian and non-negative, and it is diagonal in the charge basis.

    Implementation details
    ----------------------
    - The set of contributing triplets at ``v`` is taken from
      ``graph.connectivities[str(v)]['edges']``, their orientation signs are taken from
      ``graph.signs[str(v)]``.
    - Each triplet contributes
      :math:`\epsilon_v(e_1,e_2,e_3)\;\vec m_{e_1}\!\cdot\!(\vec m_{e_2}\times\vec m_{e_3})`.
      All contributions are summed before applying :math:`\sqrt{|\cdot|}`.
    - Edge orientations enter only through :math:`\epsilon_v(\cdot)`, the charge vectors
      themselves are used as-is.

    Notes:
        - Units: overall normalization/constant is 1 by convention
        - For non-planar embeddings, the triplets should be provided in the same ordering/convention
          as returned by the graph's ``connectivities`` (not by any planarized drawing).

    """

    def __init__(self, H, vertex):

        super().__init__(H.hilbert_netket)

        # local copy of the Hilbert object
        self._H = H

        # save the vertex too
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex

        # static layout

        # expect 3 in U(1)^3
        self.gauge_dim = int(H.gauge_dimensions)

        assert self.gauge_dim == 3, "This implementation assumes gauge_dim == 3."

        # total number of edges in each gauge level
        self.n_edges_total = int(H.size // self.gauge_dim)

        # triplets meeting at this vertex
        triplets = H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]

        # get the signs dict from the Graph object
        signs_dict = H.graph.signs[str(self.vertex)]

        # precompute component indices per triple and their signs
        comps1, comps2, comps3, signs = [], [], [], []

        # offsets for the edges through gauge levels
        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        for triple in triplets:
            # map edges to indices
            e1, e2, e3 = map(H.graph.edge_to_index, triple)

            # indices of the charge vectors
            c1 = (jnp.int32(e1) + offsets).astype(jnp.int32)
            c2 = (jnp.int32(e2) + offsets).astype(jnp.int32)
            c3 = (jnp.int32(e3) + offsets).astype(jnp.int32)

            # append
            comps1.append(c1)
            comps2.append(c2)
            comps3.append(c3)

            # also append signs while we're at it
            signs.append(int(signs_dict[get_true_edge_triplet(triple, H.graph)]))

        # stack of we have at least one contributing component
        if len(comps1) > 0:
            self._comps1 = jnp.stack(comps1, axis=0)
            self._comps2 = jnp.stack(comps2, axis=0)
            self._comps3 = jnp.stack(comps3, axis=0)
            self._signs = jnp.array(signs, dtype=jnp.int32)
            self._T = int(self._comps1.shape[0])
        else:
            # no contributing triplets at this vertex
            self._comps1 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._comps2 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._comps3 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._signs = jnp.zeros((0,), dtype=jnp.int32)
            self._T = 0

    @property
    def is_hermitian(self):
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):
        # support arbitrary leading batch dims (flatten them), with features on the last axis
        leading_shape = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading_shape))) if leading_shape else 1

        # diagonal operator: if there are no triplets, volume is zero
        if self._T == 0:
            σp = jnp.expand_dims(σ, axis=-2)
            zeros = jnp.zeros(leading_shape, dtype=self.dtype)
            mels = jnp.expand_dims(zeros, axis=-1)
            return σp, mels

        # flatten leading dims to (M, D)
        sig = σ.reshape((M, D))

        # gather three edge-vectors per triplet in one go, shapes here are (M, T, 3)
        m1 = jnp.take(sig, self._comps1, axis=-1)
        m2 = jnp.take(sig, self._comps2, axis=-1)
        m3 = jnp.take(sig, self._comps3, axis=-1)

        # cross and dot products take us from (M,T,3) to (M,T)
        cr = jnp.cross(m2, m3, axis=-1)
        dots = jnp.einsum("mtc,mtc->mt", m1, cr)

        # signed sum over triplets, abs and sqrt, this gives us shape (M,)
        Vr = jnp.sum(dots * self._signs[None, :], axis=1)
        vols_flat = jnp.sqrt(jnp.abs(Vr)).astype(self.dtype)

        # now restore leading shape
        vols = vols_flat.reshape(leading_shape)

        # one diagonal connection = itself
        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(vols, axis=-1)
        return σp, mels
