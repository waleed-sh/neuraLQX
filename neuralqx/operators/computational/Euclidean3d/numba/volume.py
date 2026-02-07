#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""
An implementation for the (2+1) "volume" (= area) operator of the Eucli. rather than a LocalOperator.
"""

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


class VolumeOperator(ComputationalOperator):
    r"""
    Returns the (diagonal) 2+1D "volume" operator at a single vertex ``v`` in the U(1)^3 model.

    In 2+1 dimensions, the spatial slice is 2D, so the geometric operator associated with a
    small region around a vertex is an **area** operator. In the abelian U(1)^3 weak-coupling
    model, the operator is diagonal in the charge basis and is built from oriented **pairs**
    of edges meeting at the vertex.

    On a charge basis state :math:`|\vec{m}\rangle`, we define

    .. math::

        \hat{A}_v \,|\vec{m}\rangle
        \;=\;
        \left\|
            \sum_{(e_1,e_2)\ni v}
            \epsilon_v(e_1,e_2)\;
            \vec m_{e_1}\times \vec m_{e_2}
        \right\|\;|\vec{m}\rangle,

    where each edge charge vector is

    .. math::

        \vec m_e = (m_e^{(1)}, m_e^{(2)}, m_e^{(3)}),

    and :math:`\epsilon_v(e_1,e_2)\in\{+1,-1\}` is the oriented sign of the ordered pair
    at the vertex (2D right-hand convention on the embedded graph).

    Notes:
        - Units / overall normalization: set to 1 by convention.
        - This is the natural 2D analog of the 4D vertex volume, replacing the signed scalar
          triple product with a signed sum of cross products and taking the norm.
        - Requires the graph to provide oriented edge pairs and their signs at the vertex.

    Implementation details
    ----------------------
    - The set of contributing pairs at ``v`` is taken from
      ``graph.connectivities[str(v)]['edges']`` (expected to be pairs for the 3D model).
    - Their orientation signs are taken from ``graph.signs[str(v)]`` using
      ``get_true_edge_pair`` to canonicalize keys.
    - Each pair contributes
      :math:`\epsilon_v(e_1,e_2)\,(\vec m_{e_1}\times \vec m_{e_2})` (a 3-vector).
      All contributions are summed into a single 3-vector, and then its Euclidean norm
      is taken. The result is non-negative and diagonal.

    """

    def __init__(self, H, vertex):

        super().__init__(H.hilbert)

        # local copy of the Hilbert object
        self._H = H

        # save vertex
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex

        # expect 3 in U(1)^3
        self.gauge_dim = int(H.gauge_dimensions)
        assert self.gauge_dim == 3, "This implementation assumes gauge_dim == 3."

        # total number of edges in each gauge level
        self.n_edges_total = int(H.size // self.gauge_dim)

        # pairs meeting at this vertex (for the 3-d model this should be a list of 2-tuples)
        pairs = H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]

        # get the signs dict from the Graph object
        signs_dict = H.graph.signs[str(self.vertex)]

        # precompute component indices per pair and their signs
        comps1, comps2, signs = [], [], []

        # offsets for the edges through gauge levels
        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        for pair in pairs:
            # map edges to indices
            e1, e2 = map(H.graph.edge_to_index, pair)

            # indices of the charge vectors
            c1 = (jnp.int32(e1) + offsets).astype(jnp.int32)
            c2 = (jnp.int32(e2) + offsets).astype(jnp.int32)

            comps1.append(c1)
            comps2.append(c2)

            # orientation sign for this ordered pair
            signs.append(int(signs_dict[str(pair)]))

        # stack if we have at least one contributing pair
        if len(comps1) > 0:
            self._comps1 = jnp.stack(comps1, axis=0)
            self._comps2 = jnp.stack(comps2, axis=0)
            self._signs = jnp.array(signs, dtype=jnp.int32)
            self._P = int(self._comps1.shape[0])
        else:
            # no contributing pairs at this vertex
            self._comps1 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._comps2 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._signs = jnp.zeros((0,), dtype=jnp.int32)
            self._P = 0

    @property
    def is_hermitian(self):
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):
        leading_shape = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading_shape))) if leading_shape else 1

        if self._P == 0:
            σp = jnp.expand_dims(σ, axis=-2)
            zeros = jnp.zeros(leading_shape, dtype=self.dtype)
            mels = jnp.expand_dims(zeros, axis=-1)
            return σp, mels

        # flatten leading dims to (M, D)
        sig = σ.reshape((M, D))

        # gather two edge-vectors per pair, shapes are (M, P, 3)
        m1 = jnp.take(sig, self._comps1, axis=-1)
        m2 = jnp.take(sig, self._comps2, axis=-1)

        # cross products -> (M, P, 3)
        cr = jnp.cross(m1, m2, axis=-1)

        # signed sum over pairs -> (M, 3)
        vec = jnp.sum(cr * self._signs[None, :, None], axis=1)

        # Euclidean norm -> (M,)
        areas_flat = jnp.linalg.norm(vec, axis=-1).astype(self.dtype)

        # restore leading shape
        areas = areas_flat.reshape(leading_shape)

        # one diagonal connection = itself
        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(areas, axis=-1)
        return σp, mels


class VolumeOperatorSquared(VolumeOperator):
    r"""
    Squared (diagonal) 2+1 "volume" operator at a single vertex ``v`` in the U(1)^3 model.

    This class implements :math:`\hat{V}_v^2 = (\hat{V}_v)^2` as a diagonal observable derived
    from :class:`~neuralqx.operators.computational.Euclidean3d.numba.VolumeOperator`.

    On a charge basis state :math:`|\vec m\rangle`, where the parent operator has the eigenvalue

    .. math::

        V_v(\vec m)
        \;=\;
        \left\|
            \sum_{(e_1,e_2)\ni v}
            \epsilon_v(e_1,e_2)\;
            \vec m_{e_1}\times \vec m_{e_2}
        \right\|
        \;\ge 0,

    this operator acts diagonally as

    .. math::

        \hat{V}_v^2 \,|\vec m\rangle \;=\; \big(V_v(\vec m)\big)^2 \,|\vec m\rangle.

    Notes:
        - Diagonal in the charge basis and therefore Hermitian.
        - Implemented by squaring the diagonal matrix elements returned by the parent kernel.

    """

    def _get_conn_padded_kernel(self, σ):
        σp, mels = super()._get_conn_padded_kernel(σ)
        return σp, mels * mels


class VolumeOperatorSqrt(VolumeOperator):
    r"""
    Square-root (diagonal) 2+1 "volume" operator at a single vertex ``v`` in the U(1)^3 model.

    This class implements the operator function :math:`\sqrt{\hat{V}_v}` derived from
    :class:`~neuralqx.operators.computational.Euclidean3d.numba.VolumeOperator`.

    On a charge basis state :math:`|\vec m\rangle`, where the parent operator has the eigenvalue

    .. math::

        V_v(\vec m)
        \;=\;
        \left\|
            \sum_{(e_1,e_2)\ni v}
            \epsilon_v(e_1,e_2)\;
            \vec m_{e_1}\times \vec m_{e_2}
        \right\|
        \;\ge 0,

    this operator acts diagonally as

    .. math::

        \sqrt{\hat{V}_v}\,|\vec m\rangle
        \;=\;
        \sqrt{V_v(\vec m)}\,|\vec m\rangle.

    Notes:
        - In this 2+1D construction, :math:`V_v(\vec m)` is a norm and thus non-negative.
        - Diagonal in the charge basis and therefore Hermitian.
        - Implemented by applying ``jnp.sqrt`` to the diagonal matrix elements returned by the
          parent kernel.

    """

    def _get_conn_padded_kernel(self, σ):
        σp, mels = super()._get_conn_padded_kernel(σ)
        return σp, jnp.sqrt(mels)
