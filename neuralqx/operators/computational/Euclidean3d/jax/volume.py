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


from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


@jax.jit
def _area_kernel(
    sigma: jnp.ndarray,
    comps1: jnp.ndarray,
    comps2: jnp.ndarray,
    signs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # if no pairs: early diagonal zero (handle shape 0 cleanly)
    P = signs.shape[0]
    if P == 0:
        return sigma[..., None, :], jnp.zeros(
            sigma.shape[:-1] + (1,), dtype=jnp.float64
        )

    # collapse leading dims for a single gather
    D = sigma.shape[-1]
    sig_flat = sigma.reshape((-1, D))

    # gather 3-vectors per pair at all copies -> (M, P, 3)
    m1 = jnp.take(sig_flat, comps1, axis=-1)
    m2 = jnp.take(sig_flat, comps2, axis=-1)

    # cross products -> (M, P, 3)
    cr = jnp.cross(m1, m2, axis=-1)

    # signed sum over pairs -> (M, 3)
    vec = jnp.sum(cr * signs[None, :, None], axis=1)

    # Euclidean norm -> (M,)
    area_M = jnp.linalg.norm(vec, axis=-1).astype(jnp.float64)

    # restore leading shape
    area = area_M.reshape(sigma.shape[:-1])

    # one diagonal connection
    return sigma[..., None, :], area[..., None]


@register_pytree_node_class
class VolumeOperatorJax(ComputationalJaxOperator):
    """
    JAX-compatible (diagonal) 2+1 "volume" (= area) operator at a single vertex v for U(1)^3.
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

        super().__init__(H.hilbert)

        # small static metadata
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex
        self.gauge_dim = int(H.gauge_dimensions)
        assert (
            self.gauge_dim == 3
        ), f"{type(self).__name__} assumes gauge_dim == 3 (U(1)^3)."

        self.n_edges_total = int(H.size // self.gauge_dim)

        # graph pairs and signs
        pairs = H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = H.graph.signs[str(self.vertex)]

        # build gather indices for all 3 copies
        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total

        comps1, comps2, signs = [], [], []
        for pair in pairs:
            e1, e2 = map(H.graph.edge_to_index, pair)

            c1 = (jnp.int64(e1) + offsets).astype(jnp.int64)
            c2 = (jnp.int64(e2) + offsets).astype(jnp.int64)

            comps1.append(c1)
            comps2.append(c2)

            signs.append(int(signs_dict[str(pair)]))

        if len(comps1) > 0:
            self._comps1 = jnp.stack(comps1, axis=0).astype(jnp.int64)
            self._comps2 = jnp.stack(comps2, axis=0).astype(jnp.int64)
            self._signs = jnp.asarray(signs, dtype=jnp.int64)
        else:
            self._comps1 = jnp.zeros((0, 3), dtype=jnp.int64)
            self._comps2 = jnp.zeros((0, 3), dtype=jnp.int64)
            self._signs = jnp.zeros((0,), dtype=jnp.int64)

        self._P = int(self._comps1.shape[0])

    #
    #   pytree support

    def tree_flatten(self):
        # arrays as leaves, small metadata as struct
        leaves = (self._comps1, self._comps2, self._signs)
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
        ComputationalJaxOperator.__init__(obj, struct["hilbert"])

        # restore static fields
        obj.vertex = int(struct["vertex"])
        obj.gauge_dim = int(struct["gauge_dim"])
        obj.n_edges_total = int(struct["n_edges_total"])

        # restore leaves
        obj._comps1, obj._comps2, obj._signs = leaves
        obj._P = int(obj._comps1.shape[0])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = jnp.asarray(x, dtype=jnp.int64)

        # early exit outside jit for P == 0 to avoid useless tracing/launches
        if self._P == 0:
            return x[..., None, :], jnp.zeros(x.shape[:-1] + (1,), dtype=jnp.float64)

        σp, mels = _area_kernel(
            sigma=x,
            comps1=self._comps1,
            comps2=self._comps2,
            signs=self._signs,
        )
        return σp, mels


@register_pytree_node_class
class VolumeOperatorJaxSquared(VolumeOperatorJax):
    r"""
    JAX-compatible (diagonal) squared 2+1 "volume" operator at a single vertex ``v`` for U(1)^3.

    This operator represents the functional calculus :math:`\hat{V}_v^2 = (\hat{V}_v)^2`
    applied to the diagonal 2+1D "volume" operator :math:`\hat{V}_v` implemented by
    :class:`~neuralqx.operators.computational.Euclidean3d.jax.VolumeOperatorJax`.

    On a charge basis state :math:`|\vec m\rangle`, where the underlying operator has the
    eigenvalue

    .. math::

        V_v(\vec m)
        \;=\;
        \left\|
            \sum_{(e_1,e_2)\ni v}
            \epsilon_v(e_1,e_2)\;
            \vec m_{e_1}\times \vec m_{e_2}
        \right\|
        \;\ge 0,

    this class acts diagonally as

    .. math::

        \hat{V}_v^2 \,|\vec m\rangle \;=\; \big(V_v(\vec m)\big)^2 \,|\vec m\rangle.

    Notes:
        - The operator is diagonal in the charge basis, hence Hermitian.
        - This is implemented by squaring the diagonal matrix elements produced by
          the parent operator.
        - In contrast to computing :math:`V_v(\vec m)` and then squaring it, one may
          also implement :math:`\hat{V}_v^2` directly as :math:`\|\vec v\|^2` to avoid
          an intermediate square root. This class keeps the implementation minimal and
          reuses the parent kernel.

    """

    def _get_conn_padded(self, x):
        sp, mels = super()._get_conn_padded(x)
        return sp, mels * mels


@register_pytree_node_class
class VolumeOperatorJaxSqrt(VolumeOperatorJax):
    r"""
    JAX-compatible (diagonal) square-root 2+1 "volume" operator at a single vertex ``v`` for U(1)^3.

    This operator represents the functional calculus :math:`\sqrt{\hat{V}_v}` applied to the
    diagonal 2+1D "volume" operator :math:`\hat{V}_v` implemented by
    :class:`~neuralqx.operators.computational.Euclidean3d.jax.VolumeOperatorJax`.

    On a charge basis state :math:`|\vec m\rangle`, where the underlying operator has the
    eigenvalue

    .. math::

        V_v(\vec m)
        \;=\;
        \left\|
            \sum_{(e_1,e_2)\ni v}
            \epsilon_v(e_1,e_2)\;
            \vec m_{e_1}\times \vec m_{e_2}
        \right\|
        \;\ge 0,

    this class acts diagonally as

    .. math::

        \sqrt{\hat{V}_v}\,|\vec m\rangle
        \;=\;
        \sqrt{V_v(\vec m)}\,|\vec m\rangle.

    Notes:
        - In the present 2+1D U(1)^3 setting, :math:`V_v(\vec m)` is a Euclidean norm and is
          therefore non-negative, so the square root is unambiguous.
        - The operator is diagonal in the charge basis, hence Hermitian.
        - This is implemented by applying ``jnp.sqrt`` to the diagonal matrix elements produced
          by the parent operator.

    """

    def _get_conn_padded(self, x):
        sp, mels = super()._get_conn_padded(x)
        return sp, jnp.sqrt(mels)
