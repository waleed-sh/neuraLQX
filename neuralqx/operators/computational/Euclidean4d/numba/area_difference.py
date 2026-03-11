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


import jax.numpy as jnp

from neuralqx.utils.errors import AreaDifferenceEdgesError
from neuralqx.utils.errors import AreaDifferenceSurfacesError

from neuralqx.operators.types.computational_operator import ComputationalOperator


class AreaDifferenceSquaredOperator(ComputationalOperator):
    r"""
    Diagonal operator implementing the squared area difference between two edges
    in the U(1)^3 model:

    .. math::

        (\hat A_{e_1} - \hat A_{e_2})^2
        \;=\;
        \hat A_{e_1}^2 + \hat A_{e_2}^2 - 2\,\hat A_{e_1}\hat A_{e_2},

    where for a single edge \(e\),
    \(\hat A_e = \|\vec m_e\|_2 = \sqrt{(m_e^{(1)})^2 + (m_e^{(2)})^2 + (m_e^{(3)})^2}\).

    Since this operator is diagonal in the \(|\vec m\rangle\) basis, its matrix element on a
    configuration \(\sigma\) is simply

    .. math::

        \big(\|\vec m_{e_1}\|_2 - \|\vec m_{e_2}\|_2\big)^2
        \;=\;
        \|\vec m_{e_1}\|_2^2 + \|\vec m_{e_2}\|_2^2
        \;-\; 2\,\|\vec m_{e_1}\|_2\,\|\vec m_{e_2}\|_2.


    The operator precomputes the three component indices for each of the two edges using the
    graph's ``edge_to_index`` and static gauge offsets. This avoids any Python-side control-flow
    inside jitted code and prevents recompilation.

    Notes:
        - Assumes U(1)^3 (``gauge_dim == 3``) and unit overall normalization (8πℓₚ² absorbed)
        - If both edges are the same, the operator correctly yields zero everywhere
    """

    def __init__(self, H, edges):

        super().__init__(H.hilbert_netket)

        # static geometry/layout
        self.gauge_dim = int(H.gauge_dimensions)

        assert self.gauge_dim == 3, (
            f"{type(self).__name__} assumes U(1)^3 (gauge_dim == 3),"
            f" got {self.gauge_dim} instead."
        )

        # validate edge count early
        edges = list(edges)

        if len(edges) != 2:
            raise AreaDifferenceEdgesError(len(edges))

        # total edges per gauge level
        self.n_edges_total = int(H.size // self.gauge_dim)

        # map edges to flat edge indices
        e1_idx = int(H.graph.edge_to_index(edges[0]))
        e2_idx = int(H.graph.edge_to_index(edges[1]))

        # to multiply the returned mels if the edges are the same
        self.mels_factor = 0 if e1_idx == e2_idx else 1

        # offsets to reach the 3 U(1) components for a given edge within the flat state
        # [0, n_edges_total, 2*n_edges_total]
        offsets = (
            jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total
        ).astype(jnp.int32)

        # precompute and freeze the three-component indices for each edge
        self._e1_comps = (jnp.int32(e1_idx) + offsets).astype(jnp.int32)
        self._e2_comps = (jnp.int32(e2_idx) + offsets).astype(jnp.int32)

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        # support arbitrary leading batch dims, last axis is D
        leading_shape = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading_shape))) if leading_shape else 1

        # flatten to (M, D) to vectorise gathers
        sig = σ.reshape((M, D))

        # gather the 3-component charge vectors for both edges
        m1 = jnp.take(sig, self._e1_comps, axis=-1)
        m2 = jnp.take(sig, self._e2_comps, axis=-1)

        # squared norms per edge
        n1_sq = jnp.sum(m1 * m1, axis=-1)
        n2_sq = jnp.sum(m2 * m2, axis=-1)

        # product of Euclidean norms (avoid negative due to dtype/rounding)
        prod_norms = jnp.sqrt(jnp.maximum(n1_sq, 0.0)) * jnp.sqrt(
            jnp.maximum(n2_sq, 0.0)
        )

        # final diagonal value per sample: (||m1|| - ||m2||)^2
        vals = (self.mels_factor * (n1_sq + n2_sq - 2.0 * prod_norms)).astype(
            self.dtype
        )

        # restore leading shape and pad as a single diagonal connection
        vals = vals.reshape(leading_shape)
        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(vals, axis=-1)

        return σp, mels


class AreaDifferenceSquaredSurfacesOperator(ComputationalOperator):
    r"""
    Diagonal operator implementing the squared area difference between two surfaces
    (each a list of edges) in the U(1)^3 model:

    .. math::

        (\hat A(S_1) - \hat A(S_2))^2
        \;=\;
        \hat A(S_1)^2 + \hat A(S_2)^2 - 2\,\hat A(S_1)\hat A(S_2),

    where for a surface \(S = \{e_i\}\),

    .. math::
        \hat A(S) = \sum_{e_i \in S} \sqrt{ (m_{e_i}^{(1)})^2 + (m_{e_i}^{(2)})^2 + (m_{e_i}^{(3)})^2 }.

    This operator is diagonal in the \(|\vec m\rangle\) basis, and so its matrix element
    on a configuration \(\sigma\) is simply

    .. math::
        \big(\hat A(S_1) - \hat A(S_2)\big)^2.

    Notes:
        - Assumes U(1)^3 (``gauge_dim == 3``) and unit normalization (8πℓₚ² absorbed).
        - If both surfaces are identical, the operator yields zero everywhere.
    """

    def __init__(self, H, surfaces):
        super().__init__(H.hilbert_netket)

        # static geometry/layout
        self.gauge_dim = int(H.gauge_dimensions)

        assert self.gauge_dim == 3, (
            f"{type(self).__name__} assumes U(1)^3 (`gauge_dim = 3`), "
            f"got {self.gauge_dim} instead."
        )

        surfaces = list(surfaces)
        if len(surfaces) != 2:
            raise AreaDifferenceSurfacesError(len(surfaces))

        # total edges per gauge level
        self.n_edges_total = int(H.size // self.gauge_dim)
        offsets = (
            jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total
        ).astype(jnp.int32)

        # helper to build flattened component indices for all edges in a surface
        def surface_comps(surface_edges):
            if not surface_edges:
                return jnp.zeros((0,), dtype=jnp.int32)
            edge_indices = jnp.array(
                [H.graph.edge_to_index(e) for e in surface_edges],
                dtype=jnp.int32,
            )
            return (edge_indices[:, None] + offsets[None, :]).reshape(-1)

        self._surf1_comps = surface_comps(surfaces[0])
        self._surf2_comps = surface_comps(surfaces[1])

        # factors: handle identical surfaces (to avoid unnecessary work)
        self.mels_factor = (
            0 if jnp.array_equal(self._surf1_comps, self._surf2_comps) else 1
        )

        # pre-store edge counts for reshaping
        self._E1 = len(surfaces[0])
        self._E2 = len(surfaces[1])

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        leading_shape = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading_shape))) if leading_shape else 1

        sig = σ.reshape((M, D))

        # gather charge vectors for both surfaces
        m1 = jnp.take(sig, self._surf1_comps, axis=-1)
        m2 = jnp.take(sig, self._surf2_comps, axis=-1)

        # reshape into (..., E, gauge_dim)
        if self._E1 > 0:
            m1 = m1.reshape((M, self._E1, self.gauge_dim))
            norms1 = jnp.sqrt(jnp.sum(m1 * m1, axis=-1))
            A1 = jnp.sum(norms1, axis=-1)
        else:
            A1 = jnp.zeros((M,), dtype=self.dtype)

        if self._E2 > 0:
            m2 = m2.reshape((M, self._E2, self.gauge_dim))
            norms2 = jnp.sqrt(jnp.sum(m2 * m2, axis=-1))
            A2 = jnp.sum(norms2, axis=-1)
        else:
            A2 = jnp.zeros((M,), dtype=self.dtype)

        # area difference squared
        vals = self.mels_factor * (A1 - A2) ** 2
        vals = vals.reshape(leading_shape)

        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(vals, axis=-1)

        return σp, mels
