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
An implementation for the area operator of the Euclidean WCL 4-d model using a ComputationalOperator
rather than a LocalOperator
"""

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


class AreaOperator(ComputationalOperator):
    r"""
    Returns the area operator for a surface through which edges pass through. The specified
    ``surface`` should be a list of edges of the graph which are assumed to pucnture through
    the desired surface.

    The area for a given graph is given by

    .. math::

        \hat{A}(S) |\vec{m}\rangle
        =
        \sum_{e \cap S\neq\varnothing}
        \sqrt{ \big(m_e^{(1)}\big)^2 + \big(m_e^{(2)}\big)^2 + \big(m_e^{(3)}\big)^2 } |\vec{m}\rangle

    i.e. for each edge intersecting \(S\) take its three U(1) charges, square them, sum, and
    take the square root. (In the generic normalization one would multiply by 8πℓₚ² here that
    prefactor is set to 1 by convention, see Notes.)

    Orientation of an edge does not affect the result since the operator uses the charge
    **norm**. If an edge punctures the surface multiple times, it should be included with
    several times.

    For non-planar graphs, these edges should be supplied in accordance to the edges appearing
    from the ``neuralqx.graph.Graph.edges`` (that is, not in the non-planar representation).

    Notes: here it is assumed the following
        - unit Planck length
        - Immirzi parameter of value 1.0
        - Hence the 8πℓₚ² factor is absorbed into units
    """

    def __init__(self, H, edges, *, squared: bool = False):
        super().__init__(H.hilbert)

        # static geometry/layout
        self.gauge_dim = int(H.gauge_dimensions)

        # get the number of edges per gauge level
        self.n_edges_total = int(H.size // self.gauge_dim)

        # precompute component indices for all requested edges
        # edges_idx: (E,)
        edges_idx = jnp.array(
            [H.graph.edge_to_index(e) for e in edges], dtype=jnp.int32
        )
        self._E = int(edges_idx.size)

        # offsets to reach (x, y, z) components of a given edge within flattened D
        # offsets: (3,) = [0, n_edges_total, 2*n_edges_total] for U(1)^3
        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        # comps_flat: (E * 3,) indices for all requested edges' 3 components (for U(1)^3)
        # layout here is [e0_x, e0_y, e0_z, e1_x, e1_y, e1_z, ...]
        if self._E > 0:
            comps = edges_idx[:, None] + offsets[None, :]
            self._comps_flat = comps.reshape(-1)
        else:
            self._comps_flat = jnp.zeros((0,), dtype=jnp.int32)

        self.squared = squared

    @property
    def is_hermitian(self) -> bool:
        # this operator is Hermitian
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        # support arbitrary leading batch dims
        leading_shape = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading_shape))) if leading_shape else 1

        # if no edges to measure, just return zeros (still one diagonal connection)
        if self._E == 0:
            σp = jnp.expand_dims(σ, axis=-2)
            zeros = jnp.zeros(leading_shape, dtype=self.dtype)
            mels = jnp.expand_dims(zeros, axis=-1)
            return σp, mels

        # flatten leading batch dims to (M, D)
        sig = σ.reshape((M, D))

        # gather all requested components in a single take
        # gathered: (M, E * 3) which will be reshaped to (M, E, 3)
        gathered = jnp.take(sig, self._comps_flat, axis=-1)
        gathered = gathered.reshape((M, self._E, self.gauge_dim))

        # per-edge Euclidean norms with now (M, E)
        norms = jnp.sum(gathered * gathered, axis=-1)

        if not self.squared:
            norms = jnp.sqrt(norms)

        # we sum over edges, now we have shape (M,)
        areas_flat = jnp.sum(norms, axis=-1)

        # restore leading shape
        areas = areas_flat.reshape(leading_shape).astype(self.dtype)

        # diagonal operator: one connection = itself
        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(areas, axis=-1)

        return σp, mels
