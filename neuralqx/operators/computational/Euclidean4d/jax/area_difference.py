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
from neuralqx.utils.errors import AreaDifferenceEdgesError, AreaDifferenceSurfacesError


@jax.jit
def _area_diff_sq_kernel(
    sigma: jnp.ndarray,
    e1_comps: jnp.ndarray,
    e2_comps: jnp.ndarray,
    mels_factor: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Diagonal kernel computing (||m_e1|| - ||m_e2||)^2.

    All inputs are arrays to avoid static-arg recompiles. Assumes U(1)^3 (three components per edge)
    """

    D = sigma.shape[-1]
    sig_flat = sigma.reshape((-1, D))

    # gather 3 comps for each edge -> (M, 3)
    m1 = jnp.take(sig_flat, e1_comps, axis=-1)
    m2 = jnp.take(sig_flat, e2_comps, axis=-1)

    # squared norms and product of norms
    n1_sq = jnp.sum(m1 * m1, axis=-1)
    n2_sq = jnp.sum(m2 * m2, axis=-1)
    prod_norms = jnp.sqrt(jnp.maximum(n1_sq, 0.0)) * jnp.sqrt(jnp.maximum(n2_sq, 0.0))

    vals_flat = mels_factor * (n1_sq + n2_sq - 2.0 * prod_norms)
    vals = vals_flat.reshape(sigma.shape[:-1]).astype(jnp.float64)

    # diagonal: one connection = itself
    return sigma[..., None, :], vals[..., None]


@register_pytree_node_class
class AreaDifferenceSquaredOperatorJax(ComputationalJaxOperator):
    r"""
    Diagonal operator for the squared area difference between two edges e1,e2 in U(1)^3:
        (||m_{e1}|| - ||m_{e2}||)^2 = ||m_{e1}||^2 + ||m_{e2}||^2 - 2 ||m_{e1}|| ||m_{e2}||.
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

    def __init__(self, H: AbstractHilbertInterface, edges):

        super().__init__(H.hilbert_netket)

        self.gauge_dim = int(H.gauge_dimensions)

        edges = list(edges)
        if len(edges) != 2:
            raise AreaDifferenceEdgesError(len(edges))

        self.n_edges_total = int(H.size // self.gauge_dim)

        # map edges to flat edge indices
        e1_idx = int(H.graph.edge_to_index(edges[0]))
        e2_idx = int(H.graph.edge_to_index(edges[1]))

        # if same edge, result is identically zero, so we will short-circuit outside jit
        self._same_edge = e1_idx == e2_idx

        # offsets to (x,y,z) components within flattened D
        offsets = (
            jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total
        ).astype(jnp.int64)

        # precompute 3-component indices for each edge (int64)
        self._e1_comps = (jnp.int64(e1_idx) + offsets).astype(jnp.int64)
        self._e2_comps = (jnp.int64(e2_idx) + offsets).astype(jnp.int64)

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (self._e1_comps, self._e2_comps)
        struct = {
            "hilbert": self.hilbert,
            "gauge_dim": self.gauge_dim,
            "n_edges_total": self.n_edges_total,
            "same_edge": self._same_edge,
        }
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.gauge_dim = int(struct["gauge_dim"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._same_edge = bool(struct["same_edge"])
        obj._e1_comps, obj._e2_comps = leaves
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):

        x = jnp.asarray(x, dtype=jnp.int64)

        # if both edges are the same, the value is identically zero (skip jit entirely)
        if self._same_edge:
            return x[..., None, :], jnp.zeros(x.shape[:-1] + (1,), dtype=jnp.float64)

        sigma_p, mels = _area_diff_sq_kernel(
            sigma=x,
            e1_comps=self._e1_comps,
            e2_comps=self._e2_comps,
            mels_factor=jnp.asarray(1.0, dtype=jnp.float64),
        )

        return sigma_p, mels


@jax.jit
def _area_diff_sq_surfaces_kernel(
    sigma: jnp.ndarray,
    surf1_comps: jnp.ndarray,
    surf2_comps: jnp.ndarray,
    mels_factor: jnp.ndarray,
):
    D = sigma.shape[-1]
    sig = sigma.reshape((-1, D))

    gathered1 = jnp.take(sig, surf1_comps, axis=-1)
    gathered2 = jnp.take(sig, surf2_comps, axis=-1)

    # derive E from width // 3, use literal 3 (not a traced scalar)
    E1 = gathered1.shape[-1] // 3
    E2 = gathered2.shape[-1] // 3

    m1 = gathered1.reshape((sig.shape[0], E1, 3))
    m2 = gathered2.reshape((sig.shape[0], E2, 3))

    norms1 = jnp.sqrt(jnp.sum(m1 * m1, axis=-1))
    norms2 = jnp.sqrt(jnp.sum(m2 * m2, axis=-1))

    A1 = jnp.sum(norms1, axis=-1)
    A2 = jnp.sum(norms2, axis=-1)

    vals_flat = mels_factor * (A1 - A2) ** 2
    vals = vals_flat.reshape(sigma.shape[:-1]).astype(jnp.float64)

    return sigma[..., None, :], vals[..., None]


@register_pytree_node_class
class AreaDifferenceSquaredSurfacesOperatorJax(ComputationalJaxOperator):
    r"""
    Diagonal operator for the squared area difference between two surfaces S1,S2 in U(1)^3:
        (A(S1) - A(S2))^2,  with  A(S) = sum_{e in S} ||m_e||_2.
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

    def __init__(self, H: AbstractHilbertInterface, surfaces):

        super().__init__(H.hilbert_netket)

        self.gauge_dim = int(H.gauge_dimensions)

        surfaces = list(surfaces)
        if len(surfaces) != 2:
            raise AreaDifferenceSurfacesError(len(surfaces))

        self.n_edges_total = int(H.size // self.gauge_dim)
        offsets = (
            jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total
        ).astype(jnp.int64)

        def surface_comps(surface_edges):
            if not surface_edges:
                return jnp.zeros((0,), dtype=jnp.int64)
            edge_idx = jnp.asarray(
                [H.graph.edge_to_index(e) for e in surface_edges], dtype=jnp.int64
            )
            return (edge_idx[:, None] + offsets[None, :]).reshape(-1)

        self._surf1_comps = surface_comps(surfaces[0])
        self._surf2_comps = surface_comps(surfaces[1])

        # (optional) quick host-side equality check of the static comps to short-circuit to zero
        # equality here respects order and multiplicity, which is consistent with the definition
        self._same_surfaces = bool(
            (self._surf1_comps.shape == self._surf2_comps.shape)
            and jnp.array_equal(self._surf1_comps, self._surf2_comps)
        )

    #
    #
    #   pytree support
    def tree_flatten(self):
        leaves = (self._surf1_comps, self._surf2_comps)
        struct = {
            "hilbert": self.hilbert,
            "gauge_dim": self.gauge_dim,
            "n_edges_total": self.n_edges_total,
            "same_surfaces": self._same_surfaces,
        }
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.gauge_dim = int(struct["gauge_dim"])
        obj.n_edges_total = int(struct["n_edges_total"])
        obj._same_surfaces = bool(struct["same_surfaces"])
        obj._surf1_comps, obj._surf2_comps = leaves
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):

        x = jnp.asarray(x, dtype=jnp.int64)

        # if both surfaces are identical, the value is identically zero (skip jit entirely)
        if self._same_surfaces:
            return x[..., None, :], jnp.zeros(x.shape[:-1] + (1,), dtype=jnp.float64)

        sigma_p, mels = _area_diff_sq_surfaces_kernel(
            sigma=x,
            surf1_comps=self._surf1_comps,
            surf2_comps=self._surf2_comps,
            mels_factor=jnp.asarray(1.0, dtype=jnp.float64),
        )

        return sigma_p, mels
