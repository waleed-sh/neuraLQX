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


"""Non-planar 3+1 U(1)^3 WCL vertex volume."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import copy_sites
from neuralqx.operator.prebuilt.utils import graph_vertex
from neuralqx.operator.prebuilt.utils import incident_edge_tuples
from neuralqx.operator.prebuilt.utils import orientation_sign
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_nonplanar_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.utils.struct import field


def volume_terms_3p1(hilbert, vertex: Any):
    """Collect ordered incident triples, their component sites, and geometric signs."""
    vertex = graph_vertex(hilbert.graph, vertex)
    terms = []
    for triple in incident_edge_tuples(hilbert.graph, vertex, 3):
        sign = orientation_sign(hilbert.graph, vertex, triple)
        if sign == 0:
            continue
        terms.append(
            (
                tuple(copy_sites(hilbert, triple[0])),
                tuple(copy_sites(hilbert, triple[1])),
                tuple(copy_sites(hilbert, triple[2])),
                int(sign),
            )
        )
    return tuple(terms)


def volume_value_3p1(state: jax.Array, terms, *, squared: bool = False) -> jax.Array:
    r"""Evaluate $\sqrt{|Q_v|}$, or $|Q_v|$ when `squared=True`, from the triple-product sum."""
    raw = jnp.asarray(0.0, dtype=jnp.float64)
    for sites1, sites2, sites3, sign in terms:
        v1 = jnp.asarray([state[site] for site in sites1], dtype=jnp.float64)
        v2 = jnp.asarray([state[site] for site in sites2], dtype=jnp.float64)
        v3 = jnp.asarray([state[site] for site in sites3], dtype=jnp.float64)
        raw = raw + jnp.asarray(sign, dtype=jnp.float64) * jnp.dot(
            v1, jnp.cross(v2, v3)
        )
    volume_sq = jnp.abs(raw)
    return volume_sq if squared else jnp.sqrt(volume_sq)


class VolumeOperator(ComputationalOperator):
    r"""The vertex volume of the spatially three-dimensional U(1)^3 model.

    For ordered triples of incident edges, the diagonal eigenvalue is

    $$
    V_v(q)=\sqrt{\left|\sum_{(e,f,g)\ni v}\epsilon_v(e,f,g)
    \mathbf q_e\cdot(\mathbf q_f\times\mathbf q_g)\right|}.
    $$

    The graph supplies the tangent-orientation signs. The sum retains all
    ordered triples and uses the dimensionless normalization of the WCL
    implementation. `squared=True` returns the absolute value inside the
    square root. Vertices with no contributing triples have zero volume.

    A non-planar graph and three gauge components are required.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        vertex (Any): Graph vertex or its index in the graph vertex sequence.
        squared (bool): Whether to square the volume eigenvalue.
    """

    vertex: Any = field(static=True)
    """Graph vertex or its index in the graph vertex sequence."""

    squared: bool = field(static=True, default=False)
    """Whether to square the volume eigenvalue."""

    _terms: tuple = field(static=True, init=False, default=(), repr=False)
    """Static incident-edge terms used to evaluate the vertex operator."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        require_nonplanar_graph(hilbert.graph, model=type(self).__name__)
        object.__setattr__(self, "_terms", volume_terms_3p1(hilbert, self.vertex))
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return 1

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return True

    @property
    def is_diagonal(self) -> bool:
        """Whether every nonzero matrix element connects a configuration to itself."""
        return True

    @property
    def adjoint(self) -> VolumeOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"VolumeOperator(vertex={self.vertex!r}, n_terms={len(self._terms)}, squared={self.squared})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        return (
            state[None, :],
            volume_value_3p1(state, self._terms, squared=self.squared)[None],
        )


__all__ = ["VolumeOperator", "volume_terms_3p1", "volume_value_3p1"]
