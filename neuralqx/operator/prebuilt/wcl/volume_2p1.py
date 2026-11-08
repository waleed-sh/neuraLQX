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


"""Planar 2+1 U(1)^3 WCL vertex volume."""

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
from neuralqx.operator.prebuilt.utils import require_planar_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.utils.struct import field


def planar_volume_terms(hilbert, vertex: Any):
    """Collect ordered incident edge pairs, their component sites, and geometric signs."""
    vertex = graph_vertex(hilbert.graph, vertex)
    terms = []
    for pair in incident_edge_tuples(hilbert.graph, vertex, 2):
        sign = orientation_sign(hilbert.graph, vertex, pair)
        if sign == 0:
            continue
        terms.append(
            (
                tuple(copy_sites(hilbert, pair[0])),
                tuple(copy_sites(hilbert, pair[1])),
                int(sign),
            )
        )
    return tuple(terms)


def planar_volume_value(state: jax.Array, terms) -> jax.Array:
    r"""Evaluate $V_v(q)$ as the norm of the signed planar cross-product sum."""
    vec_sum = jnp.zeros((3,), dtype=jnp.float64)
    for sites1, sites2, sign in terms:
        v1 = jnp.asarray([state[site] for site in sites1], dtype=jnp.float64)
        v2 = jnp.asarray([state[site] for site in sites2], dtype=jnp.float64)
        vec_sum = vec_sum + jnp.asarray(sign, dtype=jnp.float64) * jnp.cross(v1, v2)
    return jnp.linalg.norm(vec_sum)


def sqrt_planar_volume_value(state: jax.Array, terms) -> jax.Array:
    r"""Evaluate $\sqrt{V_v(q)}$ for the planar Thiemann commutators."""
    return jnp.sqrt(planar_volume_value(state, terms))


class PlanarVolumeOperator(ComputationalOperator):
    r"""The vertex volume, or spatial area, of the planar U(1)^3 model.

    For the graph's ordered incident edge pairs and orientation signs,

    $$
    V_v(q)=\left\|\sum_{(e,f)\ni v}
    \epsilon_v(e,f)\,\mathbf q_e\times\mathbf q_f\right\|_2.
    $$

    The pair sum includes both orderings, as in the upstream convention.
    The default eigenvalue is $V_v$. `squared=True` gives $V_v^2$ and
    `sqrt=True` gives $\sqrt{V_v}$. When both flags are set, squaring is
    performed first and the result is $V_v$. The square-root option is the
    quantity used in the planar Thiemann commutators.

    A planar graph and exactly three gauge components are required.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        vertex (Any): Graph vertex or its index in the graph vertex sequence.
        squared (bool): Whether to square the volume eigenvalue.
        sqrt (bool): Whether to take a square root after applying the squared option.
    """

    vertex: Any = field(static=True)
    """Graph vertex or its index in the graph vertex sequence."""

    squared: bool = field(static=True, default=False)
    """Whether to square the volume eigenvalue."""

    sqrt: bool = field(static=True, default=False)
    """Whether to take a square root after applying the squared option."""

    _terms: tuple = field(static=True, init=False, default=(), repr=False)
    """Static incident-edge terms used to evaluate the vertex operator."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        require_planar_graph(hilbert.graph, model=type(self).__name__)
        object.__setattr__(self, "_terms", planar_volume_terms(hilbert, self.vertex))
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
    def adjoint(self) -> PlanarVolumeOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"PlanarVolumeOperator(vertex={self.vertex!r}, n_terms={len(self._terms)}, squared={self.squared}, sqrt={self.sqrt})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        value = planar_volume_value(state, self._terms)
        if self.squared:
            value = value * value
        if self.sqrt:
            value = jnp.sqrt(value)
        return state[None, :], value[None]


__all__ = [
    "PlanarVolumeOperator",
    "planar_volume_terms",
    "planar_volume_value",
    "sqrt_planar_volume_value",
]
