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


"""U(1)^g Gauss constraint operator."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import site_index
from neuralqx.operator.prebuilt.utils import wrap_add_value
from neuralqx.utils.struct import field


class GaussConstraintOperator(ComputationalOperator):
    r"""The total squared Gauss-law violation of a U(1) charge network.

    At each vertex and in each gauge component, incoming charges enter with
    a positive sign and outgoing charges with a negative sign. The diagonal
    eigenvalue is

    $$
    \mathcal G(q)=\sum_v\sum_c
    \left(\sum_{e\to v}q_e^{(c)}-\sum_{e\leftarrow v}q_e^{(c)}\right)^2.
    $$

    A self-loop contributes once to each incidence sum and cancels. With
    `modular=True`, each vertex residual is reduced to the local cyclic charge
    range before squaring. This implements the finite cyclic version of
    Gauss law rather than the ordinary integer constraint.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        modular (bool): Whether to reduce each Gauss residual to the cyclic charge range.
    """

    modular: bool = field(static=True, default=False)
    """Whether to reduce each Gauss residual to the cyclic charge range."""

    _sites_by_vertex: tuple[tuple[tuple[int, ...], ...], ...] = field(
        static=True,
        init=False,
        default=(),
        repr=False,
    )
    """Incident edge-component indices grouped by graph vertex."""

    _signs_by_vertex: tuple[tuple[int, ...], ...] = field(
        static=True,
        init=False,
        default=(),
        repr=False,
    )
    """Incoming plus signs and outgoing minus signs for each vertex."""

    _q_min: int | float = field(static=True, init=False, default=0, repr=False)
    """Smallest allowed value on the local charge lattice."""

    _q_step: int | float = field(static=True, init=False, default=1, repr=False)
    """Spacing between adjacent allowed local charges."""

    _local_size: int = field(static=True, init=False, default=0, repr=False)
    """Number of allowed charges in one local degree of freedom."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        q_min, _q_max, q_step, local_size = q_bounds(hilbert)
        connectivity = getattr(hilbert.graph, "connectivity", None)
        if connectivity is None:
            raise ValueError(
                f"{type(self).__name__} requires graph connectivity metadata."
            )

        sites_by_vertex = []
        signs_by_vertex = []
        for vertex in hilbert.graph.vertices:
            incident_sites = []
            incident_signs = []
            entry = connectivity.get(vertex, {})
            for edge in entry.get("incoming", ()):
                incident_sites.append(
                    tuple(
                        site_index(hilbert, edge, copy)
                        for copy in range(hilbert.gauge_dimensions)
                    )
                )
                incident_signs.append(+1)
            for edge in entry.get("outgoing", ()):
                incident_sites.append(
                    tuple(
                        site_index(hilbert, edge, copy)
                        for copy in range(hilbert.gauge_dimensions)
                    )
                )
                incident_signs.append(-1)
            sites_by_vertex.append(tuple(incident_sites))
            signs_by_vertex.append(tuple(incident_signs))

        object.__setattr__(self, "_sites_by_vertex", tuple(sites_by_vertex))
        object.__setattr__(self, "_signs_by_vertex", tuple(signs_by_vertex))
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_step", q_step)
        object.__setattr__(self, "_local_size", local_size)
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
    def adjoint(self) -> GaussConstraintOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return (
            "GaussConstraintOperator("
            f"gauge_dimensions={self.hilbert.gauge_dimensions}, modular={self.modular})"
        )

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        total = jnp.asarray(0.0, dtype=self.dtype)
        for sites, signs in zip(
            self._sites_by_vertex, self._signs_by_vertex, strict=True
        ):
            gauss = jnp.zeros((int(self.hilbert.gauge_dimensions),), dtype=state.dtype)
            for component_sites, sign in zip(sites, signs, strict=True):
                charge = jnp.asarray(
                    [state[site] for site in component_sites], dtype=state.dtype
                )
                if self.modular:
                    gauss = wrap_add_value(
                        gauss,
                        int(sign) * charge,
                        q_min=self._q_min,
                        q_step=self._q_step,
                        local_size=self._local_size,
                    )
                else:
                    gauss = gauss + int(sign) * charge
            gauss_f = gauss.astype(self.dtype)
            total = total + jnp.dot(gauss_f, gauss_f)
        return state[None, :], total[None]


__all__ = ["GaussConstraintOperator"]
