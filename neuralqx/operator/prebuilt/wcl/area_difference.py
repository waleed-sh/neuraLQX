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


"""Area-difference operators."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import copy_sites
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.utils.struct import field


def _edge_area(state: jax.Array, sites: tuple[int, int, int]) -> jax.Array:
    """Return the Euclidean norm of one edge charge across three gauge copies."""
    vec = jnp.asarray([state[site] for site in sites], dtype=jnp.float64)
    return jnp.sqrt(jnp.dot(vec, vec))


class AreaDifferenceSquaredOperator(ComputationalOperator):
    r"""The squared difference between the areas of two edge punctures.

    The eigenvalue is $(\|\mathbf q_{e_1}\|_2-\|\mathbf q_{e_2}\|_2)^2$.
    It is non-negative and vanishes when the two puncture areas agree, even
    if their charge vectors differ. Exactly two edges and three gauge
    components are required.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        edges (tuple[Any, Any]): The two graph edges whose puncture areas are compared.
    """

    edges: tuple[Any, Any] = field(static=True)
    """The two graph edges whose puncture areas are compared."""

    _sites_1: tuple[int, int, int] = field(
        static=True, init=False, default=(), repr=False
    )
    """Component indices defining the first area."""

    _sites_2: tuple[int, int, int] = field(
        static=True, init=False, default=(), repr=False
    )
    """Component indices defining the second area."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        if not isinstance(self.edges, tuple) or len(self.edges) != 2:
            raise ValueError("AreaDifferenceSquaredOperator expects exactly two edges.")
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        object.__setattr__(self, "_sites_1", tuple(copy_sites(hilbert, self.edges[0])))
        object.__setattr__(self, "_sites_2", tuple(copy_sites(hilbert, self.edges[1])))
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
        """Every nonzero entry leaves the configuration unchanged."""
        return True

    @property
    def adjoint(self) -> AreaDifferenceSquaredOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"AreaDifferenceSquaredOperator(edges={self.edges!r})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        value = (
            _edge_area(state, self._sites_1) - _edge_area(state, self._sites_2)
        ) ** 2
        return state[None, :], value[None]


class AreaDifferenceSquaredSurfacesOperator(ComputationalOperator):
    r"""The squared difference between two total surface areas.

    Each surface is a collection of puncturing edges with area
    $A_S(q)=\sum_{e\in S}\|\mathbf q_e\|_2$. This operator has diagonal
    eigenvalue $(A_{S_1}(q)-A_{S_2}(q))^2$. Overlapping surfaces are allowed,
    and shared punctures cancel in their area difference.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        surfaces (tuple[tuple[Any, ...], tuple[Any, ...]]): The two collections of edges whose total areas are compared.
    """

    surfaces: tuple[tuple[Any, ...], tuple[Any, ...]] = field(static=True)
    """The two collections of edges whose total areas are compared."""

    _sites_1: tuple[tuple[int, int, int], ...] = field(
        static=True, init=False, default=(), repr=False
    )
    """Component indices defining the first area."""

    _sites_2: tuple[tuple[int, int, int], ...] = field(
        static=True, init=False, default=(), repr=False
    )
    """Component indices defining the second area."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        if not isinstance(self.surfaces, tuple) or len(self.surfaces) != 2:
            raise ValueError(
                "AreaDifferenceSquaredSurfacesOperator expects two surfaces."
            )
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        object.__setattr__(
            self,
            "_sites_1",
            tuple(tuple(copy_sites(hilbert, edge)) for edge in self.surfaces[0]),
        )
        object.__setattr__(
            self,
            "_sites_2",
            tuple(tuple(copy_sites(hilbert, edge)) for edge in self.surfaces[1]),
        )
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
        """Every nonzero entry leaves the configuration unchanged."""
        return True

    @property
    def adjoint(self) -> AreaDifferenceSquaredSurfacesOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"AreaDifferenceSquaredSurfacesOperator(surface_sizes=({len(self.surfaces[0])}, {len(self.surfaces[1])}))"

    @staticmethod
    def _surface_area(
        state: jax.Array, sites_by_edge: tuple[tuple[int, int, int], ...]
    ) -> jax.Array:
        """Sum the Euclidean norms of the listed three-component edge charges."""
        value = jnp.asarray(0.0, dtype=jnp.float64)
        for sites in sites_by_edge:
            value = value + _edge_area(state, sites)
        return value

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        value = (
            self._surface_area(state, self._sites_1)
            - self._surface_area(state, self._sites_2)
        ) ** 2
        return state[None, :], value[None]


__all__ = ["AreaDifferenceSquaredOperator", "AreaDifferenceSquaredSurfacesOperator"]
