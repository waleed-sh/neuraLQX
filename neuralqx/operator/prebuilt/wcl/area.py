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


"""U(1)^3 area operator."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import copy_sites
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.utils.struct import field


class AreaOperator(ComputationalOperator):
    r"""The area punctured by a collection of U(1)^3 edges.

    Write $\mathbf q_e=(q_e^{(0)},q_e^{(1)},q_e^{(2)})$. In the dimensionless
    normalization used by the WCL model, the diagonal surface area is

    $$
    A_S(q)=\sum_{e\in S}\|\mathbf q_e\|_2.
    $$

    With `squared=True`, each puncture contributes $\|\mathbf q_e\|_2^2$.
    This is the sum of squared puncture areas. To obtain the square of the
    total surface area, compose the area operator with itself. Repeated edges
    in `surface` are counted repeatedly and an empty surface has zero area.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        surface (tuple[Any, ...]): Tuple of puncturing edges, counted with their listed multiplicity.
        squared (bool): Whether to sum squared puncture areas instead of puncture areas.
    """

    surface: tuple[Any, ...] = field(static=True)
    """Tuple of puncturing edges, counted with their listed multiplicity."""

    squared: bool = field(static=True, default=False)
    """Whether to sum squared puncture areas instead of puncture areas."""

    _sites_by_edge: tuple[tuple[int, int, int], ...] = field(
        static=True, init=False, default=(), repr=False
    )
    """Three component indices for each surface puncture."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        if not isinstance(self.surface, tuple):
            raise TypeError("surface must be a tuple of graph edges or edge indices.")
        object.__setattr__(
            self,
            "_sites_by_edge",
            tuple(tuple(copy_sites(hilbert, edge)) for edge in self.surface),
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
        """Whether every nonzero matrix element connects a configuration to itself."""
        return True

    @property
    def adjoint(self) -> AreaOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"AreaOperator(n_edges={len(self.surface)}, squared={self.squared})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        value = jnp.asarray(0.0, dtype=self.dtype)
        for sites in self._sites_by_edge:
            vec = jnp.asarray([state[site] for site in sites], dtype=self.dtype)
            norm_sq = jnp.dot(vec, vec)
            value = value + (norm_sq if self.squared else jnp.sqrt(norm_sq))
        return state[None, :], value[None]


__all__ = ["AreaOperator"]
