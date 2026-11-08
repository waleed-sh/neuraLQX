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


"""Spherical Bojowald-Swiderski E^x operator."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field

from .stencil import require_spherical_hilbert
from .stencil import spherical_stencil


class SphericalExOperator(ComputationalOperator):
    r"""The radial triad at an interior vertex of a half-ladder graph.

    The neighboring radial edges carry labels $k_-$ and $k_+$. The diagonal
    eigenvalue is $E^x_v=\gamma(k_-+k_+)/2$, with the remaining physical unit
    factors suppressed. The sign is retained, so this operator distinguishes
    radial orientations. `vertex` is a radial vertex or its index in
    `graph.vertices_k`.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        vertex (Any): Radial vertex or its index in `graph.vertices_k`.
        gamma (float): Real Barbero-Immirzi factor multiplying the radial triad.
    """

    vertex: Any = field(static=True)
    """Radial vertex or its index in `graph.vertices_k`."""

    gamma: float = field(static=True, default=1.0)
    """Real Barbero-Immirzi factor multiplying the radial triad."""

    _idx_km: int = field(static=True, init=False, default=-1, repr=False)
    """Flat index of the adjacent left radial edge."""

    _idx_kp: int = field(static=True, init=False, default=-1, repr=False)
    """Flat index of the adjacent right radial edge."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_spherical_hilbert(self.hilbert, operator=type(self).__name__)
        stencil = spherical_stencil(hilbert.graph, self.vertex)
        object.__setattr__(self, "_idx_km", stencil.idx_km)
        object.__setattr__(self, "_idx_kp", stencil.idx_kp)
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
    def adjoint(self) -> SphericalExOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"SphericalExOperator(vertex={self.vertex!r}, gamma={self.gamma})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        km = jnp.where(self._idx_km >= 0, state[self._idx_km], 0)
        kp = jnp.where(self._idx_kp >= 0, state[self._idx_kp], 0)
        value = (
            0.5
            * jnp.asarray(self.gamma, dtype=self.dtype)
            * (km + kp).astype(self.dtype)
        )
        return state[None, :], value[None]


__all__ = ["SphericalExOperator"]
