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


"""Spherical Bojowald-Swiderski volume operator."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field

from .stencil import require_spherical_hilbert
from .stencil import spherical_stencil


class SphericalVolumeOperator(ComputationalOperator):
    r"""The local Bojowald-Swiderski volume on a half-ladder graph.

    The transverse edge at the radial vertex carries $\mu_v$ and the adjacent
    radial edges carry $k_-,k_+$. In the model's dimensionless units,

    $$
    V_v(q)=|\mu_v|\sqrt{|k_-+k_++a|},
    $$

    where $a$ is `shift`. The default is the unshifted volume. A nonzero shift
    is useful when forming finite differences of volume eigenvalues. The
    vertex must have both a left and a right radial neighbor.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        vertex (Any): Radial vertex or its index in `graph.vertices_k`.
        shift (float): Additive shift inside the radial-label sum under the square root.
    """

    vertex: Any = field(static=True)
    """Radial vertex or its index in `graph.vertices_k`."""

    shift: float = field(static=True, default=0.0)
    """Additive shift inside the radial-label sum under the square root."""

    _idx_mu: int = field(static=True, init=False, default=0, repr=False)
    """Flat index of the central transverse edge."""

    _idx_km: int = field(static=True, init=False, default=-1, repr=False)
    """Flat index of the adjacent left radial edge."""

    _idx_kp: int = field(static=True, init=False, default=-1, repr=False)
    """Flat index of the adjacent right radial edge."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_spherical_hilbert(self.hilbert, operator=type(self).__name__)
        stencil = spherical_stencil(hilbert.graph, self.vertex)
        object.__setattr__(self, "_idx_mu", stencil.idx_mu)
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
    def adjoint(self) -> SphericalVolumeOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"SphericalVolumeOperator(vertex={self.vertex!r}, shift={self.shift})"

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
        value = jnp.abs(state[self._idx_mu]).astype(self.dtype) * jnp.sqrt(
            jnp.abs((km + kp).astype(self.dtype) + self.shift)
        )
        return state[None, :], value[None]


__all__ = ["SphericalVolumeOperator"]
