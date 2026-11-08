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


"""Hermitian one-minus-half loop holonomy."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import parse_loop
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import shift_sites
from neuralqx.operator.prebuilt.utils import site_index
from neuralqx.utils.struct import field


class OneMinusHalfHolonomyOperator(ComputationalOperator):
    r"""The positive loop-curvature operator for one U(1) component.

    For the cyclic loop holonomy $H_\alpha$, this operator is

    $$
    K_\alpha=I-\frac12(H_\alpha+H_\alpha^\dagger)
    =\frac12(I-H_\alpha)^\dagger(I-H_\alpha).
    $$

    It is Hermitian and positive semidefinite. Its three padded branches have
    weights $1,-1/2,-1/2$. Coincident targets are retained in padded output,
    so a trivial loop cancels only when those contributions are summed.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        loop (Any): Dressed graph loop or sequence of edges defining its shifts.
        copy (int): Zero-based U(1) component in the component-major state layout.
    """

    loop: Any = field(pytree=False)
    """Dressed graph loop or sequence of edges defining its shifts."""

    copy: int = field(static=True, default=0)
    """Zero-based U(1) component in the component-major state layout."""

    _sites: tuple[int, ...] = field(static=True, init=False, default=(), repr=False)
    """Flat sites updated by the stored shift pattern."""

    _directions: tuple[int, ...] = field(
        static=True, init=False, default=(), repr=False
    )
    """Signed lattice-step counts, including any adjoint reversal."""

    _q_min: int | float = field(static=True, init=False, default=0, repr=False)
    """Smallest allowed value on the local charge lattice."""

    _q_step: int | float = field(static=True, init=False, default=1, repr=False)
    """Spacing between adjacent allowed local charges."""

    _local_size: int = field(static=True, init=False, default=0, repr=False)
    """Number of allowed charges in one local degree of freedom."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        edges, directions = parse_loop(hilbert, self.loop)
        q_min, _q_max, q_step, local_size = q_bounds(hilbert)
        object.__setattr__(
            self,
            "_sites",
            tuple(site_index(hilbert, edge, self.copy) for edge in edges),
        )
        object.__setattr__(
            self, "_directions", tuple(int(direction) for direction in directions)
        )
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_step", q_step)
        object.__setattr__(self, "_local_size", local_size)
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return 3

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return True

    @property
    def adjoint(self) -> OneMinusHalfHolonomyOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return (
            f"OneMinusHalfHolonomyOperator(length={len(self._sites)}, copy={self.copy})"
        )

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        forward = shift_sites(
            state,
            self._sites,
            self._directions,
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
        )
        backward = shift_sites(
            state,
            self._sites,
            self._directions,
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
            scale=-1,
        )
        return jnp.stack((state, forward, backward), axis=0), jnp.asarray(
            (1.0, -0.5, -0.5), dtype=self.dtype
        )


__all__ = ["OneMinusHalfHolonomyOperator"]
