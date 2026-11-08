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


"""Single-edge U(1) holonomy operators."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import copy_sites
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import site_index
from neuralqx.operator.prebuilt.utils import wrap_add_value
from neuralqx.utils.struct import field


class U1HolonomyOperator(ComputationalOperator):
    r"""A cyclic holonomy on one edge and one U(1) component.

    The forward row connects $q$ to $q+h e_{e,c}$ with unit amplitude, where
    $h$ is the local charge spacing. The target wraps into the allowed local
    range at a cutoff boundary. Consequently the finite holonomy is unitary,
    and its adjoint is the negative cyclic shift.

    `copy` selects a component when the Hilbert space has several U(1) copies.
    The direction describes connected-state rows, using the convention
    $H_{q,q'}=\langle q|H|q'\rangle$.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        edge (Any): Graph edge or base-edge index on which the operator acts.
        adjoint_flag (bool): Whether to evaluate the conjugate-transpose operator.
        copy (int): Zero-based U(1) component in the component-major state layout.
    """

    edge: Any = field(static=True)
    """Graph edge or base-edge index on which the operator acts."""

    adjoint_flag: bool = field(static=True, default=False)
    """Whether to evaluate the conjugate-transpose operator."""

    copy: int = field(static=True, default=0)
    """Zero-based U(1) component in the component-major state layout."""

    _site: int = field(static=True, init=False, default=0, repr=False)
    """Validated flat index used by the connectivity kernel."""

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
        object.__setattr__(self, "_site", site_index(hilbert, self.edge, self.copy))
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
        return False

    @property
    def adjoint(self) -> U1HolonomyOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return U1HolonomyOperator(
            self.hilbert, self.edge, not self.adjoint_flag, copy=self.copy
        )

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        kind = "adjoint" if self.adjoint_flag else "forward"
        return f"U1HolonomyOperator(edge={self.edge!r}, copy={self.copy}, direction={kind}, hilbert={self.hilbert!r})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        direction = -1 if self.adjoint_flag else +1
        shifted = wrap_add_value(
            state[self._site],
            direction * jnp.asarray(self._q_step, dtype=state.dtype),
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
        )
        return state.at[self._site].set(shifted)[None, :], jnp.ones(
            (1,), dtype=self.dtype
        )


class NDU1HolonomyOperator(ComputationalOperator):
    r"""The sum of single-edge holonomies over all U(1) components.

    For $g$ gauge copies this operator is $\sum_{c=0}^{g-1}H_e^{(c)}$.
    It emits one target per copy, each differing in only that copy's charge.
    It does not shift all components in one target. Every shift wraps at the
    local cutoff, and the adjoint reverses all component shifts.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        edge (Any): Graph edge or base-edge index on which the operator acts.
        adjoint_flag (bool): Whether to evaluate the conjugate-transpose operator.
    """

    edge: Any = field(static=True)
    """Graph edge or base-edge index on which the operator acts."""

    adjoint_flag: bool = field(static=True, default=False)
    """Whether to evaluate the conjugate-transpose operator."""

    _sites: tuple[int, ...] = field(static=True, init=False, default=(), repr=False)
    """One flat site for the chosen edge in every U(1) copy."""

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
        object.__setattr__(self, "_sites", copy_sites(hilbert, self.edge))
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_step", q_step)
        object.__setattr__(self, "_local_size", local_size)
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return len(self._sites)

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return False

    @property
    def adjoint(self) -> NDU1HolonomyOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return NDU1HolonomyOperator(self.hilbert, self.edge, not self.adjoint_flag)

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        kind = "adjoint" if self.adjoint_flag else "forward"
        return f"NDU1HolonomyOperator(edge={self.edge!r}, direction={kind}, copies={len(self._sites)})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        direction = -1 if self.adjoint_flag else +1
        delta = direction * jnp.asarray(self._q_step, dtype=state.dtype)
        out = []
        for site in self._sites:
            out.append(
                state.at[site].set(
                    wrap_add_value(
                        state[site],
                        delta,
                        q_min=self._q_min,
                        q_step=self._q_step,
                        local_size=self._local_size,
                    )
                )
            )
        return jnp.stack(out, axis=0), jnp.ones((len(out),), dtype=self.dtype)


__all__ = ["NDU1HolonomyOperator", "U1HolonomyOperator"]
