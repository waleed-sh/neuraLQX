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


"""Minimal-loop holonomy operators."""

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


class MinimalLoopHolonomyOperator(ComputationalOperator):
    r"""A cyclic holonomy around a dressed loop in one U(1) component.

    A dressing supplies signs $d_e=\pm1$ relative to each stored graph edge.
    The row target is $q'=q+h\sum_{e\in\alpha}d_e e_{e,c}$, with each charge
    wrapped into the finite local range. All loop shifts occur in the same
    configuration. The adjoint reverses the dressing signs.

    Use a loop returned by `graph.dressed_minimal_loops()` to retain its
    orientation. A plain sequence of edges is also accepted and assigns a
    positive shift to each listed edge.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        loop (Any): Dressed graph loop or sequence of edges defining its shifts.
        copy (int): Zero-based U(1) component in the component-major state layout.
        adjoint_flag (bool): Whether to evaluate the conjugate-transpose operator.
    """

    loop: Any = field(pytree=False)
    """Dressed graph loop or sequence of edges defining its shifts."""

    copy: int = field(static=True, default=0)
    """Zero-based U(1) component in the component-major state layout."""

    adjoint_flag: bool = field(static=True, default=False)
    """Whether to evaluate the conjugate-transpose operator."""

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
        if self.adjoint_flag:
            directions = tuple(-direction for direction in directions)
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
    def adjoint(self) -> MinimalLoopHolonomyOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return MinimalLoopHolonomyOperator(
            self.hilbert,
            self.loop,
            copy=self.copy,
            adjoint_flag=not self.adjoint_flag,
        )

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        kind = "adjoint" if self.adjoint_flag else "forward"
        return f"MinimalLoopHolonomyOperator(length={len(self._sites)}, copy={self.copy}, direction={kind})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        out = shift_sites(
            state,
            self._sites,
            self._directions,
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
        )
        return out[None, :], jnp.ones((1,), dtype=self.dtype)


class NDMinimalLoopHolonomyOperator(ComputationalOperator):
    r"""The sum of a dressed loop holonomy over all gauge copies.

    The operator is $\sum_c H_\alpha^{(c)}$. Each output branch traverses the
    whole loop in one component. Its connection capacity therefore equals the
    number of gauge copies, independently of the loop length. Charge shifts
    are cyclic and the adjoint reverses the loop in every component.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        loop (Any): Dressed graph loop or sequence of edges defining its shifts.
        adjoint_flag (bool): Whether to evaluate the conjugate-transpose operator.
    """

    loop: Any = field(pytree=False)
    """Dressed graph loop or sequence of edges defining its shifts."""

    adjoint_flag: bool = field(static=True, default=False)
    """Whether to evaluate the conjugate-transpose operator."""

    _sites_by_copy: tuple[tuple[int, ...], ...] = field(
        static=True, init=False, default=(), repr=False
    )
    """Flat loop sites grouped by U(1) component."""

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
        if self.adjoint_flag:
            directions = tuple(-direction for direction in directions)
        q_min, _q_max, q_step, local_size = q_bounds(hilbert)
        sites_by_copy = tuple(
            tuple(site_index(hilbert, edge, copy) for edge in edges)
            for copy in range(int(hilbert.gauge_dimensions))
        )
        object.__setattr__(self, "_sites_by_copy", sites_by_copy)
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
        return len(self._sites_by_copy)

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return False

    @property
    def adjoint(self) -> NDMinimalLoopHolonomyOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return NDMinimalLoopHolonomyOperator(
            self.hilbert, self.loop, not self.adjoint_flag
        )

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        kind = "adjoint" if self.adjoint_flag else "forward"
        return (
            "NDMinimalLoopHolonomyOperator("
            f"length={len(self._directions)}, copies={len(self._sites_by_copy)}, direction={kind})"
        )

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        out = tuple(
            shift_sites(
                state,
                sites,
                self._directions,
                q_min=self._q_min,
                q_step=self._q_step,
                local_size=self._local_size,
            )
            for sites in self._sites_by_copy
        )
        return jnp.stack(out, axis=0), jnp.ones((len(out),), dtype=self.dtype)


__all__ = ["MinimalLoopHolonomyOperator", "NDMinimalLoopHolonomyOperator"]
