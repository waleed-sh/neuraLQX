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


"""QR Lorentzian constraint."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import noncyclic_shift_site
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_single_vertex_three_edges
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.utils.struct import field

from .flux import flux_eval


class QRLorentzianConstraintOperator(ComputationalOperator):
    r"""The Lorentzian constraint of the three-edge QR model.

    For each direction $a$, let $b,c$ denote the other two directions and set

    $$
    G_a(q)=q_a^{3/8}q_b^{-1/8}q_c^{-1/8},\qquad
    D_a=G_a\left[\frac12 I-\frac14(C_a+C_a^\dagger)\right]G_a.
    $$

    The implemented constraint is

    $$
    H_L=-16\frac{1+\gamma^2}{\gamma^2}\sum_{a=0}^2 D_a^2.
    $$

    It is Hermitian and negative semidefinite for real nonzero $\gamma$.
    Inverse fluxes annihilate zero spins. Both intermediate shifts of each
    $D_a^2$ must stay inside the cutoff, even when their final displacement
    is zero. The 27 padded paths can contain repeated targets.

    Use a three-edge Hilbert space with one gauge copy and non-negative spins.
    This operator uses the upstream unit-lapse normalization.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        immirzi (float): Nonzero real Barbero-Immirzi parameter $\gamma$.
    """

    immirzi: float = field(static=True, default=1.0)
    r"""Nonzero real Barbero-Immirzi parameter $\gamma$."""

    _q_min: int | float = field(static=True, init=False, default=0, repr=False)
    """Smallest allowed value on the local charge lattice."""

    _q_max: int | float = field(static=True, init=False, default=0, repr=False)
    """Largest allowed value on the local charge lattice."""

    _q_step: int | float = field(static=True, init=False, default=1, repr=False)
    """Spacing between adjacent allowed local charges."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_single_vertex_three_edges(hilbert, operator=type(self).__name__)
        if float(self.immirzi) == 0.0:
            raise ValueError("The Immirzi parameter must be non-zero.")
        q_min, q_max, q_step, _local_size = q_bounds(hilbert)
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_max", q_max)
        object.__setattr__(self, "_q_step", q_step)
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return 27

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return True

    @property
    def adjoint(self) -> QRLorentzianConstraintOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"QRLorentzianConstraintOperator(immirzi={self.immirzi})"

    @staticmethod
    def _directional_g(
        state: jax.Array, main: int, other1: int, other2: int
    ) -> jax.Array:
        """Evaluate $q_a^{3/8}q_b^{-1/8}q_c^{-1/8}$, with inverse fluxes zero at zero."""
        return (
            flux_eval(state[main], 3.0 / 8.0, inverse=False)
            * flux_eval(state[other1], 1.0 / 8.0, inverse=True)
            * flux_eval(state[other2], 1.0 / 8.0, inverse=True)
        )

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        directions = ((0, 1, 2), (1, 0, 2), (2, 0, 1))
        deltas = ((0, 0.5), (+1, -0.25), (-1, -0.25))
        prefactor = jnp.asarray(
            (1.0 + self.immirzi * self.immirzi) / (self.immirzi * self.immirzi),
            dtype=self.dtype,
        )
        step = jnp.asarray(self._q_step, dtype=state.dtype)
        x_primes = []
        mels = []
        for main, other1, other2 in directions:
            g_right = self._directional_g(state, main, other1, other2)
            for d1, c1 in deltas:
                state_1, valid_1 = noncyclic_shift_site(
                    state, main, d1 * step, q_min=self._q_min, q_max=self._q_max
                )
                g_mid = self._directional_g(state_1, main, other1, other2)
                w1 = jnp.where(valid_1, c1 * g_mid * g_right, 0.0)
                for d2, c2 in deltas:
                    state_2, valid_2 = noncyclic_shift_site(
                        state_1, main, d2 * step, q_min=self._q_min, q_max=self._q_max
                    )
                    g_left = self._directional_g(state_2, main, other1, other2)
                    w2 = jnp.where(valid_1 & valid_2, c2 * g_left * g_mid, 0.0)
                    x_primes.append(state_2)
                    mels.append((prefactor * -16.0 * w2 * w1).astype(self.dtype))
        return jnp.stack(x_primes, axis=0), jnp.stack(mels, axis=0)


__all__ = ["QRLorentzianConstraintOperator"]
