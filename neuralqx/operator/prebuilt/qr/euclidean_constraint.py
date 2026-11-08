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


"""QR Euclidean constraint."""

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


class QREuclideanConstraintOperator(ComputationalOperator):
    r"""The symmetric Euclidean constraint of the three-edge QR model.

    Let $C_a$ be the bounded positive row shift on edge $a$ and define the
    diagonal factor $F_{ab|c}(q)=q_a^p q_b^p q_c^{-p}$. Inverse powers vanish
    at zero. The operator is

    $$
    H_E = \frac{N}{4\gamma^2}
    \sum_{(a,b,c)=(0,1,2),(0,2,1),(1,2,0)}
    F_{ab|c}(C_a-C_a^\dagger)(C_b-C_b^\dagger)F_{ab|c}.
    $$

    Each pair supplies four connections with signs $+,-,-,+$. The positive
    one-quarter coefficient agrees with the upstream QR convention. The
    diagonal factors on both sides make the operator Hermitian, including at
    the noncyclic cutoff boundaries. The physical choice is $p=1/4$.

    The Hilbert space must have three base edges, one gauge copy, and
    non-negative spins. The edge ordering fixes the three reduced directions.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        lapse (float): Real lapse multiplying this vertex contribution.
        power (float): Real exponent applied to the flux eigenvalue.
        immirzi (float): Nonzero real Barbero-Immirzi parameter $\gamma$.
    """

    lapse: float = field(static=True, default=1.0)
    """Real lapse multiplying this vertex contribution."""

    power: float = field(static=True, default=0.25)
    """Real exponent applied to the flux eigenvalue."""

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
        return 12

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return True

    @property
    def adjoint(self) -> QREuclideanConstraintOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"QREuclideanConstraintOperator(lapse={self.lapse}, power={self.power}, immirzi={self.immirzi})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        triplets = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
        combinations = ((+1, +1, +1.0), (+1, -1, -1.0), (-1, +1, -1.0), (-1, -1, +1.0))
        scale = jnp.asarray(
            self.lapse / (self.immirzi * self.immirzi), dtype=self.dtype
        )
        step = jnp.asarray(self._q_step, dtype=state.dtype)
        x_primes = []
        mels = []
        for a, b, z in triplets:
            right = (
                flux_eval(state[a], self.power, inverse=False)
                * flux_eval(state[b], self.power, inverse=False)
                * flux_eval(state[z], self.power, inverse=True)
            )
            for da, db, sign in combinations:
                out_a, valid_a = noncyclic_shift_site(
                    state, a, da * step, q_min=self._q_min, q_max=self._q_max
                )
                out_ab, valid_b = noncyclic_shift_site(
                    out_a, b, db * step, q_min=self._q_min, q_max=self._q_max
                )
                left = (
                    flux_eval(out_ab[a], self.power, inverse=False)
                    * flux_eval(out_ab[b], self.power, inverse=False)
                    * flux_eval(out_ab[z], self.power, inverse=True)
                )
                x_primes.append(out_ab)
                mels.append(
                    jnp.where(
                        valid_a & valid_b, 0.25 * sign * scale * left * right, 0.0
                    ).astype(self.dtype)
                )
        return jnp.stack(x_primes, axis=0), jnp.stack(mels, axis=0)


__all__ = ["QREuclideanConstraintOperator"]
