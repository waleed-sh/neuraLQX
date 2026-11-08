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


"""QR flux operator."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import validate_site
from neuralqx.utils.struct import field


def flux_eval(value: jax.Array, power: float, *, inverse: bool) -> jax.Array:
    """Evaluate a real QR flux power, using the zero-preserving inverse.

    Args:
        value: Spin eigenvalue or array of eigenvalues.
        power: Real exponent. Fractional exponents require non-negative values.
        inverse: Use inverse powers and map zero to zero when true.

    Returns:
        result (jax.Array): Real eigenvalues with the same shape as `value`.
    """
    value = value.astype(jnp.float64)
    if inverse:
        denominator = jnp.where(value == 0.0, 1.0, value)
        return jnp.where(value == 0.0, 0.0, jnp.power(1.0 / denominator, power))
    return jnp.power(value, power)


class QRFluxOperator(ComputationalOperator):
    """A diagonal power of one spin in the quantum-reduced model.

    The eigenvalue at site $s$ is $q_s^p$. With `inverse=True`, it is
    $q_s^{-p}$ for nonzero $q_s$ and zero at $q_s=0$. This zero prescription is
    the generalized inverse used in the QR constraints.

    Fractional powers require a non-negative spin range. Construct that range
    with `U1Hilbert(..., positive_qn=True)`. Integer powers can also be used
    on signed charge ranges. All eigenvalues are evaluated in real arithmetic.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        site (int): Flat state index of the spin on which this operator acts.
        power (float): Real exponent applied to the flux eigenvalue.
        inverse (bool): Whether to use inverse flux powers, with zero mapped to zero.
    """

    site: int = field(static=True)
    """Flat state index of the spin on which this operator acts."""

    power: float = field(static=True, default=1.0)
    """Real exponent applied to the flux eigenvalue."""

    inverse: bool = field(static=True, default=False)
    """Whether to use inverse flux powers, with zero mapped to zero."""

    _site: int = field(static=True, init=False, default=0, repr=False)
    """Validated flat index used by the connectivity kernel."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        if hilbert.q_min < 0 and not float(self.power).is_integer():
            raise ValueError("Fractional QR flux powers require non-negative spins.")
        object.__setattr__(
            self,
            "_site",
            validate_site(hilbert, self.site, operator=type(self).__name__),
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
    def adjoint(self) -> QRFluxOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"QRFluxOperator(site={self._site}, power={self.power}, inverse={self.inverse})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        return (
            state[None, :],
            flux_eval(state[self._site], self.power, inverse=self.inverse)[None],
        )


__all__ = ["QRFluxOperator", "flux_eval"]
