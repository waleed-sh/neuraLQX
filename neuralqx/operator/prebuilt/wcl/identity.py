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


"""Identity operator."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.struct import field


class IdentityOperator(ComputationalOperator):
    r"""A scalar multiple of the identity on a discrete Hilbert space.

    Every row has one connection to itself, with matrix element `const`.
    Thus $I_c|q\rangle=c|q\rangle$. Complex coefficients are supported, and
    the adjoint conjugates the coefficient. The operator is Hermitian exactly
    when the coefficient is real.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        const (Any): Scalar multiplying the identity, which may be complex.
    """

    const: Any = field(default=1.0)
    """Scalar multiplying the identity, which may be complex."""

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return 1

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.asarray(self.const).dtype

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return bool(np.all(np.isreal(np.asarray(self.const))))

    @property
    def is_diagonal(self) -> bool:
        """Whether every nonzero matrix element connects a configuration to itself."""
        return True

    @property
    def adjoint(self) -> IdentityOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return IdentityOperator(self.hilbert, jnp.conj(self.const))

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"IdentityOperator(hilbert={self.hilbert!r}, const={self.const!r})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        return state[None, :], jnp.asarray([self.const], dtype=self.dtype)


__all__ = ["IdentityOperator"]
