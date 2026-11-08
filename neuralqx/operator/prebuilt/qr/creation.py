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


"""QR creation operator."""

from __future__ import annotations

from numbers import Integral

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import noncyclic_shift_site
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import validate_site
from neuralqx.utils.struct import field


class QRCreationOperator(ComputationalOperator):
    r"""A bounded positive shift of one QR spin in connected-state rows.

    For a local lattice spacing $h$, the nonzero row entry is

    $$
    C^{(n)}_{q,q'} = 1,\qquad q'=q+nh\,e_s.
    $$

    The entry vanishes if the target leaves the finite spin range. There is no
    wraparound and no oscillator factor such as $\sqrt{q+1}$. The creation name
    follows the QR connectivity convention in neuraLQX: it refers to the shift
    of the returned configuration. With the row convention, the corresponding
    ket action shifts in the opposite direction.

    `adjoint` returns the bounded negative shift with the same site and step
    count. `n` must be a positive integer.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        site (int): Flat state index of the spin on which this operator acts.
        n (int): Positive number of local lattice steps in the bounded shift.
    """

    site: int = field(static=True)
    """Flat state index of the spin on which this operator acts."""

    n: int = field(static=True, default=1)
    """Positive number of local lattice steps in the bounded shift."""

    _site: int = field(static=True, init=False, default=0, repr=False)
    """Validated flat index used by the connectivity kernel."""

    _q_min: int | float = field(static=True, init=False, default=0, repr=False)
    """Smallest allowed value on the local charge lattice."""

    _q_max: int | float = field(static=True, init=False, default=0, repr=False)
    """Largest allowed value on the local charge lattice."""

    _q_step: int | float = field(static=True, init=False, default=1, repr=False)
    """Spacing between adjacent allowed local charges."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        if isinstance(self.n, bool) or not isinstance(self.n, Integral) or self.n <= 0:
            raise ValueError("n must be a positive integer.")
        q_min, q_max, q_step, _local_size = q_bounds(hilbert)
        object.__setattr__(
            self,
            "_site",
            validate_site(hilbert, self.site, operator=type(self).__name__),
        )
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_max", q_max)
        object.__setattr__(self, "_q_step", q_step)
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
    def adjoint(self):
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        from .annihilation import QRAnnihilationOperator

        return QRAnnihilationOperator(self.hilbert, site=self._site, n=self.n)

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"QRCreationOperator(site={self._site}, n={self.n})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        out, valid = noncyclic_shift_site(
            state,
            self._site,
            int(self.n) * jnp.asarray(self._q_step, dtype=state.dtype),
            q_min=self._q_min,
            q_max=self._q_max,
        )
        return out[None, :], valid.astype(self.dtype)[None]


__all__ = ["QRCreationOperator"]
