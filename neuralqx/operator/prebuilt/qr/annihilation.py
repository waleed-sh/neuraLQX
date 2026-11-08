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


"""QR annihilation operator."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from neuralqx.operator.prebuilt.utils import noncyclic_shift_site

from .creation import QRCreationOperator


class QRAnnihilationOperator(QRCreationOperator):
    r"""A bounded negative shift of one QR spin in connected-state rows.

    This is the transpose of `QRCreationOperator`. Its nonzero row entry is
    $C^{(n)\dagger}_{q,q-nh e_s}=1$ whenever the target lies in the local spin
    range. At the lower boundary it returns a zero-amplitude padding slot.

    The shift has unit amplitude. The annihilation name describes the returned
    configuration, using the same row convention as the creation operator.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        site (int): Flat state index of the spin on which this operator acts.
        n (int): Positive number of local lattice steps in the bounded shift.
    """

    @property
    def adjoint(self) -> QRCreationOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return QRCreationOperator(self.hilbert, site=self._site, n=self.n)

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"QRAnnihilationOperator(site={self._site}, n={self.n})"

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
            -int(self.n) * jnp.asarray(self._q_step, dtype=state.dtype),
            q_min=self._q_min,
            q_max=self._q_max,
        )
        return out[None, :], valid.astype(self.dtype)[None]


__all__ = ["QRAnnihilationOperator"]
