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


from typing import Tuple

import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


def _flux_eval(
    vals: jnp.ndarray,
    power: float,
    inverse: bool,
) -> jnp.ndarray:
    """
    Implements the QRLG flux eigenvalue evaluator, which is diagonal. If `inverse` is set to False,
    then the computed flux is `e^power`. If set to True, then the computed flux is (1/e)^power, with
    0 -> 0 (no divergence).
    """
    v = vals.astype(jnp.float32)
    p = float(power)

    if inverse:
        safe = jnp.where(v == 0.0, 0.0, jnp.power(1.0 / v, p))
        return safe
    else:
        return jnp.power(v, p)


class QRFlux(ComputationalOperator):
    r"""
    QRLG flux operator on a single site `site`:

      - Diagonal in the computational basis
      - Eigenvalue f(s_site) with
          if inverse == False:  f(e) = e^power
          if inverse == True:   f(e) = (1/e)^power, with 0 -> 0.

    Discrete representation:
      - Emits exactly one connected configuration (itself)
      - Matrix element is the diagonal eigenvalue
    """

    def __init__(
        self,
        H,
        *,
        site: int,
        power: float = 1.0,
        inverse: bool = False,
    ):
        super().__init__(H.hilbert_netket)

        self._H = H
        self.D = int(H.size)

        if not (isinstance(site, int) and 0 <= site < self.D):
            raise ValueError(f"`site` must be in [0, {self.D-1}], got {site} instead.")

        self.site = int(site)
        self.power = float(power)
        self.inverse = bool(inverse)

        # local lattice is not strictly needed for flux, but we might still want min/max
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)

        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float32

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        if D != self.D:
            raise ValueError(f"Input last dim D={D} mismatches operator D={self.D}.")

        BN = B * N
        sig = σ.reshape(BN, D).astype(jnp.int64)

        s = self.site
        vals = sig[:, s : s + 1]

        diag = _flux_eval(vals, power=self.power, inverse=self.inverse).reshape(BN)

        # σ' = σ itself, only one "connection"
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, 1, D)).copy()
        mels = diag.reshape(B, N, 1).astype(self.dtype)

        if single:
            return σp[0], mels[0]
        return σp, mels
