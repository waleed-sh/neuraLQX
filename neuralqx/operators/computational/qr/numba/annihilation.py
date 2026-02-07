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


class QRAnnihilation(ComputationalOperator):
    r"""
    Non-cyclic QRLG annihilation operator on a single site `site` with step `n`:

        (A_n ψ)(..., s_site, ...) = ψ(..., s_site + n*step, ...)

    in the ordered basis of local states, i.e. it lowers the local label by `n` steps.
    If the lowered value would leave the allowed range [state_min, state_max], the
    contribution is zero.

    - At most 1 connected configuration per input
    - Matrix element is 1.0 when valid, else 0
    """

    def __init__(self, H, *, site: int, n: int = 1):
        super().__init__(H.hilbert)

        self._H = H
        self.D = int(H.size)

        if not (isinstance(site, int) and 0 <= site < self.D):
            raise ValueError(f"`site` must be in [0, {self.D-1}], got {site} instead.")

        self.site = int(site)
        self.n = int(n)

        # local state lattice
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)

        # non-cyclic: lower by - n * step
        self._delta = -self.n * self._state_step

        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return False

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
        lo = self._state_min
        hi = self._state_max
        delta = self._delta

        cur = sig[:, s]
        new = cur + delta

        mask = (new >= lo) & (new <= hi)

        sig_prime = sig
        sig_prime = sig_prime.at[:, s].set(jnp.where(mask, new, cur))

        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, 1, D)).copy()
        σp = σp.at[:, :, 0, :].set(sig_prime.reshape(B, N, D))

        mels = jnp.where(
            mask.reshape(B, N),
            jnp.asarray(1.0, dtype=self.dtype),
            jnp.asarray(0.0, dtype=self.dtype),
        ).reshape(B, N, 1)

        if single:
            return σp[0], mels[0]
        return σp, mels
