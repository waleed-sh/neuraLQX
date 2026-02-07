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


class Identity(ComputationalOperator):
    r"""
    Diagonal identity-like operator with an optional scalar factor `const`.

        (Identity(const=c)) |σ⟩ = c · |σ⟩

    - Exactly one connected component (diagonal).
    - Hermitian iff `const` is real.
    """

    def __init__(self, H, const=1.0):
        super().__init__(H)

        # store constant and dtype
        # if const is complex -> complex128, otherwise float64
        self._is_complex = jnp.iscomplexobj(const)
        self._dtype = jnp.complex128 if self._is_complex else jnp.float64
        self._const = jnp.asarray(const, dtype=self._dtype)

        # diagonal operator means exactly one connected component
        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        # c·I is Hermitian iff c is real
        return bool(jnp.all(jnp.imag(self._const) == 0))

    @property
    def dtype(self):
        return self._dtype

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # normalise input shape to (B, N, D)
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        # connected configuration is diagonal: copy σ into a single slot
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D))

        # matrix elements: constant c for the diagonal
        mels = jnp.full((B, N, self._nconn), self._const, dtype=self.dtype)

        if single:
            return σp[0], mels[0]
        return σp, mels
