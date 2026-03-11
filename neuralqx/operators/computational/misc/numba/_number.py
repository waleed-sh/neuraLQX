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

from __future__ import annotations

from typing import Tuple
import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


class Number(ComputationalOperator):
    r"""
    Diagonal "number" operator for a single edge with U(1) DoF.

        (Number at edge=e) |σ⟩ = σ[e] · |σ⟩
    """

    def __init__(self, H, edge: int):
        super().__init__(H.hilbert_netket)

        # invariants
        self._H = H
        self.D = int(H.size)

        # validate and store target edge index
        if not (0 <= int(edge) < self.D):
            raise ValueError(
                f"{type(self).__name__}: edge index {edge} out of bounds for D={self.D}"
            )
        self._edge = int(edge)

        # diagonal operator -> exactly one connected component
        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # Normalise input shape to (B, N, D)
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        # matrix element: the quantum number at the requested edge
        vals = σ[:, :, self._edge].astype(self.dtype)

        # connected configuration is diagonal: copy σ into a single slot
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D))
        mels = vals[:, :, None]

        if single:
            return σp[0], mels[0]
        return σp, mels
