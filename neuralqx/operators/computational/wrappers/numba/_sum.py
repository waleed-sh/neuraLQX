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

import jax.numpy as jnp
from typing import Tuple

from neuralqx.operators.types.computational_operator import ComputationalOperator


class Sum(ComputationalOperator):
    r"""
    Matrix-free sum or difference of two local ComputationalOperators ``A`` and ``B``:

        .. math::

            S\,|\sigma\rangle =
            \begin{cases}
                (A + B)\,|\sigma\rangle, & \text{if ``subtract=False``} \\
                (A - B)\,|\sigma\rangle, & \text{if ``subtract=True``}
            \end{cases}

    This class does not construct dense matrices, it merges the connection graphs of ``A`` and ``B``
    and sums (or subtracts) their matrix elements. Padded output follows NetKet's fixed-shape
    contract.

    Parameters
    ----------
    A, B :
        Instances of :class:`ComputationalOperator`.
        Their Hilbert spaces must match.
    subtract : bool, optional
        If ``True``, computes ``A - B`` instead of ``A + B`` (default: ``False``).
    tol : float, optional
        Numerical threshold below which connections are treated as zero (default: ``0.0``).
    is_hermitian : bool, optional
        Whether to mark this composite operator as Hermitian (default: ``False``).

    Notes
    -----
        - All nonzero connections from ``A`` and ``B`` are merged and returned in padded format
        - Overlapping connections (identical connected states) are not deduplicated for speed,
          downstream contraction routines will handle that if needed
    """

    def __init__(
        self,
        A: ComputationalOperator,
        B: ComputationalOperator,
        *,
        subtract: bool = False,
        tol: float = 0.0,
        is_hermitian: bool = False,
    ):
        if A.hilbert != B.hilbert:
            raise ValueError("Hilbert spaces of the provided operators must match.")

        super().__init__(A.hilbert)

        self.A = A
        self.B = B
        self.subtract = bool(subtract)
        self._tol = float(tol)

        self._dtype = jnp.result_type(
            getattr(A, "dtype", jnp.float64),
            getattr(B, "dtype", jnp.float64),
        )
        self._is_hermitian = bool(is_hermitian)

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # normalise shapes
        single_input = σ.ndim == 2
        if single_input:
            σ = σ[None, :, :]

        # compute the connected components
        σp, mels = self._kernel_batched(σ)

        # return with shape adjusted if needed
        if single_input:
            σp, mels = σp[0], mels[0]

        return σp, mels

    def _kernel_batched(self, σ: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""Core routine computing A ± B on a batch input."""

        # apply A and B once to the same batch
        σp_A, mA = self.A._get_conn_padded_kernel(σ)
        σp_B, mB = self.B._get_conn_padded_kernel(σ)

        # sign for combining B contributions
        sign_B = -1.0 if self.subtract else +1.0

        # combine padded connections
        σp_all = jnp.concatenate([σp_A, σp_B], axis=2)
        m_all = jnp.concatenate([mA, sign_B * mB], axis=2)

        # mask for nonzero connections
        mask = jnp.abs(m_all) > self._tol
        score = jnp.where(mask, 1, 0).astype(jnp.int32)
        order = jnp.argsort(-score, axis=2)

        # reorder and pad
        σp_ord = jnp.take_along_axis(σp_all, order[..., None], axis=2)
        m_ord = jnp.take_along_axis(m_all, order, axis=2)

        counts = jnp.sum(mask, axis=2, dtype=jnp.int32)
        C = σp_ord.shape[2]

        idx = jnp.arange(C)[None, None, :]
        valid = idx < counts[..., None]

        σpad = jnp.broadcast_to(σ[:, :, None, :], σp_ord.shape)
        σp_out = jnp.where(valid[..., None], σp_ord, σpad)
        mels_out = jnp.where(valid, m_ord, jnp.zeros_like(m_ord, dtype=self.dtype))

        return σp_out, mels_out

    def __repr__(self):
        op = "+" if not self.subtract else "-"
        return (
            f"{type(self).__name__}(A={type(self.A).__name__}, "
            f"B={type(self.B).__name__}, op='{op}', hilbert={self.hilbert}, dtype={self.dtype})"
        )
