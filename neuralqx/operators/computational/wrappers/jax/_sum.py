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

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator import ComputationalJaxOperator


@jax.jit
def _sum_pack_kernel(
    *,
    sigma: jnp.ndarray,
    xA: jnp.ndarray,
    xB: jnp.ndarray,
    mA: jnp.ndarray,
    mB: jnp.ndarray,
    sign_B: jnp.ndarray,
    tol: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Combine and pack the (A ± B) connections in pure JAX form."""

    C_A = xA.shape[2]
    C_B = xB.shape[2]
    C_total = C_A + C_B

    # concatenate connections and matrix elements
    x_all = jnp.concatenate([xA, xB], axis=2)
    m_all = jnp.concatenate([mA, sign_B * mB], axis=2)

    # mask and stable ordering
    mask = jnp.abs(m_all) > tol
    score = jnp.where(mask, 1, 0).astype(jnp.int32)
    order = jnp.argsort(-score, axis=2)

    xp_ord = jnp.take_along_axis(x_all, order[..., None], axis=2)
    m_ord = jnp.take_along_axis(m_all, order, axis=2)

    # validity/padding
    counts = jnp.sum(mask, axis=2, dtype=jnp.int32)
    idx = jnp.arange(C_total)[None, None, :]
    valid = idx < counts[..., None]

    # pad tail with σ and zeros
    xpad = jnp.broadcast_to(sigma[:, :, None, :], xp_ord.shape)
    xp_out = jnp.where(valid[..., None], xp_ord, xpad)
    mels_out = jnp.where(valid, m_ord, jnp.zeros_like(m_ord, dtype=m_ord.dtype))

    return xp_out, mels_out


@register_pytree_node_class
class SumJax(ComputationalJaxOperator):
    r"""
    JAX-native, matrix-free sum or difference of two ComputationalJaxOperators ``A`` and ``B``:

    .. math::
        S\,|\sigma\rangle =
        \begin{cases}
            (A + B)\,|\sigma\rangle, & \text{if ``subtract=False``}, \\
            (A - B)\,|\sigma\rangle, & \text{if ``subtract=True``}.
        \end{cases}

    Parameters
    ----------
    A, B : ComputationalJaxOperator
        JAX-native local operators to be combined. They must share the same Hilbert space.
    subtract : bool, optional
        If ``True``, computes ``A - B`` instead of ``A + B`` (default: ``False``).
    tol : float, optional
        Threshold for treating matrix elements as zero (default ``0.0``).
    is_hermitian : bool, optional
        Whether to mark this composite operator as Hermitian (default: ``False``).

    Notes
    -----
        - Uses the padded connection API (NetKet contract): the connection axis has a fixed size per
          call
        - No Python-side loops, pure JAX operations and pytree registration
        - Works with both single-input ``(N, D)`` and batched ``(B, N, D)`` configurations
    """

    def __init__(
        self,
        A: ComputationalJaxOperator,
        B: ComputationalJaxOperator,
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
        self._is_hermitian = bool(is_hermitian)

        # determine dtype from operands
        self._dtype = jnp.result_type(
            getattr(A, "dtype", jnp.float64),
            getattr(B, "dtype", jnp.float64),
        )

        # probe to get static connection sizes
        D = int(self.hilbert.size)
        x_probe = jnp.zeros((1, 1, D), dtype=jnp.int64)
        xA, mA = A.get_conn_padded(x_probe)
        xB, mB = B.get_conn_padded(x_probe)

        self._C_A = int(xA.shape[2])
        self._C_B = int(xB.shape[2])

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    @property
    def max_conn_size(self) -> int:
        """Padded connection size for (A ± B): sum of sizes of the components."""
        return int(self._C_A + self._C_B)

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute padded connections for ``(A ± B)`` on input ``x``."""

        # normalise shapes
        x = jnp.asarray(x, dtype=jnp.int64)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]

        # compute A and B connections
        xA, mA = self.A.get_conn_padded(x)
        xB, mB = self.B.get_conn_padded(x)

        # call jitted pack kernel
        sign_B = jnp.asarray(-1.0 if self.subtract else 1.0, dtype=self._dtype)
        tol = jnp.asarray(self._tol, dtype=self._dtype)

        xp, mels = _sum_pack_kernel(
            sigma=x,
            xA=xA,
            xB=xB,
            mA=mA.astype(self._dtype),
            mB=mB.astype(self._dtype),
            sign_B=sign_B,
            tol=tol,
        )

        if single:
            return xp[0], mels[0]
        return xp, mels

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        leaves = (self.A, self.B)
        aux = dict(
            hilbert=self.hilbert,
            subtract=self.subtract,
            tol=float(self._tol),
            is_hermitian=bool(self._is_hermitian),
            dtype_str=str(self._dtype),
            C_A=int(self._C_A),
            C_B=int(self._C_B),
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        A, B = leaves
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.A, obj.B = A, B
        obj.subtract = bool(aux["subtract"])
        obj._tol = float(aux["tol"])
        obj._is_hermitian = bool(aux["is_hermitian"])
        obj._dtype = jnp.dtype(aux["dtype_str"])
        obj._C_A = int(aux["C_A"])
        obj._C_B = int(aux["C_B"])
        return obj

    def __repr__(self):
        op = "-" if self.subtract else "+"
        return (
            f"{type(self).__name__}(A={type(self.A).__name__}, "
            f"B={type(self.B).__name__}, op='{op}', "
            f"hilbert={self.hilbert}, dtype={self.dtype})"
        )
