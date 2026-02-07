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
def _product_pack_kernel(
    *,
    sigma: jnp.ndarray,
    xA_all: jnp.ndarray,
    mA_all: jnp.ndarray,
    mB: jnp.ndarray,
    keep_cb: jnp.ndarray,
    tol: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Combine and pack the product connections given precomputed branch-wise outputs.

    We do:
      1) form path mels: m_combined[cb, ...] = mB[..., cb, None] * mA_all[cb, ...]
      2) zero-out skipped branches with keep_cb
      3) reshape/transpose to (B, N, C_total[, D]) with C_total = C_B * C_A.
      4) stable-partition nonzeros to the front based on |mel| > tol
      5) pad tail with input σ and zeros for mels

    All inputs are arrays to keep this function pure-jax and transformation-safe
    """

    B, N, D = sigma.shape
    C_B = xA_all.shape[0]
    C_A = xA_all.shape[3]
    C_total = C_B * C_A

    # (C_B, B, N, 1) * (C_B, B, N, C_A) gives (C_B, B, N, C_A)
    mB_exp = mB.transpose(2, 0, 1)[..., None]
    m_combined = mB_exp * mA_all

    # mask-out branches we decided to skip globally
    keep_cb_bc = keep_cb[:, None, None, None]
    m_combined = jnp.where(keep_cb_bc, m_combined, 0.0)

    # flatten cb and C_A into a single connection axis
    # xA_all: (C_B, B, N, C_A, D) -> (B, N, C_total, D)
    # m_combined: (C_B, B, N, C_A) -> (B, N, C_total)
    xA_all_flat = xA_all.transpose(1, 2, 0, 3, 4).reshape(B, N, C_total, D)
    m_all_flat = m_combined.transpose(1, 2, 0, 3).reshape(B, N, C_total)

    # partition nonzeros to the front
    # (B, N, C_total) bool
    mask = jnp.abs(m_all_flat) > tol

    # stable ordering key
    score = jnp.where(mask, 1, 0).astype(jnp.int32)

    # (B, N, C_total)
    order = jnp.argsort(-score, axis=2)

    xp_ord = jnp.take_along_axis(xA_all_flat, order[..., None], axis=2)
    m_ord = jnp.take_along_axis(m_all_flat, order, axis=2)

    # validity: first 'counts' entries are real, rest padding

    # (B, N)
    counts = jnp.sum(mask, axis=2, dtype=jnp.int32)

    # (1, 1, C_total)
    idx = jnp.arange(C_total)[None, None, :]

    # (B, N, C_total)
    valid = idx < counts[..., None]

    # pad tail with the input σ and zeros
    xpad = jnp.broadcast_to(sigma[:, :, None, :], xp_ord.shape)
    xp_out = jnp.where(valid[..., None], xp_ord, xpad)
    mels_out = jnp.where(valid, m_ord, jnp.zeros_like(m_ord, dtype=m_ord.dtype))

    return xp_out, mels_out


@register_pytree_node_class
class ProductJax(ComputationalJaxOperator):
    r"""
    JAX-native, matrix-free product of two padded local operators ``A`` (left) and ``B`` (right):

    .. math::
        (A @ B)\,|\sigma\rangle \;=\; A\!\left( B\,|\sigma\rangle \right).

    The class composes the connection graphs without building dense matrices. What it does is:

    1. Apply ``B`` once to the input(s): obtain ``(σ'_B, m_B)`` of shape ``(B, N, C_B, D)`` and
       ``(B, N, C_B)``
    2. For each branch index ``cb`` of ``B``:
       - If that branch is globally zero over the batch (``\max_{b,n}|m_B[b,n,cb]| \le tol``),
         skip it (we will inject zeros in the padded tail).
       - Else apply ``A`` to ``σ'_B[..., cb, :]`` and combine path matrix elements:
         ``m_{path} = m_B[..., cb, None] * m_A``.
    3. Concatenate across kept branches, stable-partition so that all nonzero connections come
       first, and pad the tail with the input state(s) and zeros

    Parameters
    ----------
    A, B : ComputationalJaxOperator
        JAX-native operators to be composed as ``A @ B``. They must share the same Hilbert space.
        Their internal implementations may themselves use jitted kernels and PyTree registration.
    tol : float, optional
        Threshold for treating matrix elements as zero (default ``0.0``). This affects pruning and
        the stable partitioning (which connections are considered "valid")
    is_hermitian : bool, optional
        If you know ``A @ B`` is Hermitian (e.g., ``A=H``, ``B=H`` or ``B=A†`` and
        the product is provably Hermitian), set this to ``True`` to select covariance
        gradients downstream

    Notes
    -----
    - Padded contract: The connection axis has a fixed size per call. All nonzero connections are
      stably moved to the front. The tail contains copies of the input state with zero matrix
      element. Downstream Monte-Carlo code should only consume the first
      ``counts[b,n] = (|mels[b,n,:]|>tol).sum()`` connections

    - Batch behavior: Pruning is done with a global keep mask over the batch to avoid data-dependent
      recompiles, larger batches may retain more branches (others are padded)

    - JAX transforms: The heavy lifting (packing/partitioning) is a single, stateless
      :func:`jax.jit` kernel taking only arrays, there are no Python-side loops or dynamic slices
      inside jit

    Compose a Euclidean Thiemann regularised constraint for the 4D WCL model with its adjoint:

    .. code-block:: Python

        A  = ThiemannRegularisedVertexConstraintJax(lqx.model, vertex, apply_lapse=True, adjoint=False)
        Ad = ThiemannRegularisedVertexConstraintJax(lqx.model, vertex, apply_lapse=True, adjoint=True)
        HvHvdag = ProductJax(A, Ad, tol=1e-12)

        # single input ket (N, D):
        sigma_p, mels = HvHvdag.get_conn_padded(sigma)     # (N, C, D), (N, C)
        counts = (jnp.abs(mels) > 1e-12).sum(axis=1)       # (N,)

        # batched (B, N, D):
        sigma_p_b, mels_b = HvHvdag.get_conn_padded(batch_sigma)  # (B, N, C, D), (B, N, C)
        counts_b = (jnp.abs(mels_b) > 1e-12).sum(axis=2)          # (B, N)

    """

    def __init__(
        self,
        A: ComputationalJaxOperator,
        B: ComputationalJaxOperator,
        *,
        tol: float = 0.0,
        is_hermitian: bool = False,
    ):
        if A.hilbert != B.hilbert:
            raise ValueError("Hilbert spaces of the provided operators must match.")
        super().__init__(A.hilbert)

        # dev: this insanity is because NetKet uses bra/row convention for matrix elements
        self.A = A
        self.B = B
        self._tol = float(tol)
        self._is_hermitian = bool(is_hermitian)

        # prefer operators to define dtype, fall back to float64
        self._dtype = jnp.result_type(
            getattr(A, "dtype", jnp.float64),
            getattr(B, "dtype", jnp.float64),
        )

        # cache padded connection sizes once (stable shapes for the packer kernel)
        # we probe with a zero-like single sample of the correct D
        D = int(self.hilbert.size)
        x_probe = jnp.zeros((1, 1, D), dtype=jnp.int64)
        xB, mB = self.B.get_conn_padded(x_probe)
        xA, mA = self.A.get_conn_padded(x_probe)

        self._C_B = int(xB.shape[2])
        self._C_A = int(xA.shape[2])

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    @property
    def max_conn_size(self) -> int:
        """
        The padded connection size reported to chunking logic.

        For a product, this is the product of the factor sizes.
        """
        return int(self._C_A * self._C_B)

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute padded connections for ``(A @ B)`` on input ``x``.
        """

        # normalise dimensions
        x = jnp.asarray(x, dtype=jnp.int64)
        single = x.ndim == 2

        if single:
            x = x[None, :, :]
        Bsz, N, D = x.shape

        # apply B once on the full batch
        xB, mB = self.B.get_conn_padded(x)
        C_B = int(xB.shape[2])
        C_B = int(xB.shape[2])
        C_total = int(self._C_A * C_B)

        pred = jnp.logical_and(jnp.array(C_B == 1), jnp.all(mB == 0.0))

        def _early(_):
            # pad to general shape so both branches match
            xp = jnp.broadcast_to(x[:, :, None, :], (Bsz, N, C_total, D))
            m = jnp.zeros((Bsz, N, C_total), dtype=self.dtype)
            return (xp, m)

        def _cont(_):
            # existing body: build xA_all/mA_all, run packer, return (xp, mels)
            C_A = int(self._C_A)

            def body(cb, carry):
                xA_all, mA_all = carry
                x_branch = xB[:, :, cb, :]
                xA, mA = self.A.get_conn_padded(x_branch)
                # C_A' must equal self._C_A thanks to the init fix above
                xA_all = xA_all.at[cb].set(xA)
                mA_all = mA_all.at[cb].set(mA.astype(self.dtype))
                return xA_all, mA_all

            xA_all = jnp.zeros((C_B, Bsz, N, C_A, D), dtype=x.dtype)
            mA_all = jnp.zeros((C_B, Bsz, N, C_A), dtype=self.dtype)
            xA_all, mA_all = jax.lax.fori_loop(0, C_B, body, (xA_all, mA_all))

            xp, mels = _product_pack_kernel(
                sigma=x,
                xA_all=xA_all,
                mA_all=mA_all,
                mB=mB.astype(self.dtype),
                keep_cb=jnp.any(jnp.abs(mB) > self._tol, axis=(0, 1)),
                tol=jnp.asarray(self._tol, dtype=self.dtype),
            )
            return (xp, mels)

        xp, mels = jax.lax.cond(pred, _early, _cont, operand=None)
        return (xp[0], mels[0]) if single else (xp, mels)

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        """
        Put operator factors as leaves so nested pytrees (so all ComputationalJaxOperators)
        remain transform-friendly. Keep small metadata in the aux dict to avoid excessive recompiles
        """

        leaves = (self.A, self.B)
        aux = dict(
            hilbert=self.hilbert,
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

        # initialise base
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])

        # restore factors
        obj.A, obj.B = A, B

        # restore metadata
        obj._tol = float(aux["tol"])
        obj._is_hermitian = bool(aux["is_hermitian"])
        obj._dtype = jnp.dtype(aux["dtype_str"])
        obj._C_A = int(aux["C_A"])
        obj._C_B = int(aux["C_B"])

        return obj

    def __repr__(self):
        return f"{type(self).__name__}(A={type(self.A).__name__}, B={type(self.B).__name__}, hilbert={self.hilbert}, dtype={self.dtype})"
