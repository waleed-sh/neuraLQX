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


class Product(ComputationalOperator):
    r"""
    Matrix-free product of two local ComputationalOperators ``A`` (right) and ``B`` (left):

        .. math::

            (A @ B)\,|\sigma\rangle \;=\; A\big( B\,|\sigma\rangle \big).

    This class does not form dense matrices. Instead, it composes the connection graphs
    provided by ``A`` and ``B``. We:

        1. Apply ``B`` once to the input ket(s) to obtain all connected states and matrix elements
        2. For each kept branch produced by ``B``, apply ``A`` to those branch states
        3. Multiply the path matrix elements and return the final connected states

    To mitigate potential memory blow-up from too many connections, the implementation:

        - Early-prunes whole branches from ``B`` that are numerically zero for the entire batch:
          a branch index ``cb`` is kept iff ``any_{b,n} (abs(mB[b,n,cb]) > tol)``. This avoids ever
          calling ``A`` on completely useless branches
        - Stable-partitions the final connections so that all nonzeros come first along the
          connection axis, and only the tail is padded with ``(σ, 0.0)`` to preserve NetKet's padded
          contract (fixed shapes)
        - Works for both a single input ket of shape ``(N, D)`` and a batch ``(B, N, D)``. The
          single-input path simply promotes to ``(1, N, D)`` and drops the batch dimension at the
          end

    Parameters
    ----------
    A, B :
        Instances of :class:`ComputationalOperator`.
        ``A`` acts on the left, ``B`` on the right (i.e., this class represents ``A @ B``). Their
        Hilbert spaces must match
    tol : float, optional
        Numerical threshold (default: ``0.0``). Any connection with ``|mel| <= tol`` is treated as
        zero for the purposes of pruning and partitioning

    Notes
    -----
    Padded API contract.
      NetKet's operator application returns tensors with a fixed connection axis length ``C``
      for a whole call (and a whole batch). If different items/samples would produce a different
      number of connections, we must still return a uniform shape, therefore we:
        - move all nonzeros to the front,
        - pad the tail with the input configurations and zero matrix elements

      In principle downstream code should only consume the leading valid connections. You can
      compute the per-sample valid counts on the fly via:

      .. code-block:: Python

          counts = (jnp.abs(mels) > tol).sum(axis=2)  # (B, N)

      and use only ``mels[b, n, :counts[b,n]]`` and ``σp[b, n, :counts[b,n], :]``.

    Why batch shapes differ from single:
      In batch mode we keep the union of B-branches that are needed by any example in the batch.
      Therefore the connection axis length ``C`` can be larger for larger batches. For examples that
      do not require some of those branches, the corresponding slots are in the padded tail (zeros).

    Hermiticity:
      Even if ``A`` and ``B`` are Hermitian individually, the product ``A @ B`` need not be
      Hermitian. Therefore, we default to Hermiticity being False, but this can be set at
      initialisation as well.

    Examples
    --------
    Compose a Euclidean Thiemann regularised constraint for the 4D WCL model with its adjoint:

    .. code-block:: Python

        A  = ThiemannRegularisedVertexConstraint(lqx.model, vertex, apply_lapse=True, adjoint=False)
        Ad = ThiemannRegularisedVertexConstraint(lqx.model, vertex, apply_lapse=True, adjoint=True)
        HvHvdag = Product(A, Ad, tol=1e-12)

        # single input ket (N, D):
        sigma_p, mels = HvHvdag.get_conn_padded(sigma)     # (N, C, D), (N, C)
        counts = (jnp.abs(mels) > 1e-12).sum(axis=1)       # (N,)

        # batched (B, N, D):
        sigma_p_b, mels_b = HvHvdag.get_conn_padded(batch_sigma)  # (B, N, C, D), (B, N, C)
        counts_b = (jnp.abs(mels_b) > 1e-12).sum(axis=2)          # (B, N)

    Complexity
    ----------
    Let ``C_B`` be the number of connections emitted by ``B`` and ``C_A`` those of ``A``.
    After pruning keeps ``K_B <= C_B`` branches, the total connections returned are
    approximately ``C = K_B * C_A`` per input chain position. Memory scales with ``C``.

    """

    def __init__(
        self,
        A: ComputationalOperator,
        B: ComputationalOperator,
        *,
        tol: float = 0.0,
        is_hermitian: bool = False,
    ):

        if A.hilbert != B.hilbert:
            raise ValueError("Hilbert spaces of the provided operators must match.")
        super().__init__(A.hilbert)

        self.A = A
        self.B = B
        self._tol = float(tol)

        # dtype and hermiticity of the product, fallback to float64
        self._dtype = jnp.result_type(
            getattr(A, "dtype", jnp.float64),
            getattr(B, "dtype", jnp.float64),
        )
        self._is_hermitian = is_hermitian

    @property
    def dtype(self):
        """
        The dtype of matrix elements this operator produces.

        Determined by :func:`jnp.result_type(A.dtype, B.dtype)`.
        """
        return self._dtype

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Apply the product operator to one or more input configurations in padded form.

        This method preserves NetKet's padded API contract by returning a fixed-size connection axis
        (``C``) for the whole call. All true nonzero connections are placed first, any excess
        capacity is padded with the original input ket(s) and zero matrix elements.

        Returns
        -------
        σp : jnp.ndarray
            Connected configurations in padded layout.

            - If input was ``(N, D)``: shape ``(N, C, D)``.
            - If input was ``(B, N, D)``: shape ``(B, N, C, D)``.

            The first ``counts[...,]`` entries along the connection axis are valid, the remainder is
            padding (copies of the input configuration).
        mels : jnp.ndarray
            Corresponding matrix elements in padded layout.

            - If input was ``(N, D)``: shape ``(N, C)``.
            - If input was ``(B, N, D)``: shape ``(B, N, C)``.

            The first ``counts[...,]`` entries are valid, the tail entries are zeros.

        Notes
        -----
        The per-sample valid counts are not returned explicitly to conform to the base class
        signature. You can recompute them as:

        .. code-block:: Python

            counts = (jnp.abs(mels) > tol).sum(axis=2 if mels.ndim == 3 else 1)
        """

        single_input = σ.ndim == 2
        if single_input:
            # promote to (B = 1, N, D) for a unified code path drop leading dim later
            σ = σ[None, :, :]

        # compute the connected components
        σp, mels = self._kernel_batched(σ)

        if single_input:
            # remove the leading batch dimension for single-input calls
            σp = σp[0]
            mels = mels[0]

        return σp, mels

    def _kernel_batched(self, σ: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Core composition routine operating on a batch input.

        Pipeline
        --------
        1) Apply B once to the whole batch:
           ``σ --B--> (σp_B, mB)`` with shapes ``(B, N, C_B, D)``, ``(B, N, C_B)``.

        2) Early-prune B branches: keep branch index ``cb`` iff
           ``any_{b,n} (abs(mB[b,n,cb]) > tol)``. This avoids invoking ``A`` on globally zero
           branches.

        3) Apply A to each kept B-branch input: for each kept ``cb``, apply ``A`` to
           ``σp_B[..., cb, :]`` and multiply path matrix elements:
           ``m_combined = mB[..., cb][..., None] * mA``.

        4) Concatenate all per-branch results along the connection axis.

        5) Stable-partition and pad: move all nonzeros to the front, pad the tail with the input
          ket(s) and zeros to maintain fixed shapes


        :param σ: Batch input of shape ``(B, N, D)``.

        Returns
        -------
        σp_out : jnp.ndarray
            Padded connected states of shape ``(B, N, C, D)`` with all nonzeros first.
        mels_out : jnp.ndarray
            Padded matrix elements of shape ``(B, N, C)`` corresponding to ``σp_out``.

        Implementation Details
        ----------------------
        - Pruning policy is global across the batch: a B-branch is kept if any example or chain
          position needs it. With larger batches, more branches are typically kept, so the
          connection dimension ``C`` can grow accordingly.
        - Static shapes: we never change the rank, the final connection dimension ``C`` is fixed for
          the whole call.
        """

        Bsz, N, D = σ.shape

        # apply B once
        σp_B, mB = self.B._get_conn_padded_kernel(σ)
        C_B = int(σp_B.shape[2])

        # early-zero (identity-padded) shortcut: nothing to do.
        if (C_B == 1) and jnp.all(mB == 0.0):
            σp = σ[:, :, None, :]
            mels = jnp.zeros((Bsz, N, 1), dtype=self.dtype)
            return σp, mels

        # branch pruning: keep union over the whole batch
        # keep_cb[cb] = True if the branch contributes anywhere in the batch
        keep_cb = jnp.any(jnp.abs(mB) > self._tol, axis=(0, 1))
        cb_idx_all = jnp.arange(C_B)

        # turn to host ints to keep the loop small and avoid building huge tensors for dead branches
        kept_indices = list(map(int, cb_idx_all[keep_cb].tolist()))
        K_B = len(kept_indices)

        if K_B == 0:
            # all branches were globally zero
            σp = jnp.broadcast_to(σ[:, :, None, :], (Bsz, N, 1, D))
            mels = jnp.zeros((Bsz, N, 1), dtype=self.dtype)
            return σp, mels

        # apply A to each kept B-branch and combine mels
        σp_chunks = []
        m_chunks = []

        for cb in kept_indices:
            # input for A on this branch
            # (B, N, D)
            σ_branch = σp_B[:, :, cb, :]

            # apply A
            # (B, N, C_A, D), (B, N, C_A)
            σp_A, mA = self.A._get_conn_padded_kernel(σ_branch)

            # combine path mels: mB(σ->τ_cb) * mA(τ_cb->κ)
            # (B, N, 1)
            mB_cb = mB[:, :, cb][:, :, None]
            # (B, N, C_A)
            m_combined = (mB_cb * mA).astype(self.dtype)

            σp_chunks.append(σp_A)
            m_chunks.append(m_combined)

        # concatenate final connections across kept branches
        # (B, N, C, D), C = K_B * C_A
        σp_all = jnp.concatenate(σp_chunks, axis=2)
        # (B, N, C)
        m_all = jnp.concatenate(m_chunks, axis=2)

        # stable-partition nonzeros to front, pad the rest
        # (B, N, C) bool
        mask = jnp.abs(m_all) > self._tol

        # score=1 for nonzeros, 0 for zeros -> argsort(-score) keeps nonzeros first
        # (B, N, C)
        score = jnp.where(mask, 1, 0).astype(jnp.int32)
        # (B, N, C)
        order = jnp.argsort(-score, axis=2)

        # reorder connections and mels
        # (B, N, C, D)
        σp_ord = jnp.take_along_axis(σp_all, order[..., None], axis=2)
        # (B, N, C)
        m_ord = jnp.take_along_axis(m_all, order, axis=2)

        # how many valid connections per (b, n)?
        # (B, N)
        counts = jnp.sum(mask, axis=2, dtype=jnp.int32)
        C = int(σp_ord.shape[2])

        # build validity mask: first counts[b, n] entries are valid, the tail is padding
        # (1, 1, C)
        idx = jnp.arange(C)[None, None, :]
        # (B, N, C) bool
        valid = idx < counts[..., None]

        # tail padding: put back the input ket and zero mels
        # (B, N, C, D)
        σpad = jnp.broadcast_to(σ[:, :, None, :], σp_ord.shape)
        # (B, N, C, D)
        σp_out = jnp.where(valid[..., None], σp_ord, σpad)
        # (B, N, C)
        mels_out = jnp.where(valid, m_ord, jnp.zeros_like(m_ord, dtype=self.dtype))

        return σp_out, mels_out

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"(A={type(self.A).__name__}, "
            f"B={type(self.B).__name__}, "
            f"hilbert={self.hilbert}, "
            f"dtype={self.dtype})"
        )
