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

from neuralqx.operators.types import ComputationalOperator


def _flux_eval(
    vals: jnp.ndarray,
    *,
    power: float,
    inverse: bool,
) -> jnp.ndarray:
    """
    Diagonal QRLG flux evaluator.

    If ``inverse`` is False:
        f(v) = v^power

    If ``inverse`` is True:
        f(v) = (1/v)^power with the convention 0 -> 0.
    """

    v = vals.astype(jnp.float64)
    if inverse:
        return jnp.where(v == 0.0, 0.0, jnp.power(1.0 / v, power))
    return jnp.power(v, power)


def _in_bounds(x: jnp.ndarray, lo: int, hi: int) -> jnp.ndarray:
    return (x >= lo) & (x <= hi)


def _directional_g_factor(
    sig: jnp.ndarray,
    *,
    main: int,
    other1: int,
    other2: int,
) -> jnp.ndarray:
    r"""
    Evaluate the diagonal prefactor ``F_dir^(1/4)`` for one direction.

    For direction ``main``:

      F_main = E(main)^(3/2) * E(other1)^(-1/2) * E(other2)^(-1/2)

    and therefore:

      F_main^(1/4) = E(main)^(3/8) * E(other1)^(-1/8) * E(other2)^(-1/8).
    """

    v_main = sig[:, main : main + 1]
    v_o1 = sig[:, other1 : other1 + 1]
    v_o2 = sig[:, other2 : other2 + 1]

    return (
        _flux_eval(v_main, power=3.0 / 8.0, inverse=False)
        * _flux_eval(v_o1, power=1.0 / 8.0, inverse=True)
        * _flux_eval(v_o2, power=1.0 / 8.0, inverse=True)
    ).reshape(sig.shape[0])


class LorentzianConstraint(ComputationalOperator):
    r"""
    Quantum-Reduced LQG Lorentzian constraint on a single-vertex, 3-edge graph.

    With edge labels ``(x, y, z) = (0, 1, 2)``:

      H_L = -16 * (D_x^2 + D_y^2 + D_z^2)

    where for each direction ``d in {x,y,z}``:

      D_d = F_d^(1/4) * (1/2) * (1 - c(d)) * F_d^(1/4),

      F_x = E(x)^(3/2) * E(y)^(-1/2) * E(z)^(-1/2),
      F_y = E(y)^(3/2) * E(x)^(-1/2) * E(z)^(-1/2),
      F_z = E(z)^(3/2) * E(x)^(-1/2) * E(y)^(-1/2).

    Implementation notes:
      - ``c(d) = (1/2) * (C_d + A_d)`` with non-cyclic single-step shifts.
      - Per direction, ``D_d`` has three one-step channels:
          diag (delta=0, coeff=+1/2),
          raise (delta=+1, coeff=-1/4),
          lower (delta=-1, coeff=-1/4).
      - ``D_d^2`` is evaluated explicitly by summing all 3x3 path combinations.
      - Total padded connection count is fixed to ``3 directions * 9 paths = 27``.
    """

    def __init__(self, H, *, immirzi: float = 1.0):
        super().__init__(H.hilbert_netket)

        self._H = H
        self.D = int(H.size)

        n_edges_total = int(H.tiny_size)
        if n_edges_total != 3:
            raise ValueError(
                f"{type(self).__name__} requires a Single-Vertex graph with 3 edges, "
                f"got {n_edges_total} edges."
            )

        if self.D < 3:
            raise ValueError(f"Unexpected state dimension D={self.D} (< 3).")

        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)
        self.immirzi = float(immirzi)
        if self.immirzi == 0.0:
            raise ValueError("The Immirzi parameter must be non-zero.")

        # For each direction: (main, other1, other2)
        self._directions = ((0, 1, 2), (1, 0, 2), (2, 0, 1))

        # (1/2) * (1 - c) = +1/2 * I - 1/4 * C - 1/4 * A
        self._step_deltas = (0, +1, -1)
        self._step_coeffs = (+0.5, -0.25, -0.25)

        # 3 directions * (3 first-step channels * 3 second-step channels)
        self._nconn = 27

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def adjoint(self) -> "LorentzianConstraint":
        """The Lorentzian constraint is Hermitian, so its adjoint is itself."""
        return self

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, sigma: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sigma = jnp.asarray(sigma, dtype=jnp.int64)

        single = sigma.ndim == 2
        if single:
            sigma = sigma[None, :, :]

        B, N, D = sigma.shape
        if D != self.D:
            raise ValueError(f"Input last dim D={D} mismatches operator D={self.D}.")

        BN = B * N
        sig = sigma.reshape(BN, D).astype(jnp.int64)

        K = self._nconn
        sigma_p = jnp.broadcast_to(sigma[:, :, None, :], (B, N, K, D)).copy()
        mels = jnp.zeros((B, N, K), dtype=self.dtype)

        lo = self._state_min
        hi = self._state_max
        step = self._state_step
        gamma2 = self.immirzi * self.immirzi
        lorentz_prefactor = jnp.asarray((1.0 + gamma2) / gamma2, dtype=self.dtype)

        k = 0

        for main, other1, other2 in self._directions:
            v_main = sig[:, main : main + 1]
            g_right = _directional_g_factor(
                sig,
                main=main,
                other1=other1,
                other2=other2,
            )

            for d1, c1 in zip(self._step_deltas, self._step_coeffs):
                cand_1 = v_main + (d1 * step)
                mask_1 = _in_bounds(cand_1, lo, hi).reshape(BN)

                sig_1 = sig.at[:, main : main + 1].set(
                    jnp.where(mask_1[:, None], cand_1, v_main)
                )
                g_mid = _directional_g_factor(
                    sig_1,
                    main=main,
                    other1=other1,
                    other2=other2,
                )
                d1_w = jnp.where(
                    mask_1,
                    (c1 * g_mid * g_right).astype(self.dtype),
                    jnp.asarray(0.0, dtype=self.dtype),
                )

                v_1 = sig_1[:, main : main + 1]

                for d2, c2 in zip(self._step_deltas, self._step_coeffs):
                    cand_2 = v_1 + (d2 * step)
                    mask_2 = mask_1 & _in_bounds(cand_2, lo, hi).reshape(BN)

                    sig_2 = sig_1.at[:, main : main + 1].set(
                        jnp.where(mask_2[:, None], cand_2, v_1)
                    )
                    g_left = _directional_g_factor(
                        sig_2,
                        main=main,
                        other1=other1,
                        other2=other2,
                    )
                    d2_w = jnp.where(
                        mask_2,
                        (c2 * g_left * g_mid).astype(self.dtype),
                        jnp.asarray(0.0, dtype=self.dtype),
                    )

                    # H_L contribution from this path in this direction:
                    #   -16 * D_d(σ' <- τ) * D_d(τ <- σ)
                    w = (lorentz_prefactor * (-16.0 * d2_w * d1_w)).astype(self.dtype)

                    sigma_p = sigma_p.at[:, :, k, :].set(sig_2.reshape(B, N, D))
                    mels = mels.at[:, :, k].set(w.reshape(B, N))
                    k += 1

        if single:
            return sigma_p[0], mels[0]
        return sigma_p, mels
