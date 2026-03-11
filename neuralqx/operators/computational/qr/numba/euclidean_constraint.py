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
    Implements the QRLG flux eigenvalue evaluator, which is diagonal. If `inverse` is set to False,
    then the computed flux is `e^power`. If set to True, then the computed flux is (1/e)^power, with
    0 -> 0 (no divergence).
    """

    v = vals.astype(jnp.float64)
    if inverse:
        # 0 -> 0, else (1/v)^power
        safe = jnp.where(v == 0.0, 0.0, jnp.power(1.0 / v, power))
        return safe
    else:
        return jnp.power(v, power)


def _in_bounds(x: jnp.ndarray, lo: int, hi: int) -> jnp.ndarray:
    """Boolean mask for lo <= x <= hi."""
    return (x >= lo) & (x <= hi)


class EuclideanConstraint(ComputationalOperator):
    r"""
    Quantum-Reduced LQG Euclidean constraint on a Single-Vertex-Graph with 3 edges. The convention
    here is (x = 0, y = 1, z = 2).

    Definition reproduced in Discrete form:

        C_E = lapse * (t1 + t2 + t3)

    with, for (a,b,z) being a permutation of (0,1,2),

        t(a,b|z) = F(a,b|z) * s(a) * s(b) * F(a,b|z),

    where
        - s(e) = (1/2) * (C_e - A_e), non-cyclic single-step raise/lower
        - F(a,b|z) = E(a)^(1/4) * E(b)^(1/4) * [E(z)]^(-1/4)  (E_inv on z)
        - C_e tries to add +step, A_e tries to add -step, invalid moves give zero contrib

    Action:
        For each input configuration σ, this operator emits up to 12 connected configurations:
          4 from each pair (a,b) ∈ {(0,1), (0,2), (1,2)} with weights:
              (1/4) * sign * lapse * F(σ') * F(σ),
          where sign ∈ {+1,-1,-1,+1} corresponds to (+,+), (+,-), (-,+), (-,-).

    Notes:
      - Hermitian (weights are real scalars, left/right diagonal prefactors applied).
      - dtype: float64
      - Non-cyclic: attempts to step outside [state_min, state_max] are discarded (weight 0).
    """

    def __init__(self, H, *, lapse: float = 1.0, power: float = 0.25):
        super().__init__(H.hilbert_netket)

        # host references/invariants
        self._H = H
        self.D = int(H.size)

        # this operator is defined for a single-vertex, 3-edge model
        n_edges_total = int(H.tiny_size)
        if n_edges_total != 3:
            raise ValueError(
                f"{type(self).__name__} requires a Single-Vertex graph with 3 edges, "
                f"got {n_edges_total} edges."
            )

        if self.D < 3:
            raise ValueError(f"Unexpected state dimension D={self.D} (< 3).")

        # lattice of allowed local states (assumed uniform step)
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)

        # physical parameters
        self.lapse = float(lapse)
        self._flux_power = float(power)

        # max number of off-diagonal connections: 3 terms * 4 combos = 12
        self._nconn = 12

        # fixed edge triplets for the three terms:
        # (a, b, z) with z carrying the inverse flux in the prefactors
        self._triplets = ((0, 1, 2), (0, 2, 1), (1, 2, 0))

        # signs for (C_a ± A_a) * (C_b ± A_b)
        # order: (+,+), (+,-), (-,+), (-,-)
        self._comb_signs = (+1.0, -1.0, -1.0, +1.0)

        # deltas (in units of state_step) applied to (a, b) for each combo above
        self._comb_deltas = (
            (+1, +1),
            (+1, -1),
            (-1, +1),
            (-1, -1),
        )

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # normalise shape to (B, N, D)
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape
        if D != self.D:
            raise ValueError(f"Input last dim D={D} mismatches operator D={self.D}.")

        BN = B * N
        sig = σ.reshape(BN, D).astype(jnp.int64)

        # broadcast initial σ across connection axis
        K = self._nconn
        sigmas_p = jnp.broadcast_to(σ[:, :, None, :], (B, N, K, D)).copy()
        mels = jnp.zeros((B, N, K), dtype=self.dtype)

        # flux power and lapse factor (constant scalar)
        p = self._flux_power
        lapse = jnp.asarray(self.lapse, dtype=self.dtype)

        # convenience: bounds and step
        lo = self._state_min
        hi = self._state_max
        step = self._state_step

        # running connection index
        k = 0

        # iterate over the three (a, b|z) terms
        for a, b, z in self._triplets:
            # right diagonal prefactor F_right(σ) = E(a)^p * E(b)^p * (E(z))^{-p}
            va = sig[:, a : a + 1]
            vb = sig[:, b : b + 1]
            vz = sig[:, z : z + 1]

            F_right = (
                _flux_eval(va, power=p, inverse=False)
                * _flux_eval(vb, power=p, inverse=False)
                * _flux_eval(vz, power=p, inverse=True)
            ).reshape(BN)

            # 4 symmetric-holonomy combos for s(a) * s(b)
            for (da, db), sign in zip(self._comb_deltas, self._comb_signs):
                # proposed new values
                new_a = va + (da * step)
                new_b = vb + (db * step)

                # validity mask (non-cyclic)
                mask = (_in_bounds(new_a, lo, hi) & _in_bounds(new_b, lo, hi)).reshape(
                    BN
                )

                # build σ' by updating only a and b (z unchanged), (BN, D) view
                sig_k = sig

                # write into a copy to avoid aliasing
                sig_k = sig_k.at[:, a : a + 1].set(jnp.where(mask[:, None], new_a, va))
                sig_k = sig_k.at[:, b : b + 1].set(jnp.where(mask[:, None], new_b, vb))

                # left diagonal prefactor F_left(σ') with E_inv on z (z unchanged by s(a)s(b))
                va_p = sig_k[:, a : a + 1]
                vb_p = sig_k[:, b : b + 1]

                F_left = (
                    _flux_eval(va_p, power=p, inverse=False)
                    * _flux_eval(vb_p, power=p, inverse=False)
                    * _flux_eval(vz, power=p, inverse=True)
                ).reshape(BN)

                # matrix element:
                #   (1/4) * sign * lapse * F_left(σ') * F_right(σ), masked for validity
                w = ((0.25 * sign) * lapse * F_left * F_right).astype(self.dtype)

                w = jnp.where(mask, w, jnp.asarray(0.0, dtype=self.dtype))

                # scatter into the k-th connection slot
                # reshape back to (B, N, D) and (B, N)
                sig_k_BND = sig_k.reshape(B, N, D)
                sigmas_p = sigmas_p.at[:, :, k, :].set(sig_k_BND)
                mels = mels.at[:, :, k].set(w.reshape(B, N))

                # advance connection slot
                k += 1

        # return with original rank if input was (N, D)
        if single:
            return sigmas_p[0], mels[0]
        return sigmas_p, mels
