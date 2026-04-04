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
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import lax

from neuralqx.graph import HalfLadderGraph
from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


def _wrap_mod(
    vals: jnp.ndarray, delta: jnp.ndarray, mod_span: jnp.ndarray, state_min: jnp.ndarray
) -> jnp.ndarray:
    return ((vals + delta - state_min) % mod_span) + state_min


def _sgn_half_delta(mu: jnp.ndarray, half_delta: jnp.ndarray) -> jnp.ndarray:
    one = half_delta.astype(jnp.float64)
    mu_f = mu.astype(jnp.float64)
    return jnp.where(
        mu_f >= one, 1.0, jnp.where(mu_f <= -one, -1.0, mu_f / jnp.maximum(one, 1e-12))
    )


def _gamma_phi_full(
    K_left: jnp.ndarray,
    K_center: jnp.ndarray,
    K_right: jnp.ndarray,
    mu_center: jnp.ndarray,
    mu_minus: jnp.ndarray,
    mu_plus: jnp.ndarray,
) -> jnp.ndarray:
    """
    Full Γ_φ(v) with one-sided E^x' and link-averaged μ.

        E^x = K/2
        E_+^x'(v) = E^x(v+1) - E^x(v)
        E_-^x'(v) = E^x(v) - E^x(v-1)
        int_+ E^φ(v) = (μ(v) + μ(v+1))/2
        int_- E^φ(v) = (μ(v) + μ(v-1))/2
        Γ_φ(v) = [ -E_+^x'(v) / int_+E^φ(v)   +   E_-^x'(v) / int_-E^φ(v) ] / 4
    """
    fd = jnp.float64
    Ex_m = 0.5 * K_left.astype(fd)
    Ex_c = 0.5 * K_center.astype(fd)
    Ex_p = 0.5 * K_right.astype(fd)

    Ex_prime_plus = Ex_p - Ex_c
    Ex_prime_minus = Ex_c - Ex_m

    mu_c = jnp.abs(mu_center).astype(fd)
    mu_p = jnp.abs(mu_plus).astype(fd)
    mu_m = jnp.abs(mu_minus).astype(fd)

    int_plus = 0.5 * (mu_c + mu_p)
    int_minus = 0.5 * (mu_c + mu_m)

    term_plus = jnp.where(int_plus > 0.0, -Ex_prime_plus / int_plus, 0.0)
    term_minus = jnp.where(int_minus > 0.0, Ex_prime_minus / int_minus, 0.0)

    return 0.25 * (term_plus + term_minus)


@jax.jit
def _sv_bs_kernel_adjoint(
    sigma: jnp.ndarray,
    idx_mu: jnp.ndarray,
    idx_mu_m: jnp.ndarray,
    idx_mu_p: jnp.ndarray,
    idx_mu_mm: jnp.ndarray,
    idx_mu_pp: jnp.ndarray,
    idx_km: jnp.ndarray,
    idx_kp: jnp.ndarray,
    idx_kmm: jnp.ndarray,
    idx_kpp: jnp.ndarray,
    idx_kppp: jnp.ndarray,
    idx_kmmm: jnp.ndarray,
    use_gamma: jnp.ndarray,
    delta_i: jnp.ndarray,
    half_delta_i: jnp.ndarray,
    outer_km_c: jnp.ndarray,
    outer_kp_c: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_min: jnp.ndarray,
    immirzi: jnp.ndarray,
):
    D = sigma.shape[-1]
    sig = sigma.reshape((-1, D))
    M = sig.shape[0]

    take = partial(jnp.take, axis=-1)
    as1 = lambda x: x.reshape(M)

    imc = idx_mu
    imm = idx_mu_m
    imp = idx_mu_p
    ikm = idx_km
    ikp = idx_kp

    mu_c = as1(take(sig, imc))
    mu_m = as1(take(sig, imm))
    mu_p = as1(take(sig, imp))
    k_m = as1(take(sig, ikm))
    k_p = as1(take(sig, ikp))

    def _gather_opt(idx):
        idx_eff = jnp.where(idx >= 0, idx, jnp.int64(0))
        val = as1(take(sig, idx_eff))
        return jnp.where(idx >= 0, val, jnp.int64(0))

    mu_mm = _gather_opt(idx_mu_mm)
    mu_pp = _gather_opt(idx_mu_pp)

    #
    #
    #   K and derived sqrt(|K±1|)
    K = (k_m + k_p).astype(jnp.int64)
    sqrtK = jnp.sqrt(jnp.abs(K).astype(jnp.float64))
    sqrtKp1 = jnp.sqrt(jnp.abs((K + 1)).astype(jnp.float64))
    sqrtKm1 = jnp.sqrt(jnp.abs((K - 1)).astype(jnp.float64))

    #
    #
    #   ΔxV(v) = |μ(v)| [√|K+1| − √|K−1|]
    abs_mu = jnp.abs(mu_c).astype(jnp.float64)
    delta_xV = abs_mu * (sqrtKp1 - sqrtKm1)

    #
    #
    #   ΔφV(v) = sgn_{δ/2}(μ(v)) √|K(v)|
    sgn_half = _sgn_half_delta(mu_c, half_delta_i)
    delta_phiV = sgn_half * sqrtK

    #
    #
    #   Γ and Γ′
    def _with_gamma():
        #
        #
        #   K(v-1) = k_--(v) + k_-(v)  or  k_-(v) + outer_km_c
        kmm_val = _gather_opt(idx_kmm)
        left_outer_term = jnp.where(idx_kmm >= 0, jnp.int64(0), outer_km_c)
        K_left = (kmm_val + k_m + left_outer_term).astype(jnp.int64)

        #
        #
        #   K(v) = k_-(v) + k_+(v)
        K_center = (k_m + k_p).astype(jnp.int64)

        #
        #
        #   K(v+1) = k_+(v) + k_++(v)  or  k_+(v) + outer_kp_c
        kpp_val = _gather_opt(idx_kpp)
        right_outer_term = jnp.where(idx_kpp >= 0, jnp.int64(0), outer_kp_c)
        K_right = (k_p + kpp_val + right_outer_term).astype(jnp.int64)

        #
        #
        #   Γ_φ(v)
        gamma = _gamma_phi_full(
            K_left=K_left,
            K_center=K_center,
            K_right=K_right,
            mu_center=mu_c,
            mu_minus=mu_m,
            mu_plus=mu_p,
        )
        gamma2 = gamma * gamma

        #
        #
        #   For Γ′_φ(v) = Γ_φ(v+1) - Γ_φ(v)
        #   K(v+2) = k_++(v) + k_+++(v) with boundary fallbacks
        kp_contrib = jnp.where(idx_kpp >= 0, kpp_val, outer_kp_c)
        kppp_val = _gather_opt(idx_kppp)
        kpp_contrib = jnp.where(idx_kppp >= 0, kppp_val, outer_kp_c)
        K_right_right = (kp_contrib + kpp_contrib).astype(jnp.int64)

        gamma_vp1 = _gamma_phi_full(
            K_left=K_center,  # E^x(v)
            K_center=K_right,  # E^x(v+1)
            K_right=K_right_right,  # E^x(v+2)
            mu_center=mu_p,  # μ(v+1)
            mu_minus=mu_c,  # μ(v)
            mu_plus=mu_pp,  # μ(v+2) (0 if absent)
        )
        gamma_prime = gamma_vp1 - gamma
        return gamma, gamma2, gamma_prime

    def _no_gamma():
        z = jnp.zeros_like(delta_xV)
        return z, z, z

    gamma, gamma2, gamma_prime = lax.cond(use_gamma, _with_gamma, _no_gamma)

    conn = 19
    sigma_p = jnp.broadcast_to(sigma[..., None, :], sigma.shape[:-1] + (conn, D)).copy()
    mels = jnp.zeros(sigma.shape[:-1] + (conn,), dtype=jnp.float64)

    def apply_shifts(
        base: jnp.ndarray, cols: jnp.ndarray, deltas: jnp.ndarray
    ) -> jnp.ndarray:
        if cols.size == 0:
            return base
        if base.ndim == 2:
            cur = base[:, cols]
            upd = _wrap_mod(cur, deltas, mod_span, state_min).astype(base.dtype)
            return base.at[:, cols].set(upd)
        else:
            cur = base[:, :, cols]
            upd = _wrap_mod(cur, deltas, mod_span, state_min).astype(base.dtype)
            return base.at[:, :, cols].set(upd)

    cols_mu = jnp.reshape(idx_mu, (1,)).astype(jnp.int64)
    cols_kp = jnp.reshape(idx_kp, (1,)).astype(jnp.int64)
    cols_km = jnp.reshape(idx_km, (1,)).astype(jnp.int64)
    cols_musR = jnp.stack([idx_mu.astype(jnp.int64), idx_mu_p.astype(jnp.int64)])
    cols_musL = jnp.stack([idx_mu_m.astype(jnp.int64), idx_mu.astype(jnp.int64)])

    #
    #
    #   H_C off-diagonals
    del_mu_p = jnp.full((M, 1), 2 * delta_i, dtype=jnp.int64)
    del_mu_m = jnp.full((M, 1), -2 * delta_i, dtype=jnp.int64)

    sig_mu_p = apply_shifts(sig, cols_mu, del_mu_p)
    sig_mu_m = apply_shifts(sig, cols_mu, del_mu_m)

    sigma_p = sigma_p.at[..., 0, :].set(sig_mu_p.reshape(sigma.shape[:-1] + (D,)))
    sigma_p = sigma_p.at[..., 1, :].set(sig_mu_m.reshape(sigma.shape[:-1] + (D,)))
    mels = mels.at[..., 0].set(delta_xV.reshape(sigma.shape[:-1]))
    mels = mels.at[..., 1].set(delta_xV.reshape(sigma.shape[:-1]))

    # diagonal slot 2
    d64 = delta_i.astype(jnp.float64)
    diag = (
        -2.0 * delta_xV
        - 4.0 * (immirzi**2) * (d64**2) * (1.0 - gamma2) * delta_xV
        - 4.0 * (immirzi**2) * (d64**2) * gamma_prime * delta_phiV
    )
    sigma_p = sigma_p.at[..., 2, :].set(sigma)
    mels = mels.at[..., 2].set(diag.reshape(sigma.shape[:-1]))

    #
    #
    #   H_R: 8 terms on (k_+, μ_c, μ_p)
    dk = jnp.asarray([+1, +1, +1, +1, -1, -1, -1, -1], jnp.int64) * delta_i
    dmu_c = jnp.asarray([+1, +1, -1, -1, +1, +1, -1, -1], jnp.int64) * half_delta_i
    dmu_p = jnp.asarray([+1, -1, +1, -1, +1, -1, +1, -1], jnp.int64) * half_delta_i
    sgn_R = jnp.asarray([+1, -1, +1, -1, -1, +1, -1, +1], jnp.float64)

    base_R = jnp.broadcast_to(sig[:, None, :], (M, 8, D))
    del_k_R = dk.reshape(1, 8, 1).repeat(M, 0)
    base_R_k = apply_shifts(base_R, cols_kp, del_k_R)
    del_mus_R = jnp.stack(
        [dmu_c.reshape(1, 8).repeat(M, 0), dmu_p.reshape(1, 8).repeat(M, 0)],
        axis=-1,
    )
    sig_R = apply_shifts(base_R_k, cols_musR, del_mus_R)
    sigma_p = sigma_p.at[..., 3:11, :].set(sig_R.reshape(sigma.shape[:-1] + (8, D)))
    mels = mels.at[..., 3:11].set(delta_phiV.reshape(sigma.shape[:-1] + (1,)) * sgn_R)

    #
    #
    #   H_L: 8 mirror terms on (k_-, μ_m, μ_c)
    base_L = jnp.broadcast_to(sig[:, None, :], (M, 8, D))
    del_k_L = dk.reshape(1, 8, 1).repeat(M, 0)
    base_L_k = apply_shifts(base_L, cols_km, del_k_L)
    del_mus_L = jnp.stack(
        [dmu_c.reshape(1, 8).repeat(M, 0), dmu_p.reshape(1, 8).repeat(M, 0)],
        axis=-1,
    )
    sig_L = apply_shifts(base_L_k, cols_musL, del_mus_L)
    sigma_p = sigma_p.at[..., 11:19, :].set(sig_L.reshape(sigma.shape[:-1] + (8, D)))
    mels = mels.at[..., 11:19].set(delta_phiV.reshape(sigma.shape[:-1] + (1,)) * sgn_R)

    return sigma_p, mels


@jax.jit
def _sv_bs_kernel(
    sigma: jnp.ndarray,
    idx_mu: jnp.ndarray,
    idx_mu_m: jnp.ndarray,
    idx_mu_p: jnp.ndarray,
    idx_mu_mm: jnp.ndarray,
    idx_mu_pp: jnp.ndarray,
    idx_km: jnp.ndarray,
    idx_kp: jnp.ndarray,
    idx_kmm: jnp.ndarray,
    idx_kpp: jnp.ndarray,
    idx_kppp: jnp.ndarray,
    idx_kmmm: jnp.ndarray,
    use_gamma: jnp.ndarray,
    delta_i: jnp.ndarray,
    half_delta_i: jnp.ndarray,
    outer_km_c: jnp.ndarray,
    outer_kp_c: jnp.ndarray,
    mod_span: jnp.ndarray,
    state_min: jnp.ndarray,
    immirzi: jnp.ndarray,
):
    """
    Correct BRA-convention kernel.

    Returns (sigma_p, mels) such that:
        mels = <sigma | H | sigma_p>

    Implementation rule:
      - Diagonal terms are evaluated on the row state sigma.
      - Off-diagonal terms are generated as PREIMAGES (inverse shifts)
        and their prefactors are evaluated on those preimages.
    """
    D = sigma.shape[-1]
    sig = sigma.reshape((-1, D))
    M = sig.shape[0]

    take = partial(jnp.take, axis=-1)
    as1 = lambda x: x.reshape(M)

    # scalar indices (0-d arrays)
    imc = idx_mu
    imm = idx_mu_m
    imp = idx_mu_p
    ikm = idx_km
    ikp = idx_kp

    # row-state values
    mu_c = as1(take(sig, imc))
    mu_m = as1(take(sig, imm))
    mu_p = as1(take(sig, imp))
    k_m = as1(take(sig, ikm))
    k_p = as1(take(sig, ikp))

    # optional gathers (missing => 0)
    def _gather_opt(idx):
        idx_eff = jnp.where(idx >= 0, idx, jnp.int64(0))
        val = as1(take(sig, idx_eff))
        return jnp.where(idx >= 0, val, jnp.int64(0))

    mu_mm = _gather_opt(idx_mu_mm)  # kept for parity with slow code
    mu_pp = _gather_opt(idx_mu_pp)

    #
    #
    #   ROW-derived K and local volume prefactors (for DIAGONAL)
    K = (k_m + k_p).astype(jnp.int64)
    sqrtK = jnp.sqrt(jnp.abs(K).astype(jnp.float64))
    sqrtKp1 = jnp.sqrt(jnp.abs((K + 1)).astype(jnp.float64))
    sqrtKm1 = jnp.sqrt(jnp.abs((K - 1)).astype(jnp.float64))

    # ΔxV_row = |μ(v)| [√|K+1| − √|K−1|]
    abs_mu = jnp.abs(mu_c).astype(jnp.float64)
    delta_xV_row = abs_mu * (sqrtKp1 - sqrtKm1)

    # ΔφV_row = sgn_{δ/2}(μ(v)) √|K(v)|
    sgn_half = _sgn_half_delta(mu_c, half_delta_i)
    delta_phiV_row = sgn_half * sqrtK

    #
    #
    #   Γ and Γ′ (ROW state, only used in DIAGONAL)
    def _with_gamma():
        # K(v-1) = k_--(v) + k_-(v)  or  k_-(v) + outer_km_c
        kmm_val = _gather_opt(idx_kmm)
        left_outer_term = jnp.where(idx_kmm >= 0, jnp.int64(0), outer_km_c)
        K_left = (kmm_val + k_m + left_outer_term).astype(jnp.int64)

        # K(v)
        K_center = (k_m + k_p).astype(jnp.int64)

        # K(v+1) = k_+(v) + k_++(v)  or  k_+(v) + outer_kp_c
        kpp_val = _gather_opt(idx_kpp)
        right_outer_term = jnp.where(idx_kpp >= 0, jnp.int64(0), outer_kp_c)
        K_right = (k_p + kpp_val + right_outer_term).astype(jnp.int64)

        gamma = _gamma_phi_full(
            K_left=K_left,
            K_center=K_center,
            K_right=K_right,
            mu_center=mu_c,
            mu_minus=mu_m,
            mu_plus=mu_p,
        )
        gamma2 = gamma * gamma

        # K(v+2) = k_++(v) + k_+++(v) with fallbacks
        kp_contrib = jnp.where(idx_kpp >= 0, kpp_val, outer_kp_c)
        kppp_val = _gather_opt(idx_kppp)
        kpp_contrib = jnp.where(idx_kppp >= 0, kppp_val, outer_kp_c)
        K_right_right = (kp_contrib + kpp_contrib).astype(jnp.int64)

        gamma_vp1 = _gamma_phi_full(
            K_left=K_center,
            K_center=K_right,
            K_right=K_right_right,
            mu_center=mu_p,
            mu_minus=mu_c,
            mu_plus=mu_pp,
        )
        gamma_prime = gamma_vp1 - gamma
        return gamma, gamma2, gamma_prime

    def _no_gamma():
        z = jnp.zeros_like(delta_xV_row)
        return z, z, z

    gamma, gamma2, gamma_prime = lax.cond(use_gamma, _with_gamma, _no_gamma)

    #
    #
    #   outputs
    conn = 19
    sigma_p = jnp.broadcast_to(sigma[..., None, :], sigma.shape[:-1] + (conn, D)).copy()
    mels = jnp.zeros(sigma.shape[:-1] + (conn,), dtype=jnp.float64)

    #
    #
    #   shift helper
    def apply_shifts(
        base: jnp.ndarray, cols: jnp.ndarray, deltas: jnp.ndarray
    ) -> jnp.ndarray:
        if cols.size == 0:
            return base
        if base.ndim == 2:
            cur = base[:, cols]
            upd = _wrap_mod(cur, deltas, mod_span, state_min).astype(base.dtype)
            return base.at[:, cols].set(upd)
        else:
            cur = base[:, :, cols]
            upd = _wrap_mod(cur, deltas, mod_span, state_min).astype(base.dtype)
            return base.at[:, :, cols].set(upd)

    #
    #
    #   prefactor helpers on arbitrary base (..., D)
    def _delta_xV_from(base: jnp.ndarray) -> jnp.ndarray:
        mu_c_b = take(base, imc)
        k_m_b = take(base, ikm)
        k_p_b = take(base, ikp)
        K_b = (k_m_b + k_p_b).astype(jnp.int64)
        sqrtKp1_b = jnp.sqrt(jnp.abs(K_b + 1).astype(jnp.float64))
        sqrtKm1_b = jnp.sqrt(jnp.abs(K_b - 1).astype(jnp.float64))
        abs_mu_b = jnp.abs(mu_c_b).astype(jnp.float64)
        return abs_mu_b * (sqrtKp1_b - sqrtKm1_b)

    def _delta_phiV_from(base: jnp.ndarray) -> jnp.ndarray:
        mu_c_b = take(base, imc)
        k_m_b = take(base, ikm)
        k_p_b = take(base, ikp)
        K_b = (k_m_b + k_p_b).astype(jnp.int64)
        sqrtK_b = jnp.sqrt(jnp.abs(K_b).astype(jnp.float64))
        sgn_b = _sgn_half_delta(mu_c_b, half_delta_i)
        return sgn_b * sqrtK_b

    #
    #
    #   column selectors
    cols_mu = jnp.reshape(idx_mu, (1,)).astype(jnp.int64)
    cols_kp = jnp.reshape(idx_kp, (1,)).astype(jnp.int64)
    cols_km = jnp.reshape(idx_km, (1,)).astype(jnp.int64)
    cols_musR = jnp.stack([idx_mu.astype(jnp.int64), idx_mu_p.astype(jnp.int64)])
    cols_musL = jnp.stack([idx_mu_m.astype(jnp.int64), idx_mu.astype(jnp.int64)])

    #
    #
    #   H_C off-diagonals (BRA: PREIMAGES + prefactors on preimages)
    del_mu_p = jnp.full((M, 1), 2 * delta_i, dtype=jnp.int64)
    del_mu_m = jnp.full((M, 1), -2 * delta_i, dtype=jnp.int64)

    sig_mu_p_pre = apply_shifts(sig, cols_mu, -del_mu_p)
    sig_mu_m_pre = apply_shifts(sig, cols_mu, -del_mu_m)

    sigma_p = sigma_p.at[..., 0, :].set(sig_mu_p_pre.reshape(sigma.shape[:-1] + (D,)))
    sigma_p = sigma_p.at[..., 1, :].set(sig_mu_m_pre.reshape(sigma.shape[:-1] + (D,)))

    mel0 = _delta_xV_from(sig_mu_p_pre).reshape(sigma.shape[:-1])
    mel1 = _delta_xV_from(sig_mu_m_pre).reshape(sigma.shape[:-1])
    mels = mels.at[..., 0].set(mel0)
    mels = mels.at[..., 1].set(mel1)

    #
    #
    #   H_C diagonal (ROW)
    d64 = delta_i.astype(jnp.float64)
    diag = (
        -2.0 * delta_xV_row
        - 4.0 * (immirzi**2) * (d64**2) * (1.0 - gamma2) * delta_xV_row
        - 4.0 * (immirzi**2) * (d64**2) * gamma_prime * delta_phiV_row
    )
    sigma_p = sigma_p.at[..., 2, :].set(sigma)
    mels = mels.at[..., 2].set(diag.reshape(sigma.shape[:-1]))

    #
    #
    #   H_R: 8 terms on (k_+, μ_c, μ_p)
    dk = jnp.asarray([+1, +1, +1, +1, -1, -1, -1, -1], jnp.int64) * delta_i
    dmu_c = jnp.asarray([+1, +1, -1, -1, +1, +1, -1, -1], jnp.int64) * half_delta_i
    dmu_p = jnp.asarray([+1, -1, +1, -1, +1, -1, +1, -1], jnp.int64) * half_delta_i
    sgn_R = jnp.asarray([+1, -1, +1, -1, -1, +1, -1, +1], jnp.float64)

    base_R = jnp.broadcast_to(sig[:, None, :], (M, 8, D))

    del_k_R = jnp.broadcast_to(dk.reshape(1, 8, 1), (M, 8, 1))
    del_mus_R = jnp.stack(
        [
            jnp.broadcast_to(dmu_c.reshape(1, 8), (M, 8)),
            jnp.broadcast_to(dmu_p.reshape(1, 8), (M, 8)),
        ],
        axis=-1,
    )

    # BRA preimages = inverse shifts
    base_R_k_pre = apply_shifts(base_R, cols_kp, -del_k_R)
    sig_R_pre = apply_shifts(base_R_k_pre, cols_musR, -del_mus_R)

    sigma_p = sigma_p.at[..., 3:11, :].set(sig_R_pre.reshape(sigma.shape[:-1] + (8, D)))

    delta_phiV_R_pre = _delta_phiV_from(sig_R_pre)  # (M,8)
    mR = (delta_phiV_R_pre * sgn_R.reshape(1, 8)).reshape(sigma.shape[:-1] + (8,))
    mels = mels.at[..., 3:11].set(mR)

    #
    #
    #   H_L: 8 mirror terms on (k_-, μ_m, μ_c)
    base_L = jnp.broadcast_to(sig[:, None, :], (M, 8, D))

    del_k_L = del_k_R
    del_mus_L = del_mus_R  # applied to (μ_m, μ_c) via cols_musL

    base_L_k_pre = apply_shifts(base_L, cols_km, -del_k_L)
    sig_L_pre = apply_shifts(base_L_k_pre, cols_musL, -del_mus_L)

    sigma_p = sigma_p.at[..., 11:19, :].set(
        sig_L_pre.reshape(sigma.shape[:-1] + (8, D))
    )

    delta_phiV_L_pre = _delta_phiV_from(sig_L_pre)  # (M,8)
    mL = (delta_phiV_L_pre * sgn_R.reshape(1, 8)).reshape(sigma.shape[:-1] + (8,))
    mels = mels.at[..., 11:19].set(mL)

    return sigma_p, mels


@register_pytree_node_class
class SphericalVertexConstraintBojowaldSwiderskiJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 19

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(
        self,
        H,
        vertex: int,
        *,
        include_gamma_terms: bool = True,
        delta: int = 2,
        outer_km_constant: int = 0,
        outer_kp_constant: int = 0,
        immirzi: float = 1.0,
    ):
        super().__init__(H.hilbert_netket)

        if int(H.hilbert.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires `{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}`."
            )

        self.D = int(H.size)

        # validate inner vertex
        verts_k = list(H.graph.vertices_k)
        inner = verts_k[2:-2]
        if vertex not in inner:
            raise ValueError(
                f"vertex={vertex} must be an inner k-vertex. Valid vertices are: {inner}"
            )
        self.v = int(vertex)

        # local state wrap params
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)

        # resolve edges at v
        [mu_here], k_two = H.graph.get_edges_at_k_vertex(self.v)
        left = [e for e in k_two if e[1] == self.v]
        right = [e for e in k_two if e[0] == self.v]
        if len(left) != 1 or len(right) != 1:
            raise RuntimeError(
                f"Could not disambiguate left/right k-edges at v={self.v}: {k_two}"
            )

        e_minus, e_plus = left[0], right[0]
        v_minus, v_plus = e_minus[0], e_plus[1]
        [mu_left] = H.graph.get_edges_at_k_vertex(v_minus)[0]
        [mu_right] = H.graph.get_edges_at_k_vertex(v_plus)[0]

        #
        #
        #   k_--(v)
        if H.graph.has_k_vertex_neighbours(self.v, "left"):
            km_list = H.graph.get_kminus_kplus(self.v - 1)[0]
            self._idx_kmm = (
                jnp.int64(H.graph.edge_to_index(km_list[0]))
                if len(km_list) == 1
                else jnp.int64(-1)
            )
        else:
            self._idx_kmm = jnp.int64(-1)

        #
        #
        #   k_++(v)
        if H.graph.has_k_vertex_neighbours(self.v, "right"):
            kp_list = H.graph.get_kminus_kplus(self.v + 1)[1]
            self._idx_kpp = (
                jnp.int64(H.graph.edge_to_index(kp_list[0]))
                if len(kp_list) == 1
                else jnp.int64(-1)
            )
        else:
            self._idx_kpp = jnp.int64(-1)

        #
        #
        #   k_+++(v) and μ(v+2)
        if H.graph.has_k_vertex_neighbours(self.v + 1, "right"):
            kppp_list = H.graph.get_kminus_kplus(self.v + 2)[1]
            self._idx_kppp = (
                jnp.int64(H.graph.edge_to_index(kppp_list[0]))
                if len(kppp_list) == 1
                else jnp.int64(-1)
            )
            [mupp], _ = H.graph.get_edges_at_k_vertex(self.v + 2)
            self._idx_mu_pp = jnp.int64(H.graph.edge_to_index(mupp))
        else:
            self._idx_kppp = jnp.int64(-1)
            self._idx_mu_pp = jnp.int64(-1)

        #
        #
        #   k_---(v) and μ(v-2)
        if H.graph.has_k_vertex_neighbours(self.v - 1, "left"):
            kmmm_list = H.graph.get_kminus_kplus(self.v - 2)[0]
            self._idx_kmmm = (
                jnp.int64(H.graph.edge_to_index(kmmm_list[0]))
                if len(kmmm_list) == 1
                else jnp.int64(-1)
            )
            [mumm], _ = H.graph.get_edges_at_k_vertex(self.v - 2)
            self._idx_mu_mm = jnp.int64(H.graph.edge_to_index(mumm))
        else:
            self._idx_kmmm = jnp.int64(-1)
            self._idx_mu_mm = jnp.int64(-1)

        # flatten indices
        self._idx_mu = jnp.int64(H.graph.edge_to_index(mu_here))
        self._idx_mu_m = jnp.int64(H.graph.edge_to_index(mu_left))
        self._idx_mu_p = jnp.int64(H.graph.edge_to_index(mu_right))
        self._idx_km = jnp.int64(H.graph.edge_to_index(e_minus))
        self._idx_kp = jnp.int64(H.graph.edge_to_index(e_plus))

        # params
        self._use_gamma = bool(include_gamma_terms)
        self._delta = int(delta)
        self._half_delta = int(self._delta // 2)
        self._outer_km_c = int(outer_km_constant)
        self._outer_kp_c = int(outer_kp_constant)
        self._immirzi = float(immirzi)

    #
    #
    #   pytree plumbing
    def tree_flatten(self):
        leaves = (
            self._idx_mu,
            self._idx_mu_m,
            self._idx_mu_p,
            self._idx_mu_mm,
            self._idx_mu_pp,
            self._idx_km,
            self._idx_kp,
            self._idx_kmm,
            self._idx_kpp,
            self._idx_kppp,
            self._idx_kmmm,
        )
        aux = dict(
            hilbert=self.hilbert,
            D=self.D,
            v=self.v,
            use_gamma=self._use_gamma,
            delta=self._delta,
            half_delta=self._half_delta,
            outer_km_c=self._outer_km_c,
            outer_kp_c=self._outer_kp_c,
            mod_span=self._mod_span,
            state_min=self._state_min,
            immirzi=self._immirzi,
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.D = int(aux["D"])
        obj.v = int(aux["v"])
        (
            obj._idx_mu,
            obj._idx_mu_m,
            obj._idx_mu_p,
            obj._idx_mu_mm,
            obj._idx_mu_pp,
            obj._idx_km,
            obj._idx_kp,
            obj._idx_kmm,
            obj._idx_kpp,
            obj._idx_kppp,
            obj._idx_kmmm,
        ) = leaves
        obj._use_gamma = bool(aux["use_gamma"])
        obj._delta = int(aux["delta"])
        obj._half_delta = int(aux["half_delta"])
        obj._outer_km_c = int(aux["outer_km_c"])
        obj._outer_kp_c = int(aux["outer_kp_c"])
        obj._mod_span = int(aux["mod_span"])
        obj._state_min = int(aux["state_min"])
        obj._immirzi = float(aux["immirzi"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int64)
        sigma_p, mels = _sv_bs_kernel(
            sigma=x,
            idx_mu=self._idx_mu,
            idx_mu_m=self._idx_mu_m,
            idx_mu_p=self._idx_mu_p,
            idx_mu_mm=self._idx_mu_mm,
            idx_mu_pp=self._idx_mu_pp,
            idx_km=self._idx_km,
            idx_kp=self._idx_kp,
            idx_kmm=self._idx_kmm,
            idx_kpp=self._idx_kpp,
            idx_kppp=self._idx_kppp,
            idx_kmmm=self._idx_kmmm,
            use_gamma=jnp.asarray(self._use_gamma, dtype=bool),
            delta_i=jnp.asarray(self._delta, dtype=jnp.int64),
            half_delta_i=jnp.asarray(self._half_delta, dtype=jnp.int64),
            outer_km_c=jnp.asarray(self._outer_km_c, dtype=jnp.int64),
            outer_kp_c=jnp.asarray(self._outer_kp_c, dtype=jnp.int64),
            mod_span=jnp.asarray(self._mod_span, dtype=jnp.int64),
            state_min=jnp.asarray(self._state_min, dtype=jnp.int64),
            immirzi=jnp.asarray(self._immirzi, dtype=jnp.float64),
        )
        return sigma_p, mels


@register_pytree_node_class
class SphericalVertexConstraintBojowaldSwiderskiJaxAdjoint(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 19

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(
        self,
        H,
        vertex: int,
        *,
        include_gamma_terms: bool = True,
        delta: int = 2,
        outer_km_constant: int = 0,
        outer_kp_constant: int = 0,
        immirzi: float = 1.0,
    ):
        super().__init__(H.hilbert_netket)

        if int(H.hilbert.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires `{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}`."
            )

        self.D = int(H.size)

        # validate inner vertex
        verts_k = list(H.graph.vertices_k)
        inner = verts_k[2:-2]
        if vertex not in inner:
            raise ValueError(
                f"vertex={vertex} must be an inner k-vertex. Valid vertices are: {inner}"
            )
        self.v = int(vertex)

        # local state wrap params
        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)

        # resolve edges at v
        [mu_here], k_two = H.graph.get_edges_at_k_vertex(self.v)
        left = [e for e in k_two if e[1] == self.v]
        right = [e for e in k_two if e[0] == self.v]
        if len(left) != 1 or len(right) != 1:
            raise RuntimeError(
                f"Could not disambiguate left/right k-edges at v={self.v}: {k_two}"
            )

        e_minus, e_plus = left[0], right[0]
        v_minus, v_plus = e_minus[0], e_plus[1]
        [mu_left] = H.graph.get_edges_at_k_vertex(v_minus)[0]
        [mu_right] = H.graph.get_edges_at_k_vertex(v_plus)[0]

        #
        #
        #   k_--(v)
        if H.graph.has_k_vertex_neighbours(self.v, "left"):
            km_list = H.graph.get_kminus_kplus(self.v - 1)[0]
            self._idx_kmm = (
                jnp.int64(H.graph.edge_to_index(km_list[0]))
                if len(km_list) == 1
                else jnp.int64(-1)
            )
        else:
            self._idx_kmm = jnp.int64(-1)

        #
        #
        #   k_++(v)
        if H.graph.has_k_vertex_neighbours(self.v, "right"):
            kp_list = H.graph.get_kminus_kplus(self.v + 1)[1]
            self._idx_kpp = (
                jnp.int64(H.graph.edge_to_index(kp_list[0]))
                if len(kp_list) == 1
                else jnp.int64(-1)
            )
        else:
            self._idx_kpp = jnp.int64(-1)

        #
        #
        #   k_+++(v) and μ(v+2)
        if H.graph.has_k_vertex_neighbours(self.v + 1, "right"):
            kppp_list = H.graph.get_kminus_kplus(self.v + 2)[1]
            self._idx_kppp = (
                jnp.int64(H.graph.edge_to_index(kppp_list[0]))
                if len(kppp_list) == 1
                else jnp.int64(-1)
            )
            [mupp], _ = H.graph.get_edges_at_k_vertex(self.v + 2)
            self._idx_mu_pp = jnp.int64(H.graph.edge_to_index(mupp))
        else:
            self._idx_kppp = jnp.int64(-1)
            self._idx_mu_pp = jnp.int64(-1)

        #
        #
        #   k_---(v) and μ(v-2)
        if H.graph.has_k_vertex_neighbours(self.v - 1, "left"):
            kmmm_list = H.graph.get_kminus_kplus(self.v - 2)[0]
            self._idx_kmmm = (
                jnp.int64(H.graph.edge_to_index(kmmm_list[0]))
                if len(kmmm_list) == 1
                else jnp.int64(-1)
            )
            [mumm], _ = H.graph.get_edges_at_k_vertex(self.v - 2)
            self._idx_mu_mm = jnp.int64(H.graph.edge_to_index(mumm))
        else:
            self._idx_kmmm = jnp.int64(-1)
            self._idx_mu_mm = jnp.int64(-1)

        # flatten indices
        self._idx_mu = jnp.int64(H.graph.edge_to_index(mu_here))
        self._idx_mu_m = jnp.int64(H.graph.edge_to_index(mu_left))
        self._idx_mu_p = jnp.int64(H.graph.edge_to_index(mu_right))
        self._idx_km = jnp.int64(H.graph.edge_to_index(e_minus))
        self._idx_kp = jnp.int64(H.graph.edge_to_index(e_plus))

        # params
        self._use_gamma = bool(include_gamma_terms)
        self._delta = int(delta)
        self._half_delta = int(self._delta // 2)
        self._outer_km_c = int(outer_km_constant)
        self._outer_kp_c = int(outer_kp_constant)
        self._immirzi = float(immirzi)

    #
    #
    #   pytree plumbing
    def tree_flatten(self):
        leaves = (
            self._idx_mu,
            self._idx_mu_m,
            self._idx_mu_p,
            self._idx_mu_mm,
            self._idx_mu_pp,
            self._idx_km,
            self._idx_kp,
            self._idx_kmm,
            self._idx_kpp,
            self._idx_kppp,
            self._idx_kmmm,
        )
        aux = dict(
            hilbert=self.hilbert,
            D=self.D,
            v=self.v,
            use_gamma=self._use_gamma,
            delta=self._delta,
            half_delta=self._half_delta,
            outer_km_c=self._outer_km_c,
            outer_kp_c=self._outer_kp_c,
            mod_span=self._mod_span,
            state_min=self._state_min,
            immirzi=self._immirzi,
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.D = int(aux["D"])
        obj.v = int(aux["v"])
        (
            obj._idx_mu,
            obj._idx_mu_m,
            obj._idx_mu_p,
            obj._idx_mu_mm,
            obj._idx_mu_pp,
            obj._idx_km,
            obj._idx_kp,
            obj._idx_kmm,
            obj._idx_kpp,
            obj._idx_kppp,
            obj._idx_kmmm,
        ) = leaves
        obj._use_gamma = bool(aux["use_gamma"])
        obj._delta = int(aux["delta"])
        obj._half_delta = int(aux["half_delta"])
        obj._outer_km_c = int(aux["outer_km_c"])
        obj._outer_kp_c = int(aux["outer_kp_c"])
        obj._mod_span = int(aux["mod_span"])
        obj._state_min = int(aux["state_min"])
        obj._immirzi = float(aux["immirzi"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int64)
        sigma_p, mels = _sv_bs_kernel_adjoint(
            sigma=x,
            idx_mu=self._idx_mu,
            idx_mu_m=self._idx_mu_m,
            idx_mu_p=self._idx_mu_p,
            idx_mu_mm=self._idx_mu_mm,
            idx_mu_pp=self._idx_mu_pp,
            idx_km=self._idx_km,
            idx_kp=self._idx_kp,
            idx_kmm=self._idx_kmm,
            idx_kpp=self._idx_kpp,
            idx_kppp=self._idx_kppp,
            idx_kmmm=self._idx_kmmm,
            use_gamma=jnp.asarray(self._use_gamma, dtype=bool),
            delta_i=jnp.asarray(self._delta, dtype=jnp.int64),
            half_delta_i=jnp.asarray(self._half_delta, dtype=jnp.int64),
            outer_km_c=jnp.asarray(self._outer_km_c, dtype=jnp.int64),
            outer_kp_c=jnp.asarray(self._outer_kp_c, dtype=jnp.int64),
            mod_span=jnp.asarray(self._mod_span, dtype=jnp.int64),
            state_min=jnp.asarray(self._state_min, dtype=jnp.int64),
            immirzi=jnp.asarray(self._immirzi, dtype=jnp.float64),
        )
        return sigma_p, mels
