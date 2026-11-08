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


"""Bojowald-Swiderski spherical vertex constraint."""

from __future__ import annotations

from functools import partial
from numbers import Integral
from typing import Any

import jax.numpy as jnp

import jax
from jax import lax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.utils.struct import field

from .stencil import SphericalStencil
from .stencil import require_spherical_hilbert
from .stencil import spherical_stencil


def _wrap_mod(
    vals: jax.Array, delta: jax.Array, mod_span: jax.Array, state_min: jax.Array
) -> jax.Array:
    """Add charge displacements and wrap into the unit-spaced finite range."""
    return ((vals + delta - state_min) % mod_span) + state_min


def _sgn_half_delta(mu: jax.Array, half_delta: jax.Array) -> jax.Array:
    r"""Evaluate the regularized sign $\operatorname{clip}(2\mu/\delta,-1,1)$."""
    one = half_delta.astype(jnp.float64)
    mu_f = mu.astype(jnp.float64)
    return jnp.where(
        mu_f >= one,
        1.0,
        jnp.where(mu_f <= -one, -1.0, mu_f / jnp.maximum(one, 1e-12)),
    )


def _gamma_phi_full(
    K_left: jax.Array,
    K_center: jax.Array,
    K_right: jax.Array,
    mu_center: jax.Array,
    mu_minus: jax.Array,
    mu_plus: jax.Array,
) -> jax.Array:
    r"""Evaluate the spherical spin connection from neighboring fluxes.

    With $E^x=K/2$ and $I_\pm=(|\mu_v|+|\mu_{v\pm1}|)/2$, return
    $[-(E^x_{v+1}-E^x_v)/I_+ +(E^x_v-E^x_{v-1})/I_-]/4$.
    A vanishing link average contributes zero.
    """
    fd = jnp.float64
    Ex_m = 0.5 * K_left.astype(fd)
    Ex_c = 0.5 * K_center.astype(fd)
    Ex_p = 0.5 * K_right.astype(fd)
    int_plus = 0.5 * (jnp.abs(mu_center).astype(fd) + jnp.abs(mu_plus).astype(fd))
    int_minus = 0.5 * (jnp.abs(mu_center).astype(fd) + jnp.abs(mu_minus).astype(fd))
    term_plus = jnp.where(int_plus > 0.0, -(Ex_p - Ex_c) / int_plus, 0.0)
    term_minus = jnp.where(int_minus > 0.0, (Ex_c - Ex_m) / int_minus, 0.0)
    return 0.25 * (term_plus + term_minus)


def _prefactors(
    sig: jax.Array,
    *,
    idx_mu: jax.Array,
    idx_mu_m: jax.Array,
    idx_mu_p: jax.Array,
    idx_mu_pp: jax.Array,
    idx_km: jax.Array,
    idx_kp: jax.Array,
    idx_kmm: jax.Array,
    idx_kpp: jax.Array,
    idx_kppp: jax.Array,
    use_gamma: jax.Array,
    half_delta_i: jax.Array,
    outer_km_c: jax.Array,
    outer_kp_c: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate the two volume differences and optional spin-connection terms.

    Returns `delta_xV`, `delta_phiV`, the squared spin connection, and its
    forward radial difference for a rank-two configuration batch. Optional
    outer indices use the supplied boundary constants.
    """
    M = sig.shape[0]
    take = partial(jnp.take, axis=-1)

    def as1(x):
        """Flatten a gathered site column to one value per configuration."""
        return x.reshape(M)

    mu_c = as1(take(sig, idx_mu))
    mu_m = as1(take(sig, idx_mu_m))
    mu_p = as1(take(sig, idx_mu_p))
    k_m = as1(take(sig, idx_km))
    k_p = as1(take(sig, idx_kp))

    def gather_optional(idx):
        """Gather an outer label, substituting zero for a missing site."""
        idx_eff = jnp.where(idx >= 0, idx, jnp.int64(0))
        value = as1(take(sig, idx_eff))
        return jnp.where(idx >= 0, value, jnp.int64(0))

    K = (k_m + k_p).astype(jnp.int64)
    sqrtK = jnp.sqrt(jnp.abs(K).astype(jnp.float64))
    delta_xV = jnp.abs(mu_c).astype(jnp.float64) * (
        jnp.sqrt(jnp.abs(K + 1).astype(jnp.float64))
        - jnp.sqrt(jnp.abs(K - 1).astype(jnp.float64))
    )
    delta_phiV = _sgn_half_delta(mu_c, half_delta_i) * sqrtK

    def with_gamma():
        """Compute the squared spin connection and its forward difference."""
        kmm = gather_optional(idx_kmm)
        kpp = gather_optional(idx_kpp)
        K_left = (kmm + k_m + jnp.where(idx_kmm >= 0, 0, outer_km_c)).astype(jnp.int64)
        K_center = (k_m + k_p).astype(jnp.int64)
        K_right = (k_p + kpp + jnp.where(idx_kpp >= 0, 0, outer_kp_c)).astype(jnp.int64)
        gamma = _gamma_phi_full(K_left, K_center, K_right, mu_c, mu_m, mu_p)
        kppp = gather_optional(idx_kppp)
        mu_pp = gather_optional(idx_mu_pp)
        K_right_right = (
            jnp.where(idx_kpp >= 0, kpp, outer_kp_c)
            + jnp.where(idx_kppp >= 0, kppp, outer_kp_c)
        ).astype(jnp.int64)
        gamma_vp1 = _gamma_phi_full(K_center, K_right, K_right_right, mu_p, mu_c, mu_pp)
        return gamma * gamma, gamma_vp1 - gamma

    def without_gamma():
        """Return zero spin-connection contributions for the simplified model."""
        z = jnp.zeros_like(delta_xV)
        return z, z

    gamma2, gamma_prime = lax.cond(use_gamma, with_gamma, without_gamma)
    return delta_xV, delta_phiV, gamma2, gamma_prime


def _apply_shifts(
    base: jax.Array,
    cols: jax.Array,
    deltas: jax.Array,
    *,
    mod_span: jax.Array,
    state_min: jax.Array,
) -> jax.Array:
    """Update selected columns of batched configurations with cyclic charge shifts."""
    if base.ndim == 2:
        cur = base[:, cols]
        return base.at[:, cols].set(
            _wrap_mod(cur, deltas, mod_span, state_min).astype(base.dtype)
        )
    cur = base[:, :, cols]
    return base.at[:, :, cols].set(
        _wrap_mod(cur, deltas, mod_span, state_min).astype(base.dtype)
    )


@jax.jit
def _bs_kernel(
    sigma: jax.Array,
    idx_mu: jax.Array,
    idx_mu_m: jax.Array,
    idx_mu_p: jax.Array,
    idx_mu_pp: jax.Array,
    idx_km: jax.Array,
    idx_kp: jax.Array,
    idx_kmm: jax.Array,
    idx_kpp: jax.Array,
    idx_kppp: jax.Array,
    use_gamma: jax.Array,
    delta_i: jax.Array,
    half_delta_i: jax.Array,
    outer_km_c: jax.Array,
    outer_kp_c: jax.Array,
    mod_span: jax.Array,
    state_min: jax.Array,
    immirzi: jax.Array,
    adjoint: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate all 19 spherical Hamiltonian branches for arbitrary batch axes.

    The adjoint flag controls both shift directions and which configuration
    supplies the off-diagonal coefficient. The diagonal branch is unchanged.
    """
    D = sigma.shape[-1]
    sig = sigma.reshape((-1, D))
    M = sig.shape[0]
    out_shape = sigma.shape[:-1]
    delta_xV, delta_phiV, gamma2, gamma_prime = _prefactors(
        sig,
        idx_mu=idx_mu,
        idx_mu_m=idx_mu_m,
        idx_mu_p=idx_mu_p,
        idx_mu_pp=idx_mu_pp,
        idx_km=idx_km,
        idx_kp=idx_kp,
        idx_kmm=idx_kmm,
        idx_kpp=idx_kpp,
        idx_kppp=idx_kppp,
        use_gamma=use_gamma,
        half_delta_i=half_delta_i,
        outer_km_c=outer_km_c,
        outer_kp_c=outer_kp_c,
    )

    sigma_p = jnp.broadcast_to(sigma[..., None, :], (*out_shape, 19, D)).copy()
    mels = jnp.zeros(
        (
            *out_shape,
            19,
        ),
        dtype=jnp.float64,
    )

    cols_mu = jnp.reshape(idx_mu, (1,)).astype(jnp.int64)
    cols_kp = jnp.reshape(idx_kp, (1,)).astype(jnp.int64)
    cols_km = jnp.reshape(idx_km, (1,)).astype(jnp.int64)
    cols_musR = jnp.stack([idx_mu.astype(jnp.int64), idx_mu_p.astype(jnp.int64)])
    cols_musL = jnp.stack([idx_mu_m.astype(jnp.int64), idx_mu.astype(jnp.int64)])

    bra_sign = jnp.where(adjoint, jnp.int64(1), jnp.int64(-1))
    del_mu_p = jnp.full((M, 1), 2 * delta_i, dtype=jnp.int64)
    del_mu_m = jnp.full((M, 1), -2 * delta_i, dtype=jnp.int64)
    sig_mu_p = _apply_shifts(
        sig, cols_mu, bra_sign * del_mu_p, mod_span=mod_span, state_min=state_min
    )
    sig_mu_m = _apply_shifts(
        sig, cols_mu, bra_sign * del_mu_m, mod_span=mod_span, state_min=state_min
    )
    dx_p, _, _, _ = _prefactors(
        sig_mu_p,
        idx_mu=idx_mu,
        idx_mu_m=idx_mu_m,
        idx_mu_p=idx_mu_p,
        idx_mu_pp=idx_mu_pp,
        idx_km=idx_km,
        idx_kp=idx_kp,
        idx_kmm=idx_kmm,
        idx_kpp=idx_kpp,
        idx_kppp=idx_kppp,
        use_gamma=jnp.asarray(False),
        half_delta_i=half_delta_i,
        outer_km_c=outer_km_c,
        outer_kp_c=outer_kp_c,
    )
    dx_m, _, _, _ = _prefactors(
        sig_mu_m,
        idx_mu=idx_mu,
        idx_mu_m=idx_mu_m,
        idx_mu_p=idx_mu_p,
        idx_mu_pp=idx_mu_pp,
        idx_km=idx_km,
        idx_kp=idx_kp,
        idx_kmm=idx_kmm,
        idx_kpp=idx_kpp,
        idx_kppp=idx_kppp,
        use_gamma=jnp.asarray(False),
        half_delta_i=half_delta_i,
        outer_km_c=outer_km_c,
        outer_kp_c=outer_kp_c,
    )
    sigma_p = sigma_p.at[..., 0, :].set(
        sig_mu_p.reshape(
            (
                *out_shape,
                D,
            )
        )
    )
    sigma_p = sigma_p.at[..., 1, :].set(
        sig_mu_m.reshape(
            (
                *out_shape,
                D,
            )
        )
    )
    mels = mels.at[..., 0].set(jnp.where(adjoint, delta_xV, dx_p).reshape(out_shape))
    mels = mels.at[..., 1].set(jnp.where(adjoint, delta_xV, dx_m).reshape(out_shape))

    d64 = delta_i.astype(jnp.float64)
    diag = (
        -2.0 * delta_xV
        - 4.0 * immirzi**2 * d64**2 * (1.0 - gamma2) * delta_xV
        - 4.0 * immirzi**2 * d64**2 * gamma_prime * delta_phiV
    )
    sigma_p = sigma_p.at[..., 2, :].set(sigma)
    mels = mels.at[..., 2].set(diag.reshape(out_shape))

    dk = jnp.asarray([+1, +1, +1, +1, -1, -1, -1, -1], jnp.int64) * delta_i
    dmu_c = jnp.asarray([+1, +1, -1, -1, +1, +1, -1, -1], jnp.int64) * half_delta_i
    dmu_p = jnp.asarray([+1, -1, +1, -1, +1, -1, +1, -1], jnp.int64) * half_delta_i
    signs = jnp.asarray([+1, -1, +1, -1, -1, +1, -1, +1], jnp.float64)
    shift_sign = jnp.where(adjoint, jnp.int64(1), jnp.int64(-1))
    base = jnp.broadcast_to(sig[:, None, :], (M, 8, D))
    del_k = jnp.broadcast_to(dk.reshape(1, 8, 1), (M, 8, 1))
    del_mus = jnp.stack(
        (
            jnp.broadcast_to(dmu_c.reshape(1, 8), (M, 8)),
            jnp.broadcast_to(dmu_p.reshape(1, 8), (M, 8)),
        ),
        axis=-1,
    )
    sig_R = _apply_shifts(
        _apply_shifts(
            base, cols_kp, shift_sign * del_k, mod_span=mod_span, state_min=state_min
        ),
        cols_musR,
        shift_sign * del_mus,
        mod_span=mod_span,
        state_min=state_min,
    )
    sig_L = _apply_shifts(
        _apply_shifts(
            base, cols_km, shift_sign * del_k, mod_span=mod_span, state_min=state_min
        ),
        cols_musL,
        shift_sign * del_mus,
        mod_span=mod_span,
        state_min=state_min,
    )

    def phi_from(base_state: jax.Array) -> jax.Array:
        """Evaluate the transverse volume difference on shifted branch states."""
        take = partial(jnp.take, axis=-1)
        mu = take(base_state, idx_mu)
        km = take(base_state, idx_km)
        kp = take(base_state, idx_kp)
        return _sgn_half_delta(mu, half_delta_i) * jnp.sqrt(
            jnp.abs((km + kp).astype(jnp.int64)).astype(jnp.float64)
        )

    phi_R = jnp.where(adjoint, delta_phiV.reshape(M, 1), phi_from(sig_R))
    phi_L = jnp.where(adjoint, delta_phiV.reshape(M, 1), phi_from(sig_L))
    sigma_p = sigma_p.at[..., 3:11, :].set(sig_R.reshape((*out_shape, 8, D)))
    sigma_p = sigma_p.at[..., 11:19, :].set(sig_L.reshape((*out_shape, 8, D)))
    mels = mels.at[..., 3:11].set(
        (phi_R * signs.reshape(1, 8)).reshape(
            (
                *out_shape,
                8,
            )
        )
    )
    mels = mels.at[..., 11:19].set(
        (phi_L * signs.reshape(1, 8)).reshape(
            (
                *out_shape,
                8,
            )
        )
    )
    return sigma_p, mels


class BojowaldSwiderskiVertexConstraintOperator(ComputationalOperator):
    r"""The 19-branch spherical Bojowald-Swiderski Hamiltonian at one vertex.

    Write $K=k_-+k_+$ and define the volume differences

    $$
    X=|\mu_v|(\sqrt{|K+1|}-\sqrt{|K-1|}),\qquad
    P=\operatorname{clip}(2\mu_v/\delta,-1,1)\sqrt{|K|}.
    $$

    There are two transverse shifts by $\pm2\delta$, eight branches on each
    adjacent radial edge, and one diagonal branch. A radial branch shifts
    $k_\pm$ by $s_k\delta$ and its two endpoint transverse labels by
    $s_1\delta/2,s_2\delta/2$. Its coefficient is $s_k s_2 P$. The transverse
    branches have coefficient $X$. All shifts wrap at the finite charge cutoff.

    The diagonal coefficient is

    $$
    -2X-4\gamma^2\delta^2(1-\Gamma_\varphi^2)X
    -4\gamma^2\delta^2\Gamma'_\varphi P.
    $$

    The spin connection uses neighboring radial-triad differences divided by
    averaged absolute transverse labels. A zero denominator contributes zero.
    Its derivative is the forward difference to the next radial vertex.
    Setting `include_gamma_terms=False` sets both connection terms to zero.

    The forward row uses the negative of each listed shift and evaluates its
    coefficient at the target. The adjoint uses positive shifts and evaluates
    the coefficient at the input. This ordering matters because the operator
    is not Hermitian. Missing outer radial labels are supplied by the boundary
    constants. Integer charges with unit spacing and a positive even `delta`
    are required.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        vertex (Any): Radial vertex or its index in `graph.vertices_k`.
        include_gamma_terms (bool): Whether to include the spin connection and its radial difference.
        delta (int): Positive even integer setting radial and transverse shift sizes.
        outer_km_constant (int): Radial label used when a required outer left edge is absent.
        outer_kp_constant (int): Radial label used when a required outer right edge is absent.
        immirzi (float): Nonzero real Barbero-Immirzi parameter $\gamma$.
        adjoint_kernel (bool): Whether to reverse the row shifts and move coefficients to the input.
    """

    vertex: Any = field(static=True)
    """Radial vertex or its index in `graph.vertices_k`."""

    include_gamma_terms: bool = field(static=True, default=True)
    """Whether to include the spin connection and its radial difference."""

    delta: int = field(static=True, default=2)
    """Positive even integer setting radial and transverse shift sizes."""

    outer_km_constant: int = field(static=True, default=0)
    """Radial label used when a required outer left edge is absent."""

    outer_kp_constant: int = field(static=True, default=0)
    """Radial label used when a required outer right edge is absent."""

    immirzi: float = field(static=True, default=1.0)
    r"""Nonzero real Barbero-Immirzi parameter $\gamma$."""

    adjoint_kernel: bool = field(static=True, default=False)
    """Whether to reverse the row shifts and move coefficients to the input."""

    _stencil: SphericalStencil | None = field(
        static=True, init=False, default=None, repr=False
    )
    """Resolved half-ladder site indices for the vertex neighborhood."""

    _q_min: int | float = field(static=True, init=False, default=0, repr=False)
    """Smallest allowed value on the local charge lattice."""

    _local_size: int = field(static=True, init=False, default=0, repr=False)
    """Number of allowed charges in one local degree of freedom."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_spherical_hilbert(self.hilbert, operator=type(self).__name__)
        if (
            isinstance(self.delta, bool)
            or not isinstance(self.delta, Integral)
            or self.delta <= 0
            or self.delta % 2 != 0
        ):
            raise ValueError("delta must be a positive even integer.")
        if float(self.immirzi) == 0.0:
            raise ValueError("immirzi must be non-zero.")
        q_min, _q_max, q_step, local_size = q_bounds(hilbert)
        if q_step != 1 or not float(q_min).is_integer():
            raise ValueError(
                "Bojowald-Swiderski operators require integer charges with U(1) step=1."
            )
        object.__setattr__(
            self, "_stencil", spherical_stencil(hilbert.graph, self.vertex)
        )
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_local_size", local_size)
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return 19

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return False

    @property
    def adjoint(self) -> BojowaldSwiderskiVertexConstraintOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return BojowaldSwiderskiVertexConstraintOperator(
            self.hilbert,
            self.vertex,
            include_gamma_terms=self.include_gamma_terms,
            delta=self.delta,
            outer_km_constant=self.outer_km_constant,
            outer_kp_constant=self.outer_kp_constant,
            immirzi=self.immirzi,
            adjoint_kernel=not self.adjoint_kernel,
        )

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        side = "adjoint" if self.adjoint_kernel else "forward"
        return f"BojowaldSwiderskiVertexConstraintOperator(vertex={self.vertex!r}, side={side}, delta={self.delta})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        stencil = self._stencil
        assert stencil is not None
        return _bs_kernel(
            jnp.asarray(state, dtype=jnp.int64),
            idx_mu=jnp.asarray(stencil.idx_mu, dtype=jnp.int64),
            idx_mu_m=jnp.asarray(stencil.idx_mu_m, dtype=jnp.int64),
            idx_mu_p=jnp.asarray(stencil.idx_mu_p, dtype=jnp.int64),
            idx_mu_pp=jnp.asarray(stencil.idx_mu_pp, dtype=jnp.int64),
            idx_km=jnp.asarray(stencil.idx_km, dtype=jnp.int64),
            idx_kp=jnp.asarray(stencil.idx_kp, dtype=jnp.int64),
            idx_kmm=jnp.asarray(stencil.idx_kmm, dtype=jnp.int64),
            idx_kpp=jnp.asarray(stencil.idx_kpp, dtype=jnp.int64),
            idx_kppp=jnp.asarray(stencil.idx_kppp, dtype=jnp.int64),
            use_gamma=jnp.asarray(self.include_gamma_terms),
            delta_i=jnp.asarray(self.delta, dtype=jnp.int64),
            half_delta_i=jnp.asarray(self.delta // 2, dtype=jnp.int64),
            outer_km_c=jnp.asarray(self.outer_km_constant, dtype=jnp.int64),
            outer_kp_c=jnp.asarray(self.outer_kp_constant, dtype=jnp.int64),
            mod_span=jnp.asarray(self._local_size, dtype=jnp.int64),
            state_min=jnp.asarray(self._q_min, dtype=jnp.int64),
            immirzi=jnp.asarray(self.immirzi, dtype=jnp.float64),
            adjoint=jnp.asarray(self.adjoint_kernel),
        )


__all__ = ["BojowaldSwiderskiVertexConstraintOperator"]
