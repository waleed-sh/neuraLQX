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

"""
Spherical (U(1)) single-vertex Euclidean Hamiltonian constraint (weak-coupling/polymer)

This operator encodes the local action H_v := H_L(v) + H_C(v) + H_R(v) at a chosen
inner k-vertex v of a half-ladder graph, without summing over vertices. It is implemented
as a NetKet-style "ComputationalOperator" that directly enumerates the connected
configurations and their matrix elements for each input computational basis state σ.


High-level summary
----------------------------------------------------------------------------------------------------
- Degrees of freedom (DoFs) at/around vertex v:
    - k_-(v): integer "k" charge on the k-edge (v-1 -> v)
    - k_+(v): integer "k" charge on the k-edge (v -> v+1)
    - μ(v)  : integer "μ" charge on the vertical μ-edge at v
    - μ(v-1), μ(v+1): integer μ charges at the neighbouring vertices

- Derived quantities at v:
    - K(v) := k_-(v) + k_+(v)                (integer)
    - ΔxV(v) := |μ(v)| [ √|K(v)+1| − √|K(v)-1| ]           (real)
    - ΔφV(v) := sgn_{δ/2}(μ(v)) · √|K(v)|                   (real)
      with δ = 2 (so δ/2 = 1) and sgn_{δ/2}(μ) ∈ {−1, 0, +1} for integer μ.

- Discrete spin connection and its forward difference:
    - E^x(v) := K(v) / 2,  E^φ(v) := |μ(v)|
    - Γ_φ(v) ≈ − [ E^x(v+1) − E^x(v−1) ] / [ 4 |E^φ(v)| ]
    - Γ′_φ(v) ≈ Γ_φ(v+1) − Γ_φ(v)

- Action on |σ⟩ (overall constants absorbed, we take γ=1 in these units):
    H_C(v)|σ⟩ =
        ΔxV(v) |σ; μ(v) -> μ(v)+2δ/2⟩  +  ΔxV(v) |σ; μ(v) -> μ(v)−2δ/2⟩
      + [ −2 ΔxV(v) − 4 δ^2 (1 − Γ_φ(v)^2) ΔxV(v) − 4 δ^2 Γ′_φ(v) ΔφV(v) ] |σ⟩

    H_R(v)|σ⟩:
      8 off-diagonals that shift k_+(v) by ±2δ/2 = ±2 and (μ(v), μ(v+1)) each by ±δ/2 = ±1,
      with alternating signs, each weighted by ΔφV(v).

    H_L(v)|σ⟩:
      mirror of H_R(v) on the left side: acts on k_-(v), μ(v-1), μ(v), same sign/shift pattern.

- Output connectivity (fixed-size, padded):
    We always emit exactly 19 connections per input state, in this order:
      [ H_C: μ+2,  H_C: μ−2,  H_C: diagonal,  H_R × 8,  H_L × 8 ].



Design choices/numerical notes
----------------------------------------------------------------------------------------------------
- Gauge sector:
    We assume the gauge (Gauss) constraint has been solved so that the remaining k-labels
    satisfy K(v) = k_-(v) + k_+(v). That is the quantity entering ΔxV and ΔφV

- Units and parameters:
    We fix polymerization step δ = 2 (so δ/2=1 on integer lattices) and absorb γ into units
    so that the diagonal matches the paper with γ=1. If a non-unity γ is needed globally,
    carry it in H and scale the diagonal accordingly upstream.

- Modular wrapping:
    Shifts are wrapped into the discrete window [−cutoff, cutoff] per DoF. This defines a
    finite computational Hilbert space and avoids dtype promotion by casting back to the
    input state dtype after modular arithmetic.

- Numerical stability:
    Γ_φ uses a denominator 4|E^φ| = 4|μ|, we guard with a small floor (1e−12) to avoid NaNs
    when μ=0 (which is physically fine since ΔφV also vanishes there).

- include_gamma_terms flag:
    - True: full diagonal as written (non-Γ "sin²" piece plus Γ² and Γ′ terms).
    - False: Γ_φ and Γ′_φ are set to zero but the non-Γ diagonal contributions remain.
             (i.e. we do NOT zero the diagonal entirely, we drop only Γ-dependent pieces.)


Performance notes
----------------------------------------------------------------------------------------------------
- Single pass over σ with vectorized gathers/scatters.
- Fixed-size (19) connectivity avoids dynamic shapes and JIT recompiles in JAX.
- No Python loops over (B, N), the only loops are over the 8+8 = 16 fixed right/left terms.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
from jax import lax

from neuralqx.graph import HalfLadderGraph
from neuralqx.operators.types.computational_operator import ComputationalOperator


class SphericalVertexConstraintBojowaldSwiderski(ComputationalOperator):
    r"""
    A ComputationalOperator implementing the single-vertex Euclidean constraint
    \( H_v = H_L(v) + H_C(v) + H_R(v) \) at a chosen inner k-vertex \(v\).

    Notes
    -----
    - This operator is not Hermitian in this bare Euclidean form. If a Hermitian positive
      operator is desired, wrap with a squaring construction upstream (e.g., `Squared(...)`).
    - All overall constants are taken to be absorbed into units with \(\gamma=1\) and \(\delta=2\).
    """

    def __init__(
        self,
        H,
        vertex: int,
        *,
        include_gamma_terms: bool = True,
        delta: int = 2,  # dev: strictly an integer, see the first dev comment below
        outer_km_constant: int = 0,
        outer_kp_constant: int = 0,
        immirzi: float = 1.0,
    ):

        #
        #
        # setup and class attributes

        # initialize parent with NetKet Hilbert
        super().__init__(H.hilbert)

        # enforce U(1), one scalar per edge
        if int(H.core.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )

        # HalfLadderGraph type check
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires the graph to be an instance of "
                f"`{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}` instead."
            )

        # cache host and invariants
        self._H = H

        # total number of scalar DoFs
        self.D = int(H.size)

        # integer cutoff for modular wrapping
        self.cutoff = int(H.cutoff)

        # modulus for wrap: values in [−cutoff, cutoff]
        # self._mod_span = int(2 * self.cutoff + 1)

        # get the allowed quantum numbers
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )
        self._state_step = jnp.asarray(
            int(H.allowed_basis_states.step), dtype=jnp.int64
        )
        self._state_max = jnp.asarray(
            int(self._state_min + (self._mod_span - 1) * self._state_step),
            dtype=jnp.int64,
        )

        # polymer step δ, fixed in this implementation
        # dev: this is now strictly an integer when working with delta = 2
        #      this can be changed, but some JAX arrays in the _get_conn_padded_kernel will need
        #      their dtype changed as well
        if type(delta) != int:
            raise ValueError(
                f"Currently, only integer (type `int`) values for `delta` is allowed, "
                f"got `{type(delta).__name__}` instead."
            )

        self.delta: int = delta

        # = 1.0 with δ=2, used in sgn_{δ/2}(μ)
        # dev: un-int this when going for general delta, see the dev comment above
        self._half_delta = int(self.delta / 2)

        # a flag to include gamma dependent terms or not
        self._use_gamma = bool(include_gamma_terms)

        # a constant to be added for constructing K(v) when there is no outer right edge
        self._outer_kp_constant = jnp.asarray(outer_kp_constant)

        # a constant to be added for constructing K(v) when there is no outer left edge
        self._outer_km_constant = jnp.asarray(outer_km_constant)

        # save the Immirzi parameter value
        self.immirzi = immirzi

        #
        #
        # validate vertex: it must be an "inner" k-vertex
        # the Graph object exposes H.g.vertices_k in order along the ladder spine
        # (e.g., [0,1,2,3,...]) and "inner" means it has both left and right neighbours
        verts_k = list(H.graph.vertices_k)
        inner = verts_k[2:-2]
        if vertex not in inner:
            raise ValueError(
                f"vertex={vertex} must be an inner k-vertex. Valid vertices are: {inner}"
            )

        # all clear, set it
        self.v = int(vertex)

        #
        #
        # resolve the three edges touching v: one μ-edge at v, two k-edges (left/right)
        # for each k-vertex id v, the Graph API provides:
        #   get_edges_at_k_vertex(v) -> ([μ_edge], [k_minus_edge, k_plus_edge])
        # where:
        #   - μ_edge is the vertical μ-edge "stuck into" vertex v,
        #   - k_minus_edge is the k-edge coming from v-1 -> v,
        #   - k_plus_edge  is the k-edge going from v -> v+1.
        # The graph stores a k-edge as a tuple/list (a, b, key) with a < b. Hence:
        #   - for the "left" incident k-edge at v, its b == v (since it ends at v)
        #   - for the "right" incident k-edge at v, its a == v (since it starts at v)
        #
        # dev: for this HalfLadderGraph type, it is true that a < b. However, if a user
        #      custom implements a half ladder, the following code will break, there is no
        #      guaranteed contract for a being < b
        [mu_here], k_two = H.graph.get_edges_at_k_vertex(self.v)

        # disambiguate left/right k-edges using endpoints
        # left edge is (v-1, v, *), right edge is (v, v+1, *).
        left_candidates = [e for e in k_two if e[1] == self.v]
        right_candidates = [e for e in k_two if e[0] == self.v]

        if len(left_candidates) != 1 or len(right_candidates) != 1:
            raise RuntimeError(
                f"Could not disambiguate left/right k-edges at v={self.v}: {k_two}"
            )

        # k_-(v): (v-1, v, key)
        e_minus = left_candidates[0]

        # k_+(v): (v, v+1, key)
        e_plus = right_candidates[0]

        # neighbour vertex ids (needed both for μ-edges at neighbours and for K(v±1))
        # v-1
        v_minus = e_minus[0]

        # v+1
        v_plus = e_plus[1]

        #
        #
        #   get the next nearest k-edges to build k_--(v) and k_++(v)

        # check if there is a k-vertex to the right, we know it does because it has to be an
        # inner vertex, but does not harm
        has_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "right")

        # if it does not, fast track
        if not has_right_k_vertex:
            self._has_kpp = False
            self._idx_kpp = jnp.int32(-1)
        else:
            # it has a k-vertex to its right, so get the k-edges attached to it
            k_edges_at_v_plus = H.graph.get_kminus_kplus(self.v + 1)

            # here, we need the outer k-edge, so we look up the index 1 in the returned list
            self._has_kpp = len(k_edges_at_v_plus[1]) == 1
            self._idx_kpp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus[1][0]))
                if self._has_kpp
                else jnp.int32(-1)
            )

        # check if there is a k-vertex to the left, we know it does because it has to be an
        # inner vertex, but does not harm
        has_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "left")

        # if it does not, fast track
        if not has_left_k_vertex:
            self._has_kmm = False
            self._idx_kmm = jnp.int32(-1)
        else:
            # it has a k-vertex to its left, so get the k-edges attached to it
            k_edges_at_v_minus = H.graph.get_kminus_kplus(self.v - 1)

            # here, we need the inner k-edge, so we look up the index 0 in the returned list
            self._has_kmm = len(k_edges_at_v_minus[0]) == 1
            self._idx_kmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus[0][0]))
                if self._has_kmm
                else jnp.int32(-1)
            )

        #
        #
        #   get the next nearest k-edges to build k_---(v) and k_+++(v)

        # check if there is a k-vertex to the right of v + 1
        has_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v + 1, "right")

        # if it does not, fast track
        if not has_right_k_vertex:
            # mark that there are no k_+++(v)
            self._has_kppp = False
            self._idx_kppp = jnp.int32(-1)

            # and consequently also no μ(v+2)
            self._has_mu_pp = False
            self._idx_mu_pp = jnp.int32(-1)
        else:
            # there is a k-vertex to the right of v+1, so get the k-edges attached to it
            k_edges_at_v_plus_plus = H.graph.get_kminus_kplus(self.v + 2)

            # here, we need the outer k-edge, so we look up the index 1 in the returned list
            self._has_kppp = len(k_edges_at_v_plus_plus[1]) == 1
            self._idx_kppp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus_plus[1][0]))
                if self._has_kppp
                else jnp.int32(-1)
            )

            # save the μ(v+2)
            self._has_mu_pp = True
            [mupp], _ = H.graph.get_edges_at_k_vertex(self.v + 2)
            self._idx_mu_pp = jnp.int32(H.graph.edge_to_index(mupp))

        # check if there is a k-vertex to the left of v - 1
        has_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v - 1, "left")

        # if it does not, fast track
        if not has_left_k_vertex:
            # mark that there are no k_---(v)
            self._has_kmmm = False
            self._idx_kmmm = jnp.int32(-1)

            # and consequently also no μ(v-2)
            self._has_mu_mm = False
            self._idx_mu_mm = jnp.int32(-1)
        else:
            # ther is a k-vertex to the left of v-1, so get the k-edges attached to it
            k_edges_at_v_minus_minus = H.graph.get_kminus_kplus(self.v - 2)

            # here, we need the inner k-edge, so we look up the index 0 in the returned list
            self._has_kmmm = len(k_edges_at_v_minus_minus[0]) == 1
            self._idx_kmmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus_minus[0][0]))
                if self._has_kmmm
                else jnp.int32(-1)
            )

            # save the μ(v-2)
            self._has_mu_mm = True
            [mumm], _ = H.graph.get_edges_at_k_vertex(self.v - 2)
            self._idx_mu_mm = jnp.int32(H.graph.edge_to_index(mumm))

        #
        #
        # μ-edges at neighbours (by id)
        # we fetch the μ-edges at the neighbour k-vertices by their k-vertex ids, this is
        # robust because it does not rely on any positional convention—only on the graph query

        # μ(v-1)
        [mu_left] = H.graph.get_edges_at_k_vertex(v_minus)[0]

        # μ(v+1)
        [mu_right] = H.graph.get_edges_at_k_vertex(v_plus)[0]

        #
        #
        # flatten to σ-column indices
        # every edge (μ or k) occupies exactly one scalar position in σ (U(1)), and the graph has a
        # bijection "edge -> index in σ", so we convert all used edges to their flattened indices

        self._idx_mu = jnp.int32(H.graph.edge_to_index(mu_here))
        self._idx_mu_m = jnp.int32(H.graph.edge_to_index(mu_left))
        self._idx_mu_p = jnp.int32(H.graph.edge_to_index(mu_right))
        self._idx_km = jnp.int32(H.graph.edge_to_index(e_minus))
        self._idx_kp = jnp.int32(H.graph.edge_to_index(e_plus))

        #
        #
        # fixed connectivity size:
        #   H_C: 2 off-diag (μ±2) + 1 diag = 3
        #   H_R: 8 off-diag
        #   H_L: 8 off-diag
        self._nconn = 19

    @property
    def is_hermitian(self) -> bool:
        """Return False: the bare Euclidean constraint is not Hermitian in this form."""
        return False

    @property
    def dtype(self):
        """All matrix elements are real, use float64 for numerical stability."""
        return jnp.float64

    @staticmethod
    def _wrap_mod(
        vals: jnp.ndarray,
        delta: jnp.ndarray,
        mod_span: Union[int, float],
        state_min: Union[int, float],
    ) -> jnp.ndarray:
        """
        Modular wrapping of quantum numbers into [state_min, state_max].

        Works for both symmetric ([-N,N]) and positive ([0, 2N-1]) Hilbert spaces.
        """
        # return (vals + delta + cutoff) % mod_span - cutoff
        return ((vals + delta - state_min) % mod_span) + state_min

    @staticmethod
    def _sgn_delta_over_2(mu: jnp.ndarray, half_delta: float) -> jnp.ndarray:
        """
        Three-valued (piecewise-linear) sign sgn_{δ/2}(μ) used in ΔφV.

        For δ=2 ⇒ δ/2=1, so on integer μ:
            sgn_{1}(μ) = +1 if μ ≥ 1
                         0 if μ = 0
                         −1 if μ ≤ −1

        this returns a (BN,) float array in {−1.0, 0.0, +1.0} (on integer μ) or linear
        interpolation otherwise
        """
        one = half_delta
        return jnp.where(
            mu >= one, 1.0, jnp.where(mu <= -one, -1.0, mu.astype(jnp.float64) / one)
        )

    def _gamma_phi_full(
        self,
        K_left: jnp.ndarray,
        K_center: jnp.ndarray,
        K_right: jnp.ndarray,
        mu_center: jnp.ndarray,
        mu_minus: jnp.ndarray,
        mu_plus: jnp.ndarray,
    ):
        r"""
        Discrete spin connection Γ_φ(v) via a centered difference stencil.

        Definitions:
            E^x(v) := K(v)/2
            int_+ E^φ(v) := (μ(v) + μ(v+1))/2
            int_- E^φ(v) := (μ(v) + μ(v-1))/2
            E_+^x'(v) ≈ E^x(v+1) − E^x(v)
            E_-^x'(v) ≈ E^x(v) - E^x(v-1)
            Γ_φ(v)  ≈ − (E_+^x'(v) / (int_+ E^φ(v))) + (E_-^x'(v) / (int_- E^φ(v))) / 4
        :return:
        """

        # just easier this way...
        _dtype = jnp.float64

        # E^x at v-1, v, v+1
        Ex_m = 0.5 * K_left.astype(_dtype)
        Ex_c = 0.5 * K_center.astype(_dtype)
        Ex_p = 0.5 * K_right.astype(_dtype)

        # one-sided Ex' on each link

        # E^x(v+1) - E^x(v)
        Ex_prime_plus = Ex_p - Ex_c

        # E^x(v) - E^x(v-1)
        Ex_prime_minus = Ex_c - Ex_m

        # Link-averaged μ on each side
        abs_mu_c = mu_center.astype(_dtype)
        abs_mu_p = mu_plus.astype(_dtype)
        abs_mu_m = mu_minus.astype(_dtype)

        int_plus = 0.5 * (abs_mu_c + abs_mu_p)
        int_minus = 0.5 * (abs_mu_c + abs_mu_m)

        int_plus = jnp.maximum(int_plus, 1e-12)
        int_minus = jnp.maximum(int_minus, 1e-12)

        #
        #
        #   option for safe divisions: drop term when its link average is exactly zero
        # term_plus = jnp.where(int_plus > 0.0, -Ex_prime_plus / int_plus, 0.0)
        # term_minus = jnp.where(int_minus > 0.0, Ex_prime_minus / int_minus, 0.0)
        # return 0.25 * (term_plus + term_minus)

        return 0.25 * ((-Ex_prime_plus / int_plus) + (Ex_prime_minus / int_minus))

    def _gamma_phi(
        self,
        K_left: jnp.ndarray,
        K_center: jnp.ndarray,
        K_right: jnp.ndarray,
        mu_center: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""
        Discrete spin connection Γ_φ(v) via a centered difference stencil.

        Definitions:
            E^x(v) := K(v)/2
            E^φ(v) := |μ(v)|
            E^x'(v) ≈ E^x(v+1) − E^x(v−1)
            Γ_φ(v)  ≈ − E^x'(v) / (4 |E^φ(v)|)

        Notes
        -----
        - We compute E^x(v±1) from K(v±1), then form E^x'(v) and divide by 4|μ(v)|.
        - A small floor is applied to |μ(v)| to avoid NaNs at μ=0, this is safe because
          ΔφV ∝ sgn_{δ/2}(μ) vanishes at μ=0 anyway
        """

        # E^x at v−1
        Ex_m = 0.5 * K_left.astype(jnp.float64)

        # E^x at v+1
        Ex_p = 0.5 * K_right.astype(jnp.float64)

        # centered first difference
        Ex_prime = Ex_p - Ex_m

        denom = 4.0 * jnp.maximum(jnp.abs(mu_center).astype(jnp.float64), 1e-12)

        return -Ex_prime / denom

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Enumerate the 19 connected configurations σ' and their matrix elements at vertex v.

        Right-branch 8-term pattern (applied to k_+(v), μ(v), μ(v+1)):
            (Δk, Δμ_c, Δμ_p, sign)
            (+2, +1, +1, +), (+2, +1, −1, −), (+2, −1, +1, +), (+2, −1, −1, −),
            (−2, +1, +1, −), (−2, +1, −1, +), (−2, −1, +1, −), (−2, −1, −1, +)

        Left-branch is the same sign table applied to k_-(v), μ(v−1), μ(v).
        All right/left magnitudes are |mel| = ΔφV(v).
        """

        # normalize input to (B, N, D)
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        # pre-allocate outputs with the same base σ
        # (padded, fixed 19 `_nconn` connections, will overwrite per slot)
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D)).copy()
        mels = jnp.zeros((B, N, self._nconn), dtype=self.dtype)

        # flatten batch-block for convenience
        BN = B * N
        sig = σ.reshape(BN, D)

        #
        #
        # gather indices of DoFs touched by H_v

        # μ(v)
        imc = int(self._idx_mu)

        # μ(v−1)
        imm = int(self._idx_mu_m)

        # μ(v+1)
        imp = int(self._idx_mu_p)

        # μ(v+2)
        impp = int(self._idx_mu_pp)

        # μ(v-2)
        immm = int(self._idx_mu_mm)

        # k_−(v)
        ikm = int(self._idx_km)

        # k_+(v)
        ikp = int(self._idx_kp)

        #
        #
        # read current values (BN,)

        mu_c = sig[:, imc]
        mu_m = sig[:, imm]
        mu_p = sig[:, imp]
        k_m = sig[:, ikm]
        k_p = sig[:, ikp]

        if self._has_mu_mm:
            mu_mm = sig[:, immm]
        else:
            mu_mm = jnp.asarray(0, dtype=jnp.int32)

        if self._has_mu_pp:
            mu_pp = sig[:, impp]
        else:
            # dev: this is a 0-dim array, and it will be added later to a multi-dim array coming
            #      from sig. JAX will promote this 0-dim array to match that, so this is safe for
            #      now but should be noted
            # dev: this is perhaps too harsh physically, but I dont know what you want to do here
            mu_pp = jnp.asarray(0, dtype=jnp.int32)

        #
        #
        # build ΔxV(v) and ΔφV(v) from K(v) and μ(v)

        # K(v) is the k-sum: k_−(v) + k_+(v)
        K = (k_m + k_p).astype(jnp.int32)

        # dev: uncomment the next line to mod wrap the K(v)
        # K = self._wrap_mod(K, 0, self._mod_span, self._state_min)

        #
        #
        # √|K|, √|K±1| appear in ΔxV and ΔφV, we compute them once in float64

        # √|K|
        sqrtK = jnp.sqrt(jnp.abs(K).astype(jnp.float64))

        # √|K+1|
        Kp1 = K + 1

        # dev: uncomment the next line to mod wrap the K(v) + 1
        # Kp1 = self._wrap_mod(Kp1, 0, self._mod_span, self._state_min)

        sqrtKp1 = jnp.sqrt(jnp.abs(Kp1).astype(jnp.float64))

        # √|K-1|
        Km1 = K - 1

        # dev: uncomment the next line to mod wrap the K(v) - 1
        # Km1 = self._wrap_mod(Km1, 0, self._mod_span, self._state_min)

        sqrtKm1 = jnp.sqrt(jnp.abs(Km1).astype(jnp.float64))

        # ΔxV(v) := |μ(v)| [√|K+1| − √|K−1|]
        abs_mu = jnp.abs(mu_c).astype(jnp.float64)

        # dev: mod wrap here too?
        d_sqrt = sqrtKp1 - sqrtKm1

        # ΔxV(v)
        delta_xV = abs_mu * d_sqrt

        # ΔφV(v) := sgn_{δ/2}(μ(v)) √|K(v)|, with δ/2 = 1 here.
        sgn_half = self._sgn_delta_over_2(mu_c, self._half_delta)

        # ΔφV(v)
        delta_phiV = sgn_half * sqrtK

        # spin connection Γφ and forward difference Γ′φ (if enabled)
        if self._use_gamma:
            #
            #
            #   here, we need to build the full Γφ:
            #       E^x(v) := K(v)/2
            #       int_+ E^φ(v) := (μ(v) + μ(v+1))/2
            #       int_- E^φ(v) := (μ(v) + μ(v-1))/2
            #       E_+^x'(v) ≈ E^x(v+1) − E^x(v)
            #       E_-^x'(v) ≈ E^x(v) − E^x(v-1)
            #       Γ_φ(v)  ≈ − (E_+^x'(v) / (int_+ E^φ(v))) + (E_-^x'(v) / (int_- E^φ(v))) / 4

            #   which means we need:
            #       - K(v) for E^x(v)
            #           - for this we need k_-(v) and k_+(v)
            #       - K(v+1) for E^x(v+1)
            #           - for this we need k_+(v) and k_++(v)
            #       - K(v-1) for E^x(v-1)
            #           - for this we need k_-(v) and k_--(v)
            #       - μ(v)
            #       - μ(v+1)
            #       - μ(v-1)
            #
            # we fall back to using the single adjacent k-edge (k_m at v−1, k_p at v+1), consistent
            # with the graph's boundary

            #
            #
            #   build K(v) for E^x(v)
            #   for this we need k_-(v) and k_+(v)
            #   ikm is k_-(v) and ikp is k_+(v)
            _ksum_center = (sig[:, ikm] + sig[:, ikp]).astype(jnp.int32)

            #
            #
            #   build K(v+1) for E^x(v+1)
            #   for this we need k_+(v) and k_++(v)
            if self._has_kpp:
                # ikp is k_+(v) and _idx_kpp is k_++(v)
                _ksum_right = (sig[:, ikp] + sig[:, int(self._idx_kpp)]).astype(
                    jnp.int32
                )
            else:
                # case that there are no outer edges
                _ksum_right = (sig[:, ikp] + self._outer_kp_constant).astype(jnp.int32)

            #
            #
            #   build K(v-1) for E^x(v-1)
            #   for this we need k_-(v) and k_--(v)
            if self._has_kmm:
                # ikm is k_-(v) and _idx_km_outer is k_--(v)
                _ksum_left = (sig[:, int(self._idx_kmm)] + sig[:, ikm]).astype(
                    jnp.int32
                )
            else:
                # case that there are no outer edges
                _ksum_left = (sig[:, ikm] + self._outer_km_constant).astype(jnp.int32)

            #
            #
            #   compute Γφ(v)
            gamma = self._gamma_phi_full(
                K_left=_ksum_left,
                K_center=_ksum_center,
                K_right=_ksum_right,
                mu_center=mu_c,
                mu_minus=mu_m,
                mu_plus=mu_p,
            )

            gamma2 = gamma * gamma

            #
            #
            #   forward difference approximation for Γ′φ(v)
            #   for this we need to compute Γφ(v+1), which means we need:
            #       Γ_φ(v+1)  ≈ − (E_+^x'(v+1) / (int_+ E^φ(v+1))) + (E_-^x'(v+1) / (int_- E^φ(v+1))) / 4
            #
            #   where generally:
            #       E^x(v) := K(v)/2
            #
            #   and now we need to assemble:
            #       int_+ E^φ(v+1) := (μ(v+1) + μ(v+2))/2
            #       int_- E^φ(v+1) := (μ(v+1) + μ(v))/2
            #       E_+^x'(v+1) ≈ E^x(v+2) − E^x(v+1)
            #       E_-^x'(v+1) ≈ E^x(v+1) − E^x(v)
            #
            #   which means we need:
            #       - K(v) for E^x(v) to assemble E_-^x'(v+1)
            #           - for this we need k_-(v) and k_+(v)
            #       - K(v+1) for E^x(v+1) to assemble E_-^x'(v+1) and E_+^x'(v+1)
            #           - for this we need k_+(v) and k_++(v)
            #       - K(v+2) for E^x(v+2) to assemble E_+^x'(v+1)
            #           - for this we need k_-(v) and k_--(v)
            #       - μ(v+1)
            #       - μ(v+2)
            #       - μ(v)

            #
            #
            #   K(v) and K(v+1) are already computed above, so we will reuse them, compute K(v+2)

            #
            #
            #   build K(v+2) for E^x(v+2)
            #   for this we need k_++(v) and k_+++(v)

            if self._has_kpp:
                # k_++(v) exists
                kp_contrib = sig[:, int(self._idx_kpp)]
            else:
                kp_contrib = self._outer_kp_constant

            if self._has_kppp:
                # k_+++(v) exists
                kpp_contrib = sig[:, int(self._idx_kppp)]
            else:
                kpp_contrib = self._outer_kp_constant

            _ksum_right_right = (kp_contrib + kpp_contrib).astype(jnp.int32)

            #
            #
            #   assemble:
            #   Γ_φ(v+1)  ≈ − (E_+^x'(v+1) / (int_+ E^φ(v+1))) + (E_-^x'(v+1) / (int_- E^φ(v+1))) / 4
            #
            #   here:
            #       - K_center will build E^x(v+1)
            #       - K_left will build E^x(v)
            #       - K_right will build E^x(v+1)
            #       - mu_center will be μ(v+1)
            #       - mu_minus will be μ(v)
            #       - mu_plus will be μ(v+2)
            gamma_v_plus_one = self._gamma_phi_full(
                K_left=_ksum_center,
                K_center=_ksum_right,
                K_right=_ksum_right_right,
                mu_center=mu_p,
                mu_minus=mu_c,
                mu_plus=mu_pp,
            )

            gamma_prime = gamma_v_plus_one - gamma

        else:
            #
            #
            # no Γ-physics branch
            # treat Γ_φ = Γ′_φ = 0. The non-Γ diagonal terms still apply.
            gamma = jnp.zeros_like(delta_xV)
            gamma2 = gamma
            gamma_prime = gamma

        # helper: apply shifts with modular wrap on a selected set of columns
        def apply_shifts(
            base_sigma: jnp.ndarray, comps: jnp.ndarray, deltas: jnp.ndarray
        ):
            """
            In-place-like (functional) update of selected DoFs with modular wrapping

            :param base_sigma: (BN, D) integer array
            :param comps: (nC,) int32 array of column indices to update
            :param deltas: (BN, nC) int32 additive shifts

            :return updated : (BN, D) integer array, dtype preserved from base_sigma
            """
            if comps.size == 0:
                return base_sigma

            # (BN, nC)
            cur = base_sigma[:, comps]

            # (BN, nC)
            # upd = self._wrap_mod(cur, deltas, self.cutoff, self._mod_span)
            upd = self._wrap_mod(cur, deltas, self._mod_span, self._state_min)
            upd = upd.astype(base_sigma.dtype)

            return base_sigma.at[:, comps].set(upd)

        # helper: record one connection across the entire (B,N)
        # running index along the 19-slot axis
        idx_write = 0

        def write_conn(sig_variant_bn: jnp.ndarray, mel_bn: jnp.ndarray):
            """
            Place a σ' and its matrix element in the next slot for all (B, N) items.

            :param sig_variant_bn: (BN, D) integers
            :param mel_bn: (BN,)   floats
            """

            nonlocal idx_write, σp, mels
            sig_bnd = sig_variant_bn.reshape(B, N, D)
            σp = σp.at[:, :, idx_write, :].set(sig_bnd)
            mels = mels.at[:, :, idx_write].set(mel_bn.reshape(B, N).astype(self.dtype))
            idx_write += 1

        #
        #
        #   H_C
        #   2 off-diagonal μ-shifts (±4) + 1 diagonal with Γ-pieces
        #
        #   NOTE: the prefactors (which include an Immirzi^(3/2)) from equation 48 are not
        #         implemented here

        #
        #
        # only μ(v) changes in the H_C off-diagonals, with δ=2 and δ/2=1, the central action
        # moves μ by ±2 units of δ/2 -> ±2 on the μ-lattice -> ±4 in raw integer storage
        comps_mu = jnp.asarray([imc], jnp.int32)

        # with δ=2, μ-shifts for the central off-diags are ±2 (i.e., ±2*(δ/2))
        plus_delta = jnp.full((BN, 1), 2 * self.delta, jnp.int32)
        minus_delta = jnp.full((BN, 1), -2 * self.delta, jnp.int32)

        #
        #
        # off-diagonal configurations σ' and their matrix elements (both mels are ΔxV)

        # |σ; μ(v) -> μ(v)+2|
        sig_mu_p2 = apply_shifts(sig, comps_mu, plus_delta)

        # |σ; μ(v) -> μ(v)−2|
        sig_mu_m2 = apply_shifts(sig, comps_mu, minus_delta)

        # slot 0
        write_conn(sig_mu_p2, delta_xV)

        # slot 1
        write_conn(sig_mu_m2, delta_xV)

        #
        #
        # diagonal element for H_C:
        #   -2 ΔxV  - 4 γ^2 δ^2 (1 - Γ^2) ΔxV  - 4 γ^2 δ^2 Γ′ ΔφV
        # numerically with δ=2 this means: -2 ΔxV - 16 (1-Γ^2) ΔxV - 16 Γ′ ΔφV   (γ = 1.0)
        diag = (
            -2.0 * delta_xV
            - 4.0 * (self.immirzi**2) * (self.delta**2) * (1.0 - gamma2) * delta_xV
            - 4.0 * (self.immirzi**2) * (self.delta**2) * gamma_prime * delta_phiV
        )

        # slot 2
        write_conn(sig, diag)

        #
        #
        #   H_R
        #   acts on the triplet (k_+(v), μ(v), μ(v+1))
        #
        # we apply the canonical 8-term table where each term shifts:
        #   - k_+(v) by ±2   (polymerised k-holonomy step),
        #   - μ(v) and μ(v+1) by ±1 each (one δ/2 unit each in μ),
        # and uses an alternating sign pattern.
        # the magnitude for all eight is ΔφV(v)
        #
        #   NOTE: the prefactors (which include an Immirzi^(3/2)) from equation 49 are not
        #         implemented here

        # column of k_+(v)
        comps_kp = jnp.asarray([ikp], jnp.int32)

        # columns of μ(v), μ(v+1)
        comps_musR = jnp.asarray([imc, imp], jnp.int32)

        # magnitude for all 8 moves on the right
        base_R = delta_phiV

        # eight right terms as (Δk, Δμ_c, Δμ_p, sign), applied to (k_+, μ_c, μ_p) respectively
        right_terms = (
            (+2 * (self.delta / 2), +self._half_delta, +self._half_delta, +1),
            (+2 * (self.delta / 2), +self._half_delta, -self._half_delta, -1),
            (+2 * (self.delta / 2), -self._half_delta, +self._half_delta, +1),
            (+2 * (self.delta / 2), -self._half_delta, -self._half_delta, -1),
            (-2 * (self.delta / 2), +self._half_delta, +self._half_delta, -1),
            (-2 * (self.delta / 2), +self._half_delta, -self._half_delta, +1),
            (-2 * (self.delta / 2), -self._half_delta, +self._half_delta, -1),
            (-2 * (self.delta / 2), -self._half_delta, -self._half_delta, +1),
        )

        for dk, dmu_c, dmu_p, sgn in right_terms:
            del_k = jnp.full((BN, 1), dk, jnp.int32)

            # (BN,2)
            del_mus = jnp.stack(
                (jnp.full((BN,), dmu_c, jnp.int32), jnp.full((BN,), dmu_p, jnp.int32)),
                axis=1,
            )

            sig_tmp = apply_shifts(sig, comps_kp, del_k)
            sig_tmp = apply_shifts(sig_tmp, comps_musR, del_mus)
            write_conn(sig_tmp, sgn * base_R)

        #
        #
        #   H_L
        #   mirror of H_R on the triplet (k_−(v), μ(v−1), μ(v))
        #
        #   NOTE: the prefactors (which include an Immirzi^(3/2)) from equation 49 are not
        #         implemented here

        # column of k_−(v)
        comps_km = jnp.asarray([ikm], jnp.int32)

        # columns of μ(v−1), μ(v)
        comps_musL = jnp.asarray([imm, imc], jnp.int32)

        # identical magnitude
        base_L = delta_phiV

        # same pattern applied to (k_-, μ_m, μ_c)
        left_terms = right_terms

        for dk, dmu_m, dmu_c, sgn in left_terms:
            del_k = jnp.full((BN, 1), dk, jnp.int32)

            del_mus = jnp.stack(
                (jnp.full((BN,), dmu_m, jnp.int32), jnp.full((BN,), dmu_c, jnp.int32)),
                axis=1,
            )

            sig_tmp = apply_shifts(sig, comps_km, del_k)
            sig_tmp = apply_shifts(sig_tmp, comps_musL, del_mus)
            write_conn(sig_tmp, sgn * base_L)

        # restore original shape convention if the input was (N, D)
        if single:
            return σp[0], mels[0]

        return σp, mels


class SphericalVertexConstraintBojowaldSwiderskiFixed(ComputationalOperator):
    r"""
    A ComputationalOperator implementing the single-vertex Euclidean constraint
    \( H_v = H_L(v) + H_C(v) + H_R(v) \) at a chosen inner k-vertex \(v\).

    Notes
    -----
    - This operator is not Hermitian in this bare Euclidean form. If a Hermitian positive
      operator is desired, wrap with a squaring construction upstream (e.g., `Squared(...)`).
    - All overall constants are taken to be absorbed into units with \(\gamma=1\) and \(\delta=2\).
    """

    def __init__(
        self,
        H,
        vertex: int,
        *,
        include_gamma_terms: bool = True,
        delta: int = 2,  # dev: strictly an integer, see the first dev comment below
        outer_km_constant: int = 0,
        outer_kp_constant: int = 0,
        immirzi: float = 1.0,
    ):

        #
        #
        # setup and class attributes

        # initialize parent with NetKet Hilbert
        super().__init__(H.hilbert)

        # enforce U(1), one scalar per edge
        if int(H.core.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )

        # HalfLadderGraph type check
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires the graph to be an instance of "
                f"`{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}` instead."
            )

        # cache host and invariants
        self._H = H

        # total number of scalar DoFs
        self.D = int(H.size)

        # integer cutoff for modular wrapping
        self.cutoff = int(H.cutoff)

        # modulus for wrap: values in [−cutoff, cutoff]
        # self._mod_span = int(2 * self.cutoff + 1)

        # get the allowed quantum numbers
        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )
        self._state_step = jnp.asarray(
            int(H.allowed_basis_states.step), dtype=jnp.int64
        )
        self._state_max = jnp.asarray(
            int(self._state_min + (self._mod_span - 1) * self._state_step),
            dtype=jnp.int64,
        )

        # polymer step δ, fixed in this implementation
        # dev: this is now strictly an integer when working with delta = 2
        #      this can be changed, but some JAX arrays in the _get_conn_padded_kernel will need
        #      their dtype changed as well
        if type(delta) != int:
            raise ValueError(
                f"Currently, only integer (type `int`) values for `delta` is allowed, "
                f"got `{type(delta).__name__}` instead."
            )

        self.delta: int = delta

        # = 1.0 with δ=2, used in sgn_{δ/2}(μ)
        # dev: un-int this when going for general delta, see the dev comment above
        self._half_delta = int(self.delta / 2)

        # a flag to include gamma dependent terms or not
        self._use_gamma = bool(include_gamma_terms)

        # a constant to be added for constructing K(v) when there is no outer right edge
        self._outer_kp_constant = jnp.asarray(outer_kp_constant)

        # a constant to be added for constructing K(v) when there is no outer left edge
        self._outer_km_constant = jnp.asarray(outer_km_constant)

        # save the Immirzi parameter value
        self.immirzi = immirzi

        #
        #
        # validate vertex: it must be an "inner" k-vertex
        # the Graph object exposes H.g.vertices_k in order along the ladder spine
        # (e.g., [0,1,2,3,...]) and "inner" means it has both left and right neighbours
        verts_k = list(H.graph.vertices_k)
        inner = verts_k[2:-2]
        if vertex not in inner:
            raise ValueError(
                f"vertex={vertex} must be an inner k-vertex. Valid vertices are: {inner}"
            )

        # all clear, set it
        self.v = int(vertex)

        #
        #
        # resolve the three edges touching v: one μ-edge at v, two k-edges (left/right)
        # for each k-vertex id v, the Graph API provides:
        #   get_edges_at_k_vertex(v) -> ([μ_edge], [k_minus_edge, k_plus_edge])
        # where:
        #   - μ_edge is the vertical μ-edge "stuck into" vertex v,
        #   - k_minus_edge is the k-edge coming from v-1 -> v,
        #   - k_plus_edge  is the k-edge going from v -> v+1.
        # The graph stores a k-edge as a tuple/list (a, b, key) with a < b. Hence:
        #   - for the "left" incident k-edge at v, its b == v (since it ends at v)
        #   - for the "right" incident k-edge at v, its a == v (since it starts at v)
        #
        # dev: for this HalfLadderGraph type, it is true that a < b. However, if a user
        #      custom implements a half ladder, the following code will break, there is no
        #      guaranteed contract for a being < b
        [mu_here], k_two = H.graph.get_edges_at_k_vertex(self.v)

        # disambiguate left/right k-edges using endpoints
        # left edge is (v-1, v, *), right edge is (v, v+1, *).
        left_candidates = [e for e in k_two if e[1] == self.v]
        right_candidates = [e for e in k_two if e[0] == self.v]

        if len(left_candidates) != 1 or len(right_candidates) != 1:
            raise RuntimeError(
                f"Could not disambiguate left/right k-edges at v={self.v}: {k_two}"
            )

        # k_-(v): (v-1, v, key)
        e_minus = left_candidates[0]

        # k_+(v): (v, v+1, key)
        e_plus = right_candidates[0]

        # neighbour vertex ids (needed both for μ-edges at neighbours and for K(v±1))
        # v-1
        v_minus = e_minus[0]

        # v+1
        v_plus = e_plus[1]

        #
        #
        #   get the next nearest k-edges to build k_--(v) and k_++(v)

        # check if there is a k-vertex to the right, we know it does because it has to be an
        # inner vertex, but does not harm
        has_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "right")

        # if it does not, fast track
        if not has_right_k_vertex:
            self._has_kpp = False
            self._idx_kpp = jnp.int32(-1)
        else:
            # it has a k-vertex to its right, so get the k-edges attached to it
            k_edges_at_v_plus = H.graph.get_kminus_kplus(self.v + 1)

            # here, we need the outer k-edge, so we look up the index 1 in the returned list
            self._has_kpp = len(k_edges_at_v_plus[1]) == 1
            self._idx_kpp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus[1][0]))
                if self._has_kpp
                else jnp.int32(-1)
            )

        # check if there is a k-vertex to the left, we know it does because it has to be an
        # inner vertex, but does not harm
        has_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "left")

        # if it does not, fast track
        if not has_left_k_vertex:
            self._has_kmm = False
            self._idx_kmm = jnp.int32(-1)
        else:
            # it has a k-vertex to its left, so get the k-edges attached to it
            k_edges_at_v_minus = H.graph.get_kminus_kplus(self.v - 1)

            # here, we need the inner k-edge, so we look up the index 0 in the returned list
            self._has_kmm = len(k_edges_at_v_minus[0]) == 1
            self._idx_kmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus[0][0]))
                if self._has_kmm
                else jnp.int32(-1)
            )

        #
        #
        #   get the next nearest k-edges to build k_---(v) and k_+++(v)

        # check if there is a k-vertex to the right of v + 1
        has_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v + 1, "right")

        # if it does not, fast track
        if not has_right_k_vertex:
            # mark that there are no k_+++(v)
            self._has_kppp = False
            self._idx_kppp = jnp.int32(-1)

            # and consequently also no μ(v+2)
            self._has_mu_pp = False
            self._idx_mu_pp = jnp.int32(-1)
        else:
            # there is a k-vertex to the right of v+1, so get the k-edges attached to it
            k_edges_at_v_plus_plus = H.graph.get_kminus_kplus(self.v + 2)

            # here, we need the outer k-edge, so we look up the index 1 in the returned list
            self._has_kppp = len(k_edges_at_v_plus_plus[1]) == 1
            self._idx_kppp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus_plus[1][0]))
                if self._has_kppp
                else jnp.int32(-1)
            )

            # save the μ(v+2)
            self._has_mu_pp = True
            [mupp], _ = H.graph.get_edges_at_k_vertex(self.v + 2)
            self._idx_mu_pp = jnp.int32(H.graph.edge_to_index(mupp))

        # check if there is a k-vertex to the left of v - 1
        has_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v - 1, "left")

        # if it does not, fast track
        if not has_left_k_vertex:
            # mark that there are no k_---(v)
            self._has_kmmm = False
            self._idx_kmmm = jnp.int32(-1)

            # and consequently also no μ(v-2)
            self._has_mu_mm = False
            self._idx_mu_mm = jnp.int32(-1)
        else:
            # ther is a k-vertex to the left of v-1, so get the k-edges attached to it
            k_edges_at_v_minus_minus = H.graph.get_kminus_kplus(self.v - 2)

            # here, we need the inner k-edge, so we look up the index 0 in the returned list
            self._has_kmmm = len(k_edges_at_v_minus_minus[0]) == 1
            self._idx_kmmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus_minus[0][0]))
                if self._has_kmmm
                else jnp.int32(-1)
            )

            # save the μ(v-2)
            self._has_mu_mm = True
            [mumm], _ = H.graph.get_edges_at_k_vertex(self.v - 2)
            self._idx_mu_mm = jnp.int32(H.graph.edge_to_index(mumm))

        #
        #
        # μ-edges at neighbours (by id)
        # we fetch the μ-edges at the neighbour k-vertices by their k-vertex ids, this is
        # robust because it does not rely on any positional convention—only on the graph query

        # μ(v-1)
        [mu_left] = H.graph.get_edges_at_k_vertex(v_minus)[0]

        # μ(v+1)
        [mu_right] = H.graph.get_edges_at_k_vertex(v_plus)[0]

        #
        #
        # flatten to σ-column indices
        # every edge (μ or k) occupies exactly one scalar position in σ (U(1)), and the graph has a
        # bijection "edge -> index in σ", so we convert all used edges to their flattened indices

        self._idx_mu = jnp.int32(H.graph.edge_to_index(mu_here))
        self._idx_mu_m = jnp.int32(H.graph.edge_to_index(mu_left))
        self._idx_mu_p = jnp.int32(H.graph.edge_to_index(mu_right))
        self._idx_km = jnp.int32(H.graph.edge_to_index(e_minus))
        self._idx_kp = jnp.int32(H.graph.edge_to_index(e_plus))

        #
        #
        # fixed connectivity size:
        #   H_C: 2 off-diag (μ±2) + 1 diag = 3
        #   H_R: 8 off-diag
        #   H_L: 8 off-diag
        self._nconn = 19

    @property
    def is_hermitian(self) -> bool:
        """Return False: the bare Euclidean constraint is not Hermitian in this form."""
        return False

    @property
    def dtype(self):
        """All matrix elements are real, use float64 for numerical stability."""
        return jnp.float64

    @staticmethod
    def _wrap_mod(
        vals: jnp.ndarray,
        delta: jnp.ndarray,
        mod_span: Union[int, float],
        state_min: Union[int, float],
    ) -> jnp.ndarray:
        """
        Modular wrapping of quantum numbers into [state_min, state_max].

        Works for both symmetric ([-N,N]) and positive ([0, 2N-1]) Hilbert spaces.
        """
        # return (vals + delta + cutoff) % mod_span - cutoff
        return ((vals + delta - state_min) % mod_span) + state_min

    @staticmethod
    def _sgn_delta_over_2(mu: jnp.ndarray, half_delta: float) -> jnp.ndarray:
        """
        Three-valued (piecewise-linear) sign sgn_{δ/2}(μ) used in ΔφV.

        For δ=2 ⇒ δ/2=1, so on integer μ:
            sgn_{1}(μ) = +1 if μ ≥ 1
                         0 if μ = 0
                         −1 if μ ≤ −1

        this returns a (BN,) float array in {−1.0, 0.0, +1.0} (on integer μ) or linear
        interpolation otherwise
        """
        one = half_delta
        return jnp.where(
            mu >= one, 1.0, jnp.where(mu <= -one, -1.0, mu.astype(jnp.float64) / one)
        )

    def _gamma_phi_full(
        self,
        K_left: jnp.ndarray,
        K_center: jnp.ndarray,
        K_right: jnp.ndarray,
        mu_center: jnp.ndarray,
        mu_minus: jnp.ndarray,
        mu_plus: jnp.ndarray,
    ):
        r"""
        Discrete spin connection Γ_φ(v) via a centered difference stencil.

        Definitions:
            E^x(v) := K(v)/2
            int_+ E^φ(v) := (μ(v) + μ(v+1))/2
            int_- E^φ(v) := (μ(v) + μ(v-1))/2
            E_+^x'(v) ≈ E^x(v+1) − E^x(v)
            E_-^x'(v) ≈ E^x(v) - E^x(v-1)
            Γ_φ(v)  ≈ − (E_+^x'(v) / (int_+ E^φ(v))) + (E_-^x'(v) / (int_- E^φ(v))) / 4
        :return:
        """

        # just easier this way...
        _dtype = jnp.float64

        # E^x at v-1, v, v+1
        Ex_m = 0.5 * K_left.astype(_dtype)
        Ex_c = 0.5 * K_center.astype(_dtype)
        Ex_p = 0.5 * K_right.astype(_dtype)

        # one-sided Ex' on each link

        # E^x(v+1) - E^x(v)
        Ex_prime_plus = Ex_p - Ex_c

        # E^x(v) - E^x(v-1)
        Ex_prime_minus = Ex_c - Ex_m

        # Link-averaged μ on each side
        abs_mu_c = mu_center.astype(_dtype)
        abs_mu_p = mu_plus.astype(_dtype)
        abs_mu_m = mu_minus.astype(_dtype)

        int_plus = 0.5 * (abs_mu_c + abs_mu_p)
        int_minus = 0.5 * (abs_mu_c + abs_mu_m)

        int_plus = jnp.maximum(int_plus, 1e-12)
        int_minus = jnp.maximum(int_minus, 1e-12)

        #
        #
        #   option for safe divisions: drop term when its link average is exactly zero
        # term_plus = jnp.where(int_plus > 0.0, -Ex_prime_plus / int_plus, 0.0)
        # term_minus = jnp.where(int_minus > 0.0, Ex_prime_minus / int_minus, 0.0)
        # return 0.25 * (term_plus + term_minus)

        return 0.25 * ((-Ex_prime_plus / int_plus) + (Ex_prime_minus / int_minus))

    def _gamma_phi(
        self,
        K_left: jnp.ndarray,
        K_center: jnp.ndarray,
        K_right: jnp.ndarray,
        mu_center: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""
        Discrete spin connection Γ_φ(v) via a centered difference stencil.

        Definitions:
            E^x(v) := K(v)/2
            E^φ(v) := |μ(v)|
            E^x'(v) ≈ E^x(v+1) − E^x(v−1)
            Γ_φ(v)  ≈ − E^x'(v) / (4 |E^φ(v)|)

        Notes
        -----
        - We compute E^x(v±1) from K(v±1), then form E^x'(v) and divide by 4|μ(v)|.
        - A small floor is applied to |μ(v)| to avoid NaNs at μ=0, this is safe because
          ΔφV ∝ sgn_{δ/2}(μ) vanishes at μ=0 anyway
        """

        # E^x at v−1
        Ex_m = 0.5 * K_left.astype(jnp.float64)

        # E^x at v+1
        Ex_p = 0.5 * K_right.astype(jnp.float64)

        # centered first difference
        Ex_prime = Ex_p - Ex_m

        denom = 4.0 * jnp.maximum(jnp.abs(mu_center).astype(jnp.float64), 1e-12)

        return -Ex_prime / denom

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        BRA-convention connectivity:
        returns σ' and mels such that mels = <σ|H|σ'>.

        This is the transpose of the naive "act-on-ket" enumeration.
        For each off-diagonal ket-move σ' -> σ, we now return the preimage σ'
        obtained by applying the inverse shifts to the input σ (row state),
        and we evaluate the off-diagonal prefactors on that preimage.
        """

        # normalize input to (B, N, D)
        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        # pre-allocate outputs with the same base σ
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D)).copy()
        mels = jnp.zeros((B, N, self._nconn), dtype=self.dtype)

        # flatten batch-block for convenience
        BN = B * N
        sig = σ.reshape(BN, D)

        #
        #
        # gather indices of DoFs touched by H_v

        imc = int(self._idx_mu)  # μ(v)
        imm = int(self._idx_mu_m)  # μ(v−1)
        imp = int(self._idx_mu_p)  # μ(v+1)

        impp = int(self._idx_mu_pp)  # μ(v+2)
        immm = int(self._idx_mu_mm)  # μ(v−2)

        ikm = int(self._idx_km)  # k_−(v)
        ikp = int(self._idx_kp)  # k_+(v)

        #
        #
        # read current (ROW) values (BN,)

        mu_c = sig[:, imc]
        mu_m = sig[:, imm]
        mu_p = sig[:, imp]
        k_m = sig[:, ikm]
        k_p = sig[:, ikp]

        if self._has_mu_mm:
            mu_mm = sig[:, immm]
        else:
            mu_mm = jnp.asarray(0, dtype=jnp.int32)

        if self._has_mu_pp:
            mu_pp = sig[:, impp]
        else:
            mu_pp = jnp.asarray(0, dtype=jnp.int32)

        #
        #
        # helper: apply shifts with modular wrap on a selected set of columns

        def apply_shifts(
            base_sigma: jnp.ndarray, comps: jnp.ndarray, deltas: jnp.ndarray
        ):
            if comps.size == 0:
                return base_sigma

            cur = base_sigma[:, comps]
            upd = self._wrap_mod(cur, deltas, self._mod_span, self._state_min)
            upd = upd.astype(base_sigma.dtype)
            return base_sigma.at[:, comps].set(upd)

        #
        #
        # local helpers to compute prefactors on an arbitrary BN×D state

        def compute_delta_xV(sig_bn: jnp.ndarray) -> jnp.ndarray:
            mu_c_ = sig_bn[:, imc]
            k_m_ = sig_bn[:, ikm]
            k_p_ = sig_bn[:, ikp]

            K_ = (k_m_ + k_p_).astype(jnp.int32)

            sqrtKp1_ = jnp.sqrt(jnp.abs(K_ + 1).astype(jnp.float64))
            sqrtKm1_ = jnp.sqrt(jnp.abs(K_ - 1).astype(jnp.float64))

            abs_mu_ = jnp.abs(mu_c_).astype(jnp.float64)
            return abs_mu_ * (sqrtKp1_ - sqrtKm1_)

        def compute_delta_phiV(sig_bn: jnp.ndarray) -> jnp.ndarray:
            mu_c_ = sig_bn[:, imc]
            k_m_ = sig_bn[:, ikm]
            k_p_ = sig_bn[:, ikp]

            K_ = (k_m_ + k_p_).astype(jnp.int32)
            sqrtK_ = jnp.sqrt(jnp.abs(K_).astype(jnp.float64))

            sgn_half_ = self._sgn_delta_over_2(mu_c_, self._half_delta)
            return sgn_half_ * sqrtK_

        #
        #
        # ROW-derived quantities for the DIAGONAL ONLY

        K_row = (k_m + k_p).astype(jnp.int32)

        sqrtK_row = jnp.sqrt(jnp.abs(K_row).astype(jnp.float64))
        sqrtKp1_row = jnp.sqrt(jnp.abs(K_row + 1).astype(jnp.float64))
        sqrtKm1_row = jnp.sqrt(jnp.abs(K_row - 1).astype(jnp.float64))

        abs_mu_row = jnp.abs(mu_c).astype(jnp.float64)
        delta_xV_row = abs_mu_row * (sqrtKp1_row - sqrtKm1_row)

        sgn_half_row = self._sgn_delta_over_2(mu_c, self._half_delta)
        delta_phiV_row = sgn_half_row * sqrtK_row

        # spin connection Γφ and forward difference Γ′φ (row-state)
        if self._use_gamma:
            _ksum_center = (sig[:, ikm] + sig[:, ikp]).astype(jnp.int32)

            if self._has_kpp:
                _ksum_right = (sig[:, ikp] + sig[:, int(self._idx_kpp)]).astype(
                    jnp.int32
                )
            else:
                _ksum_right = (sig[:, ikp] + self._outer_kp_constant).astype(jnp.int32)

            if self._has_kmm:
                _ksum_left = (sig[:, int(self._idx_kmm)] + sig[:, ikm]).astype(
                    jnp.int32
                )
            else:
                _ksum_left = (sig[:, ikm] + self._outer_km_constant).astype(jnp.int32)

            gamma = self._gamma_phi_full(
                K_left=_ksum_left,
                K_center=_ksum_center,
                K_right=_ksum_right,
                mu_center=mu_c,
                mu_minus=mu_m,
                mu_plus=mu_p,
            )
            gamma2 = gamma * gamma

            if self._has_kpp:
                kp_contrib = sig[:, int(self._idx_kpp)]
            else:
                kp_contrib = self._outer_kp_constant

            if self._has_kppp:
                kpp_contrib = sig[:, int(self._idx_kppp)]
            else:
                kpp_contrib = self._outer_kp_constant

            _ksum_right_right = (kp_contrib + kpp_contrib).astype(jnp.int32)

            gamma_v_plus_one = self._gamma_phi_full(
                K_left=_ksum_center,
                K_center=_ksum_right,
                K_right=_ksum_right_right,
                mu_center=mu_p,
                mu_minus=mu_c,
                mu_plus=mu_pp,
            )
            gamma_prime = gamma_v_plus_one - gamma
        else:
            gamma = jnp.zeros_like(delta_xV_row)
            gamma2 = gamma
            gamma_prime = gamma

        #
        #
        # helper: record one connection across the entire (B,N)

        idx_write = 0

        def write_conn(sig_variant_bn: jnp.ndarray, mel_bn: jnp.ndarray):
            nonlocal idx_write, σp, mels
            sig_bnd = sig_variant_bn.reshape(B, N, D)
            σp = σp.at[:, :, idx_write, :].set(sig_bnd)
            mels = mels.at[:, :, idx_write].set(mel_bn.reshape(B, N).astype(self.dtype))
            idx_write += 1

        #
        #   H_C off-diagonals (BRA: use PREIMAGE states + evaluate ΔxV on them)

        comps_mu = jnp.asarray([imc], jnp.int32)

        plus_delta = jnp.full((BN, 1), 2 * self.delta, jnp.int32)
        minus_delta = jnp.full((BN, 1), -2 * self.delta, jnp.int32)

        # preimages:
        # ket-move: μ -> μ + plus_delta
        # bra row σ corresponds to the OUT state, so σ' = σ - plus_delta
        sig_mu_p2_pre = apply_shifts(sig, comps_mu, -plus_delta)

        # ket-move: μ -> μ + minus_delta
        sig_mu_m2_pre = apply_shifts(sig, comps_mu, -minus_delta)

        write_conn(sig_mu_p2_pre, compute_delta_xV(sig_mu_p2_pre))
        write_conn(sig_mu_m2_pre, compute_delta_xV(sig_mu_m2_pre))

        #
        #   H_C diagonal (ROW state)

        diag_row = (
            -2.0 * delta_xV_row
            - 4.0 * (self.immirzi**2) * (self.delta**2) * (1.0 - gamma2) * delta_xV_row
            - 4.0 * (self.immirzi**2) * (self.delta**2) * gamma_prime * delta_phiV_row
        )

        write_conn(sig, diag_row)

        #
        #   H_R off-diagonals (BRA: preimages + evaluate ΔφV on them)

        comps_kp = jnp.asarray([ikp], jnp.int32)
        comps_musR = jnp.asarray([imc, imp], jnp.int32)

        right_terms = (
            (+2 * (self.delta / 2), +self._half_delta, +self._half_delta, +1),
            (+2 * (self.delta / 2), +self._half_delta, -self._half_delta, -1),
            (+2 * (self.delta / 2), -self._half_delta, +self._half_delta, +1),
            (+2 * (self.delta / 2), -self._half_delta, -self._half_delta, -1),
            (-2 * (self.delta / 2), +self._half_delta, +self._half_delta, -1),
            (-2 * (self.delta / 2), +self._half_delta, -self._half_delta, +1),
            (-2 * (self.delta / 2), -self._half_delta, +self._half_delta, -1),
            (-2 * (self.delta / 2), -self._half_delta, -self._half_delta, +1),
        )

        for dk, dmu_c, dmu_p, sgn in right_terms:
            del_k = jnp.full((BN, 1), dk, jnp.int32)
            del_mus = jnp.stack(
                (
                    jnp.full((BN,), dmu_c, jnp.int32),
                    jnp.full((BN,), dmu_p, jnp.int32),
                ),
                axis=1,
            )

            # preimage = inverse shifts
            sig_tmp = apply_shifts(sig, comps_kp, -del_k)
            sig_tmp = apply_shifts(sig_tmp, comps_musR, -del_mus)

            base_R_pre = compute_delta_phiV(sig_tmp)
            write_conn(sig_tmp, sgn * base_R_pre)

        #
        #   H_L off-diagonals (BRA: preimages + evaluate ΔφV on them)

        comps_km = jnp.asarray([ikm], jnp.int32)
        comps_musL = jnp.asarray([imm, imc], jnp.int32)

        left_terms = right_terms

        for dk, dmu_m, dmu_c, sgn in left_terms:
            del_k = jnp.full((BN, 1), dk, jnp.int32)
            del_mus = jnp.stack(
                (
                    jnp.full((BN,), dmu_m, jnp.int32),
                    jnp.full((BN,), dmu_c, jnp.int32),
                ),
                axis=1,
            )

            sig_tmp = apply_shifts(sig, comps_km, -del_k)
            sig_tmp = apply_shifts(sig_tmp, comps_musL, -del_mus)

            base_L_pre = compute_delta_phiV(sig_tmp)
            write_conn(sig_tmp, sgn * base_L_pre)

        if single:
            return σp[0], mels[0]
        return σp, mels


class SphericalVertexConstraintBojowaldSwiderskiFast(ComputationalOperator):
    """

    This version is an optimised and jax.jit-friendly version of the
    SphericalVertexConstraintBojowaldSwiderski implementation above.

    The implementation is not self-documented, as it aims to mirror the well-documented
    implementation in the SphericalVertexConstraintBojowaldSwiderski class.
    """

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
    ) -> None:

        super().__init__(H.hilbert)

        # enforce U(1), one scalar per edge
        if int(H.core.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )

        # HalfLadderGraph type check
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires the graph to be an instance of "
                f"`{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}` instead."
            )

        self._H = H
        self.D = int(H.size)
        self.cutoff = int(H.cutoff)

        states = jnp.asarray(H.tiny_hilbert.local_states)
        self._state_min = int(states[0])
        self._state_max = int(states[-1])
        self._mod_span = int(len(states))

        if type(delta) != int:
            raise ValueError(
                f"Currently, only integer (type `int`) values for `delta` is allowed, "
                f"got `{type(delta).__name__}` instead."
            )

        self.delta = int(delta)
        self._half_delta = int(self.delta // 2)

        self._use_gamma = bool(include_gamma_terms)
        self._outer_kp_constant = int(outer_kp_constant)
        self._outer_km_constant = int(outer_km_constant)
        self.immirzi = float(immirzi)

        # validate an inner k-vertex
        verts_k = list(H.graph.vertices_k)
        inner = verts_k[2:-2]
        if vertex not in inner:
            raise ValueError(
                f"vertex={vertex} must be an inner k-vertex. Valid vertices are: {inner}"
            )

        self.v = int(vertex)

        # incident edges at v
        [mu_here], k_two = H.graph.get_edges_at_k_vertex(self.v)
        left_candidates = [e for e in k_two if e[1] == self.v]
        right_candidates = [e for e in k_two if e[0] == self.v]
        if len(left_candidates) != 1 or len(right_candidates) != 1:
            raise RuntimeError(
                f"Could not disambiguate left/right k-edges at v={self.v}: {k_two}"
            )

        #
        #
        #   k_-(v): (v-1, v, key)
        e_minus = left_candidates[0]

        #
        #
        #   k_+(v): (v, v+1, key)
        e_plus = right_candidates[0]
        v_minus = e_minus[0]
        v_plus = e_plus[1]

        #
        #
        #   neighbour μ-edges μ(v-1), μ(v+1)
        [mu_left] = H.graph.get_edges_at_k_vertex(v_minus)[0]
        [mu_right] = H.graph.get_edges_at_k_vertex(v_plus)[0]

        #
        #
        # neighbor outer k-edges via get_kminus_kplus (to match the implementation above)

        #
        #
        #   k_--(v)
        has_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "left")
        if not has_left_k_vertex:
            self._has_kmm = False
            self._idx_kmm = jnp.int32(-1)
        else:
            k_edges_at_v_minus = H.graph.get_kminus_kplus(self.v - 1)
            self._has_kmm = len(k_edges_at_v_minus[0]) == 1
            self._idx_kmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus[0][0]))
                if self._has_kmm
                else jnp.int32(-1)
            )

        #
        #
        #   k_++(v)
        has_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "right")
        if not has_right_k_vertex:
            self._has_kpp = False
            self._idx_kpp = jnp.int32(-1)
        else:
            k_edges_at_v_plus = H.graph.get_kminus_kplus(self.v + 1)
            self._has_kpp = len(k_edges_at_v_plus[1]) == 1
            self._idx_kpp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus[1][0]))
                if self._has_kpp
                else jnp.int32(-1)
            )

        #
        #
        #   k_+++(v) and μ(v+2)
        has_right_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v + 1, "right")
        if not has_right_right_k_vertex:
            self._has_kppp = False
            self._idx_kppp = jnp.int32(-1)
            self._has_mu_pp = False
            self._idx_mu_pp = jnp.int32(-1)
        else:
            k_edges_at_v_plus_plus = H.graph.get_kminus_kplus(self.v + 2)
            self._has_kppp = len(k_edges_at_v_plus_plus[1]) == 1
            self._idx_kppp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus_plus[1][0]))
                if self._has_kppp
                else jnp.int32(-1)
            )
            self._has_mu_pp = True
            [mupp], _ = H.graph.get_edges_at_k_vertex(self.v + 2)
            self._idx_mu_pp = jnp.int32(H.graph.edge_to_index(mupp))

        #
        #
        #   k_---(v) and μ(v-2)
        has_left_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v - 1, "left")
        if not has_left_left_k_vertex:
            self._has_kmmm = False
            self._idx_kmmm = jnp.int32(-1)
            self._has_mu_mm = False
            self._idx_mu_mm = jnp.int32(-1)
        else:
            k_edges_at_v_minus_minus = H.graph.get_kminus_kplus(self.v - 2)
            self._has_kmmm = len(k_edges_at_v_minus_minus[0]) == 1
            self._idx_kmmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus_minus[0][0]))
                if self._has_kmmm
                else jnp.int32(-1)
            )
            self._has_mu_mm = True
            [mumm], _ = H.graph.get_edges_at_k_vertex(self.v - 2)
            self._idx_mu_mm = jnp.int32(H.graph.edge_to_index(mumm))

        # flatten to σ indices
        self._idx_mu = jnp.int32(H.graph.edge_to_index(mu_here))
        self._idx_mu_m = jnp.int32(H.graph.edge_to_index(mu_left))
        self._idx_mu_p = jnp.int32(H.graph.edge_to_index(mu_right))
        self._idx_km = jnp.int32(H.graph.edge_to_index(e_minus))
        self._idx_kp = jnp.int32(H.graph.edge_to_index(e_plus))

        # fixed connectivity
        self._nconn = 19

        # dtypes
        self._idtype = jnp.int32
        self._fdtype = jnp.float64

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return self._fdtype

    @staticmethod
    def _wrap_mod(
        vals: jnp.ndarray, delta: jnp.ndarray, mod_span: int, state_min: int
    ) -> jnp.ndarray:
        return ((vals + delta - state_min) % mod_span) + state_min

    @staticmethod
    def _sgn_delta_over_2(mu: jnp.ndarray, half_delta: int) -> jnp.ndarray:
        one = float(half_delta)
        return jnp.where(
            mu >= one, 1.0, jnp.where(mu <= -one, -1.0, mu.astype(jnp.float64) / one)
        )

    def _gamma_phi_full(
        self,
        K_left: jnp.ndarray,
        K_center: jnp.ndarray,
        K_right: jnp.ndarray,
        mu_center: jnp.ndarray,
        mu_minus: jnp.ndarray,
        mu_plus: jnp.ndarray,
    ) -> jnp.ndarray:

        _dtype = self._fdtype
        Ex_m = 0.5 * K_left.astype(_dtype)
        Ex_c = 0.5 * K_center.astype(_dtype)
        Ex_p = 0.5 * K_right.astype(_dtype)

        Ex_prime_plus = Ex_p - Ex_c
        Ex_prime_minus = Ex_c - Ex_m

        mu_c = mu_center.astype(_dtype)
        mu_p = mu_plus.astype(_dtype)
        mu_m = mu_minus.astype(_dtype)

        int_plus = 0.5 * (mu_c + mu_p)
        int_minus = 0.5 * (mu_c + mu_m)

        # same floor as slow code (see the implementation above for the safe division option)
        int_plus = jnp.maximum(int_plus, 1e-12)
        int_minus = jnp.maximum(int_minus, 1e-12)

        return 0.25 * ((-Ex_prime_plus / int_plus) + (Ex_prime_minus / int_minus))

    def _apply_shifts_multi(
        self, base: jnp.ndarray, comps: jnp.ndarray, deltas: jnp.ndarray
    ) -> jnp.ndarray:

        C = comps.shape[0]

        def body(c, cur):
            idx_c = comps[c]
            sel = jnp.take(cur, idx_c, axis=-1)
            upd = self._wrap_mod(sel, deltas[..., c], self._mod_span, self._state_min)
            return cur.at[..., idx_c].set(upd.astype(cur.dtype))

        return lax.fori_loop(0, C, body, base)

    @partial(jax.jit, static_argnums=0)
    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]

        B, N, D = σ.shape
        BN = B * N

        # outputs
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D)).copy()
        mels = jnp.zeros((B, N, self._nconn), dtype=self._fdtype)

        sig = σ.reshape(BN, D)

        # indices
        imc = int(self._idx_mu)
        imm = int(self._idx_mu_m)
        imp = int(self._idx_mu_p)
        ikm = int(self._idx_km)
        ikp = int(self._idx_kp)

        # possibly further μ indices
        impp = int(self._idx_mu_pp)
        immm = int(self._idx_mu_mm)

        # read current values
        mu_c = sig[:, imc]
        mu_m = sig[:, imm]
        mu_p = sig[:, imp]
        k_m = sig[:, ikm]
        k_p = sig[:, ikp]

        if self._has_mu_mm:
            mu_mm = sig[:, immm]
        else:
            mu_mm = jnp.asarray(0, dtype=self._idtype)

        if self._has_mu_pp:
            mu_pp = sig[:, impp]
        else:
            mu_pp = jnp.asarray(0, dtype=self._idtype)

        # K(v) and derived bits
        K = (k_m + k_p).astype(self._idtype)
        sqrtK = jnp.sqrt(jnp.abs(K).astype(self._fdtype))
        Kp1 = (K + 1).astype(self._idtype)
        Km1 = (K - 1).astype(self._idtype)
        sqrtKp1 = jnp.sqrt(jnp.abs(Kp1).astype(self._fdtype))
        sqrtKm1 = jnp.sqrt(jnp.abs(Km1).astype(self._fdtype))

        # ΔxV(v) = |μ(v)| [√|K+1| − √|K−1|]
        abs_mu = jnp.abs(mu_c).astype(self._fdtype)
        delta_xV = abs_mu * (sqrtKp1 - sqrtKm1)

        # ΔφV(v) = sgn_{δ/2}(μ(v)) √|K(v)|
        delta_phiV = self._sgn_delta_over_2(mu_c, self._half_delta) * sqrtK

        if self._use_gamma:
            #
            #
            #   K(v-1) = k_--(v) + k_-(v)
            if self._has_kmm:
                K_left = (sig[:, int(self._idx_kmm)] + sig[:, ikm]).astype(self._idtype)
            else:
                K_left = (sig[:, ikm] + jnp.int32(self._outer_km_constant)).astype(
                    self._idtype
                )

            #
            #
            #   K(v) = k_-(v) + k_+(v)
            K_center = (sig[:, ikm] + sig[:, ikp]).astype(self._idtype)

            #
            #
            #   K(v+1) = k_+(v) + k_++(v)
            if self._has_kpp:
                K_right = (sig[:, ikp] + sig[:, int(self._idx_kpp)]).astype(
                    self._idtype
                )
            else:
                K_right = (sig[:, ikp] + jnp.int32(self._outer_kp_constant)).astype(
                    self._idtype
                )

            #
            #
            #   Γ_φ(v)
            gamma = self._gamma_phi_full(
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
            #   For Γ′_φ(v) ≈ Γ_φ(v+1) − Γ_φ(v)
            #   K(v+2) = k_++(v) + k_+++(v) with boundary fallbacks to outer_kp_constant
            if self._has_kpp:
                kp_contrib = sig[:, int(self._idx_kpp)]
            else:
                kp_contrib = jnp.int32(self._outer_kp_constant)

            if self._has_kppp:
                kpp_contrib = sig[:, int(self._idx_kppp)]
            else:
                kpp_contrib = jnp.int32(self._outer_kp_constant)

            K_right_right = (kp_contrib + kpp_contrib).astype(self._idtype)

            gamma_v_plus_one = self._gamma_phi_full(
                K_left=K_center,
                K_center=K_right,
                K_right=K_right_right,
                mu_center=mu_p,
                mu_minus=mu_c,
                mu_plus=mu_pp,
            )
            gamma_prime = gamma_v_plus_one - gamma

        else:
            z = jnp.zeros_like(delta_xV)
            gamma = gamma2 = gamma_prime = z

        #
        #
        #   H_C: two off-diagonals on μ(v) and one diagonal
        comps_mu = jnp.asarray([imc], self._idtype)
        deltas_mu = jnp.stack(
            [
                jnp.full((BN, 1), 2 * self.delta, self._idtype),
                jnp.full((BN, 1), -2 * self.delta, self._idtype),
            ],
            axis=1,
        )
        base_mu = jnp.broadcast_to(sig[:, None, :], (BN, 2, D))
        sig_mu = self._apply_shifts_multi(base_mu, comps_mu, deltas_mu)

        σp = σp.at[:, :, 0:2, :].set(sig_mu.reshape(B, N, 2, D))
        mels = mels.at[:, :, 0:2].set(
            jnp.stack([delta_xV, delta_xV], axis=1).reshape(B, N, 2)
        )

        diag = (
            -2.0 * delta_xV
            - 4.0 * (self.immirzi**2) * (self.delta**2) * (1.0 - gamma2) * delta_xV
            - 4.0 * (self.immirzi**2) * (self.delta**2) * gamma_prime * delta_phiV
        ).astype(self._fdtype)

        σp = σp.at[:, :, 2, :].set(σ)
        mels = mels.at[:, :, 2].set(diag.reshape(B, N))

        #
        #
        #   H_R: 8 fixed patterns on (k_+, μ_c, μ_p)
        comps_kp = jnp.asarray([ikp], self._idtype)
        comps_musR = jnp.asarray([imc, imp], self._idtype)

        dk_R = jnp.asarray(
            [
                +self.delta,
                +self.delta,
                +self.delta,
                +self.delta,
                -self.delta,
                -self.delta,
                -self.delta,
                -self.delta,
            ],
            self._idtype,
        )
        dmu_c_R = jnp.asarray(
            [
                +self._half_delta,
                +self._half_delta,
                -self._half_delta,
                -self._half_delta,
                +self._half_delta,
                +self._half_delta,
                -self._half_delta,
                -self._half_delta,
            ],
            self._idtype,
        )
        dmu_p_R = jnp.asarray(
            [
                +self._half_delta,
                -self._half_delta,
                +self._half_delta,
                -self._half_delta,
                +self._half_delta,
                -self._half_delta,
                +self._half_delta,
                -self._half_delta,
            ],
            self._idtype,
        )
        sgn_R = jnp.asarray([+1, -1, +1, -1, -1, +1, -1, +1], self._fdtype)

        deltas_k_R = jnp.broadcast_to(dk_R[None, :, None], (BN, 8, 1))
        deltas_mus_R = jnp.stack(
            [
                jnp.broadcast_to(dmu_c_R[None, :], (BN, 8)),
                jnp.broadcast_to(dmu_p_R[None, :], (BN, 8)),
            ],
            axis=2,
        )

        base_R_k = jnp.broadcast_to(sig[:, None, :], (BN, 8, D))
        sig_R_k = self._apply_shifts_multi(base_R_k, comps_kp, deltas_k_R)
        sig_R = self._apply_shifts_multi(sig_R_k, comps_musR, deltas_mus_R)

        σp = σp.at[:, :, 3:11, :].set(sig_R.reshape(B, N, 8, D))
        mR = delta_phiV.reshape(B, N, 1) * sgn_R.reshape(1, 1, 8)
        mels = mels.at[:, :, 3:11].set(mR)

        #
        #
        #   H_L: mirror 8 patterns on (k_-, μ_m, μ_c)
        comps_km = jnp.asarray([ikm], self._idtype)
        comps_musL = jnp.asarray([imm, imc], self._idtype)

        dk_L = dk_R
        dmu_m_L = dmu_c_R
        dmu_c_L = dmu_p_R
        sgn_L = sgn_R

        deltas_k_L = jnp.broadcast_to(dk_L[None, :, None], (BN, 8, 1))
        deltas_mus_L = jnp.stack(
            [
                jnp.broadcast_to(dmu_m_L[None, :], (BN, 8)),
                jnp.broadcast_to(dmu_c_L[None, :], (BN, 8)),
            ],
            axis=2,
        )

        base_L_k = jnp.broadcast_to(sig[:, None, :], (BN, 8, D))
        sig_L_k = self._apply_shifts_multi(base_L_k, comps_km, deltas_k_L)
        sig_L = self._apply_shifts_multi(sig_L_k, comps_musL, deltas_mus_L)

        σp = σp.at[:, :, 11:19, :].set(sig_L.reshape(B, N, 8, D))
        mL = delta_phiV.reshape(B, N, 1) * sgn_L.reshape(1, 1, 8)
        mels = mels.at[:, :, 11:19].set(mL)

        if single:
            return σp[0], mels[0]
        return σp, mels


class SphericalVertexConstraintBojowaldSwiderskiFastFixed(ComputationalOperator):
    """

    This version is an optimised and jax.jit-friendly version of the
    SphericalVertexConstraintBojowaldSwiderski implementation above.

    The implementation is not self-documented, as it aims to mirror the well-documented
    implementation in the SphericalVertexConstraintBojowaldSwiderski class.
    """

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
    ) -> None:

        super().__init__(H.hilbert)

        # enforce U(1), one scalar per edge
        if int(H.core.gauge_dimensions) != 1:
            raise ValueError(
                f"{type(self).__name__} requires U(1) `gauge_dimensions = 1`."
            )

        # HalfLadderGraph type check
        if not isinstance(H.graph, HalfLadderGraph):
            raise ValueError(
                f"This constraint requires the graph to be an instance of "
                f"`{HalfLadderGraph.__name__}`, got `{type(H.graph).__name__}` instead."
            )

        self._H = H
        self.D = int(H.size)
        self.cutoff = int(H.cutoff)

        states = jnp.asarray(H.tiny_hilbert.local_states)
        self._state_min = int(states[0])
        self._state_max = int(states[-1])
        self._mod_span = int(len(states))

        if type(delta) != int:
            raise ValueError(
                f"Currently, only integer (type `int`) values for `delta` is allowed, "
                f"got `{type(delta).__name__}` instead."
            )

        self.delta = int(delta)
        self._half_delta = int(self.delta // 2)

        self._use_gamma = bool(include_gamma_terms)
        self._outer_kp_constant = int(outer_kp_constant)
        self._outer_km_constant = int(outer_km_constant)
        self.immirzi = float(immirzi)

        # validate an inner k-vertex
        verts_k = list(H.graph.vertices_k)
        inner = verts_k[2:-2]
        if vertex not in inner:
            raise ValueError(
                f"vertex={vertex} must be an inner k-vertex. Valid vertices are: {inner}"
            )

        self.v = int(vertex)

        # incident edges at v
        [mu_here], k_two = H.graph.get_edges_at_k_vertex(self.v)
        left_candidates = [e for e in k_two if e[1] == self.v]
        right_candidates = [e for e in k_two if e[0] == self.v]
        if len(left_candidates) != 1 or len(right_candidates) != 1:
            raise RuntimeError(
                f"Could not disambiguate left/right k-edges at v={self.v}: {k_two}"
            )

        #
        #
        #   k_-(v): (v-1, v, key)
        e_minus = left_candidates[0]

        #
        #
        #   k_+(v): (v, v+1, key)
        e_plus = right_candidates[0]
        v_minus = e_minus[0]
        v_plus = e_plus[1]

        #
        #
        #   neighbour μ-edges μ(v-1), μ(v+1)
        [mu_left] = H.graph.get_edges_at_k_vertex(v_minus)[0]
        [mu_right] = H.graph.get_edges_at_k_vertex(v_plus)[0]

        #
        #
        # neighbor outer k-edges via get_kminus_kplus (to match the implementation above)

        #
        #
        #   k_--(v)
        has_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "left")
        if not has_left_k_vertex:
            self._has_kmm = False
            self._idx_kmm = jnp.int32(-1)
        else:
            k_edges_at_v_minus = H.graph.get_kminus_kplus(self.v - 1)
            self._has_kmm = len(k_edges_at_v_minus[0]) == 1
            self._idx_kmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus[0][0]))
                if self._has_kmm
                else jnp.int32(-1)
            )

        #
        #
        #   k_++(v)
        has_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v, "right")
        if not has_right_k_vertex:
            self._has_kpp = False
            self._idx_kpp = jnp.int32(-1)
        else:
            k_edges_at_v_plus = H.graph.get_kminus_kplus(self.v + 1)
            self._has_kpp = len(k_edges_at_v_plus[1]) == 1
            self._idx_kpp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus[1][0]))
                if self._has_kpp
                else jnp.int32(-1)
            )

        #
        #
        #   k_+++(v) and μ(v+2)
        has_right_right_k_vertex = H.graph.has_k_vertex_neighbours(self.v + 1, "right")
        if not has_right_right_k_vertex:
            self._has_kppp = False
            self._idx_kppp = jnp.int32(-1)
            self._has_mu_pp = False
            self._idx_mu_pp = jnp.int32(-1)
        else:
            k_edges_at_v_plus_plus = H.graph.get_kminus_kplus(self.v + 2)
            self._has_kppp = len(k_edges_at_v_plus_plus[1]) == 1
            self._idx_kppp = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_plus_plus[1][0]))
                if self._has_kppp
                else jnp.int32(-1)
            )
            self._has_mu_pp = True
            [mupp], _ = H.graph.get_edges_at_k_vertex(self.v + 2)
            self._idx_mu_pp = jnp.int32(H.graph.edge_to_index(mupp))

        #
        #
        #   k_---(v) and μ(v-2)
        has_left_left_k_vertex = H.graph.has_k_vertex_neighbours(self.v - 1, "left")
        if not has_left_left_k_vertex:
            self._has_kmmm = False
            self._idx_kmmm = jnp.int32(-1)
            self._has_mu_mm = False
            self._idx_mu_mm = jnp.int32(-1)
        else:
            k_edges_at_v_minus_minus = H.graph.get_kminus_kplus(self.v - 2)
            self._has_kmmm = len(k_edges_at_v_minus_minus[0]) == 1
            self._idx_kmmm = (
                jnp.int32(H.graph.edge_to_index(k_edges_at_v_minus_minus[0][0]))
                if self._has_kmmm
                else jnp.int32(-1)
            )
            self._has_mu_mm = True
            [mumm], _ = H.graph.get_edges_at_k_vertex(self.v - 2)
            self._idx_mu_mm = jnp.int32(H.graph.edge_to_index(mumm))

        # flatten to σ indices
        self._idx_mu = jnp.int32(H.graph.edge_to_index(mu_here))
        self._idx_mu_m = jnp.int32(H.graph.edge_to_index(mu_left))
        self._idx_mu_p = jnp.int32(H.graph.edge_to_index(mu_right))
        self._idx_km = jnp.int32(H.graph.edge_to_index(e_minus))
        self._idx_kp = jnp.int32(H.graph.edge_to_index(e_plus))

        # fixed connectivity
        self._nconn = 19

        # dtypes
        self._idtype = jnp.int32
        self._fdtype = jnp.float64

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return self._fdtype

    @staticmethod
    def _wrap_mod(
        vals: jnp.ndarray, delta: jnp.ndarray, mod_span: int, state_min: int
    ) -> jnp.ndarray:
        return ((vals + delta - state_min) % mod_span) + state_min

    @staticmethod
    def _sgn_delta_over_2(mu: jnp.ndarray, half_delta: int) -> jnp.ndarray:
        one = float(half_delta)
        return jnp.where(
            mu >= one, 1.0, jnp.where(mu <= -one, -1.0, mu.astype(jnp.float64) / one)
        )

    def _gamma_phi_full(
        self,
        K_left: jnp.ndarray,
        K_center: jnp.ndarray,
        K_right: jnp.ndarray,
        mu_center: jnp.ndarray,
        mu_minus: jnp.ndarray,
        mu_plus: jnp.ndarray,
    ) -> jnp.ndarray:

        _dtype = self._fdtype
        Ex_m = 0.5 * K_left.astype(_dtype)
        Ex_c = 0.5 * K_center.astype(_dtype)
        Ex_p = 0.5 * K_right.astype(_dtype)

        Ex_prime_plus = Ex_p - Ex_c
        Ex_prime_minus = Ex_c - Ex_m

        mu_c = mu_center.astype(_dtype)
        mu_p = mu_plus.astype(_dtype)
        mu_m = mu_minus.astype(_dtype)

        int_plus = 0.5 * (mu_c + mu_p)
        int_minus = 0.5 * (mu_c + mu_m)

        # same floor as slow code (see the implementation above for the safe division option)
        int_plus = jnp.maximum(int_plus, 1e-12)
        int_minus = jnp.maximum(int_minus, 1e-12)

        return 0.25 * ((-Ex_prime_plus / int_plus) + (Ex_prime_minus / int_minus))

    def _apply_shifts_multi(
        self, base: jnp.ndarray, comps: jnp.ndarray, deltas: jnp.ndarray
    ) -> jnp.ndarray:

        C = comps.shape[0]

        def body(c, cur):
            idx_c = comps[c]
            sel = jnp.take(cur, idx_c, axis=-1)
            upd = self._wrap_mod(sel, deltas[..., c], self._mod_span, self._state_min)
            return cur.at[..., idx_c].set(upd.astype(cur.dtype))

        return lax.fori_loop(0, C, body, base)

    @partial(jax.jit, static_argnums=0)
    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        single = σ.ndim == 2
        if single:
            σ = σ[None, :, :]

        B, N, D = σ.shape
        BN = B * N

        # outputs
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, self._nconn, D)).copy()
        mels = jnp.zeros((B, N, self._nconn), dtype=self._fdtype)

        sig = σ.reshape(BN, D)

        # indices
        imc = int(self._idx_mu)
        imm = int(self._idx_mu_m)
        imp = int(self._idx_mu_p)
        ikm = int(self._idx_km)
        ikp = int(self._idx_kp)

        impp = int(self._idx_mu_pp)
        immm = int(self._idx_mu_mm)

        # read row values
        mu_c = sig[:, imc]
        mu_m = sig[:, imm]
        mu_p = sig[:, imp]
        k_m = sig[:, ikm]
        k_p = sig[:, ikp]

        if self._has_mu_mm:
            mu_mm = sig[:, immm]
        else:
            mu_mm = jnp.asarray(0, dtype=self._idtype)

        if self._has_mu_pp:
            mu_pp = sig[:, impp]
        else:
            mu_pp = jnp.asarray(0, dtype=self._idtype)

        #
        # ROW-derived pieces for the DIAGONAL

        K = (k_m + k_p).astype(self._idtype)
        sqrtK = jnp.sqrt(jnp.abs(K).astype(self._fdtype))
        Kp1 = (K + 1).astype(self._idtype)
        Km1 = (K - 1).astype(self._idtype)
        sqrtKp1 = jnp.sqrt(jnp.abs(Kp1).astype(self._fdtype))
        sqrtKm1 = jnp.sqrt(jnp.abs(Km1).astype(self._fdtype))

        abs_mu = jnp.abs(mu_c).astype(self._fdtype)
        delta_xV_row = abs_mu * (sqrtKp1 - sqrtKm1)

        delta_phiV_row = self._sgn_delta_over_2(mu_c, self._half_delta) * sqrtK

        if self._use_gamma:
            if self._has_kmm:
                K_left = (sig[:, int(self._idx_kmm)] + sig[:, ikm]).astype(self._idtype)
            else:
                K_left = (sig[:, ikm] + jnp.int32(self._outer_km_constant)).astype(
                    self._idtype
                )

            K_center = (sig[:, ikm] + sig[:, ikp]).astype(self._idtype)

            if self._has_kpp:
                K_right = (sig[:, ikp] + sig[:, int(self._idx_kpp)]).astype(
                    self._idtype
                )
            else:
                K_right = (sig[:, ikp] + jnp.int32(self._outer_kp_constant)).astype(
                    self._idtype
                )

            gamma = self._gamma_phi_full(
                K_left=K_left,
                K_center=K_center,
                K_right=K_right,
                mu_center=mu_c,
                mu_minus=mu_m,
                mu_plus=mu_p,
            )
            gamma2 = gamma * gamma

            if self._has_kpp:
                kp_contrib = sig[:, int(self._idx_kpp)]
            else:
                kp_contrib = jnp.int32(self._outer_kp_constant)

            if self._has_kppp:
                kpp_contrib = sig[:, int(self._idx_kppp)]
            else:
                kpp_contrib = jnp.int32(self._outer_kp_constant)

            K_right_right = (kp_contrib + kpp_contrib).astype(self._idtype)

            gamma_v_plus_one = self._gamma_phi_full(
                K_left=K_center,
                K_center=K_right,
                K_right=K_right_right,
                mu_center=mu_p,
                mu_minus=mu_c,
                mu_plus=mu_pp,
            )
            gamma_prime = gamma_v_plus_one - gamma
        else:
            z = jnp.zeros_like(delta_xV_row)
            gamma = gamma2 = gamma_prime = z

        #
        #   H_C off-diagonals on μ(v)
        #   BRA: build PREIMAGES by using negative deltas

        comps_mu = jnp.asarray([imc], self._idtype)
        deltas_mu = jnp.stack(
            [
                jnp.full((BN, 1), 2 * self.delta, self._idtype),
                jnp.full((BN, 1), -2 * self.delta, self._idtype),
            ],
            axis=1,
        )
        deltas_mu_bra = -deltas_mu

        base_mu = jnp.broadcast_to(sig[:, None, :], (BN, 2, D))
        sig_mu_pre = self._apply_shifts_multi(base_mu, comps_mu, deltas_mu_bra)

        # compute ΔxV on each preimage branch
        mu_c_mu = sig_mu_pre[:, :, imc]
        k_m_mu = sig_mu_pre[:, :, ikm]
        k_p_mu = sig_mu_pre[:, :, ikp]

        K_mu = (k_m_mu + k_p_mu).astype(self._idtype)
        sqrtKp1_mu = jnp.sqrt(jnp.abs((K_mu + 1).astype(self._fdtype)))
        sqrtKm1_mu = jnp.sqrt(jnp.abs((K_mu - 1).astype(self._fdtype)))
        abs_mu_mu = jnp.abs(mu_c_mu).astype(self._fdtype)
        delta_xV_mu = abs_mu_mu * (sqrtKp1_mu - sqrtKm1_mu)

        σp = σp.at[:, :, 0:2, :].set(sig_mu_pre.reshape(B, N, 2, D))
        mels = mels.at[:, :, 0:2].set(delta_xV_mu.reshape(B, N, 2))

        #
        #   H_C diagonal (row)

        diag = (
            -2.0 * delta_xV_row
            - 4.0 * (self.immirzi**2) * (self.delta**2) * (1.0 - gamma2) * delta_xV_row
            - 4.0 * (self.immirzi**2) * (self.delta**2) * gamma_prime * delta_phiV_row
        ).astype(self._fdtype)

        σp = σp.at[:, :, 2, :].set(σ)
        mels = mels.at[:, :, 2].set(diag.reshape(B, N))

        #
        #   H_R: 8 patterns on (k_+, μ_c, μ_p)
        #   BRA: preimages via inverse deltas and ΔφV evaluated on preimages

        comps_kp = jnp.asarray([ikp], self._idtype)
        comps_musR = jnp.asarray([imc, imp], self._idtype)

        dk_R = jnp.asarray(
            [
                +self.delta,
                +self.delta,
                +self.delta,
                +self.delta,
                -self.delta,
                -self.delta,
                -self.delta,
                -self.delta,
            ],
            self._idtype,
        )
        dmu_c_R = jnp.asarray(
            [
                +self._half_delta,
                +self._half_delta,
                -self._half_delta,
                -self._half_delta,
                +self._half_delta,
                +self._half_delta,
                -self._half_delta,
                -self._half_delta,
            ],
            self._idtype,
        )
        dmu_p_R = jnp.asarray(
            [
                +self._half_delta,
                -self._half_delta,
                +self._half_delta,
                -self._half_delta,
                +self._half_delta,
                -self._half_delta,
                +self._half_delta,
                -self._half_delta,
            ],
            self._idtype,
        )
        sgn_R = jnp.asarray([+1, -1, +1, -1, -1, +1, -1, +1], self._fdtype)

        deltas_k_R = jnp.broadcast_to(dk_R[None, :, None], (BN, 8, 1))
        deltas_mus_R = jnp.stack(
            [
                jnp.broadcast_to(dmu_c_R[None, :], (BN, 8)),
                jnp.broadcast_to(dmu_p_R[None, :], (BN, 8)),
            ],
            axis=2,
        )

        deltas_k_R_bra = -deltas_k_R
        deltas_mus_R_bra = -deltas_mus_R

        base_R_k = jnp.broadcast_to(sig[:, None, :], (BN, 8, D))
        sig_R_k_pre = self._apply_shifts_multi(base_R_k, comps_kp, deltas_k_R_bra)
        sig_R_pre = self._apply_shifts_multi(sig_R_k_pre, comps_musR, deltas_mus_R_bra)

        mu_c_R_pre = sig_R_pre[:, :, imc]
        k_m_R_pre = sig_R_pre[:, :, ikm]
        k_p_R_pre = sig_R_pre[:, :, ikp]

        K_R_pre = (k_m_R_pre + k_p_R_pre).astype(self._idtype)
        sqrtK_R_pre = jnp.sqrt(jnp.abs(K_R_pre).astype(self._fdtype))
        delta_phiV_R_pre = (
            self._sgn_delta_over_2(mu_c_R_pre, self._half_delta) * sqrtK_R_pre
        )

        σp = σp.at[:, :, 3:11, :].set(sig_R_pre.reshape(B, N, 8, D))
        mR = delta_phiV_R_pre.reshape(B, N, 8) * sgn_R.reshape(1, 1, 8)
        mels = mels.at[:, :, 3:11].set(mR)

        #
        #   H_L: mirror 8 patterns on (k_-, μ_m, μ_c)
        #   BRA: preimages + ΔφV on preimages

        comps_km = jnp.asarray([ikm], self._idtype)
        comps_musL = jnp.asarray([imm, imc], self._idtype)

        dk_L = dk_R
        dmu_m_L = dmu_c_R
        dmu_c_L = dmu_p_R
        sgn_L = sgn_R

        deltas_k_L = jnp.broadcast_to(dk_L[None, :, None], (BN, 8, 1))
        deltas_mus_L = jnp.stack(
            [
                jnp.broadcast_to(dmu_m_L[None, :], (BN, 8)),
                jnp.broadcast_to(dmu_c_L[None, :], (BN, 8)),
            ],
            axis=2,
        )

        deltas_k_L_bra = -deltas_k_L
        deltas_mus_L_bra = -deltas_mus_L

        base_L_k = jnp.broadcast_to(sig[:, None, :], (BN, 8, D))
        sig_L_k_pre = self._apply_shifts_multi(base_L_k, comps_km, deltas_k_L_bra)
        sig_L_pre = self._apply_shifts_multi(sig_L_k_pre, comps_musL, deltas_mus_L_bra)

        mu_c_L_pre = sig_L_pre[:, :, imc]
        k_m_L_pre = sig_L_pre[:, :, ikm]
        k_p_L_pre = sig_L_pre[:, :, ikp]

        K_L_pre = (k_m_L_pre + k_p_L_pre).astype(self._idtype)
        sqrtK_L_pre = jnp.sqrt(jnp.abs(K_L_pre).astype(self._fdtype))
        delta_phiV_L_pre = (
            self._sgn_delta_over_2(mu_c_L_pre, self._half_delta) * sqrtK_L_pre
        )

        σp = σp.at[:, :, 11:19, :].set(sig_L_pre.reshape(B, N, 8, D))
        mL = delta_phiV_L_pre.reshape(B, N, 8) * sgn_L.reshape(1, 1, 8)
        mels = mels.at[:, :, 11:19].set(mL)

        if single:
            return σp[0], mels[0]
        return σp, mels
