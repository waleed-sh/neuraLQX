#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import ast

from typing import Tuple
from typing import Union
from typing import List

import jax
import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


def _emanates_from(graph, v: int, edge: tuple) -> bool:
    """
    True if the directed edge is oriented as emanating from v in the stored graph.
    Handles keyed edges (u,v,key).
    """

    # strip key for membership checks
    u, w = edge[0], edge[1]
    if v not in (u, w):
        return False

    rev = (edge[1], edge[0], edge[2])
    if (edge in graph.edges and edge[0] == v) or (rev in graph.edges and rev[0] == v):
        return True
    return False


@jax.jit
def _sqrt_volume_kernel(
    sigma: jnp.ndarray,
    comps1: jnp.ndarray,
    comps2: jnp.ndarray,
    signs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute sqrt(V_v)(sigma) for a batch sigma of shape (..., D).

    Here V_v is the 2+1 "volume" (= area) eigenvalue:
        V_v(sigma) = || sum_pairs eps(p) * (m_{e1} x m_{e2}) ||  >= 0
    and we return sqrt(V_v).
    """

    P = signs.shape[0]
    if P == 0:
        return jnp.zeros(sigma.shape[:-1], dtype=jnp.float64)

    D = sigma.shape[-1]
    sig = sigma.reshape((-1, D))

    m1 = jnp.take(sig, comps1, axis=-1)
    m2 = jnp.take(sig, comps2, axis=-1)

    cr = jnp.cross(m1, m2, axis=-1)
    vec = jnp.sum(cr * signs[None, :, None], axis=1)  # (M, 3)

    V = jnp.linalg.norm(vec, axis=-1).astype(jnp.float64)  # (M,)
    V = V.reshape(sigma.shape[:-1])

    return jnp.sqrt(V).astype(jnp.float64)


class Euclidean3dVertexTRCOperator(ComputationalOperator):
    r"""
    Single-vertex Thiemann-regularised Euclidean Hamiltonian constraint H_v
    for the 2+1 U(1)^3 weak-coupling model, implemented as a ComputationalOperator.

    For each triangulation pairing at vertex v, and for each U(1) copy c in {0,1,2},
    the operator produces a loop-shifted configuration:
        |σ'> = h_{α_{ij}}^{(c)} |σ>
    with matrix element
        <σ'|H_v|σ> = eps_term * Δ_k^{(c)}(σ) * Δ_l^{(c)}(σ),
    where
        Δ_s^{(c)}(σ) = sqrt(V_v(σ)) - sqrt(V_v(h_{s}^{(c)-1} σ)).
    The orientation of the loop holonomy is encoded by eps(ij) via an adjoint flip of the loop.

    Notes:
      - This is the "non-adjoint" (ket-action) ordering
    """

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    def __init__(
        self,
        lqx: "LqxWCL3D",
        vertex: Union[int, str],
        *,
        apply_lapse: bool = False,
    ):
        # hilbert handle
        super().__init__(lqx.hilbert.hilbert)

        self._H = lqx.hilbert
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex

        # U(1)^3
        self.gauge_dim = int(lqx.hilbert.gauge_dimensions)
        assert self.gauge_dim == 3, "This operator assumes gauge_dim == 3 (U(1)^3)."

        # representation: D = n_edges_total * gauge_dim
        self.n_edges_total = int(lqx.hilbert.size // self.gauge_dim)
        self._offsets = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        # modular wrap params for charge shifts
        self._mod_span = int(lqx.hilbert.allowed_basis_states.length)
        self._state_min = int(lqx.hilbert.allowed_basis_states.start)

        # lapse (optional)
        self._apply_lapse = bool(apply_lapse)
        self._Nv = (
            float(lqx.lapses[self.vertex])
            if (apply_lapse and hasattr(lqx, "lapses"))
            else 1.0
        )

        #
        #
        #   precompute sqrt(volume) geometry at this vertex (edge pairs + signs)

        pairs = lqx.hilbert.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = lqx.hilbert.graph.signs[str(self.vertex)]

        comps1, comps2, signs = [], [], []
        for pair in pairs:
            e1, e2 = map(lqx.hilbert.graph.edge_to_index, pair)
            c1 = (jnp.int32(e1) + self._offsets).astype(jnp.int32)
            c2 = (jnp.int32(e2) + self._offsets).astype(jnp.int32)
            comps1.append(c1)
            comps2.append(c2)
            signs.append(int(signs_dict[str(pair)]))

        if len(comps1) > 0:
            self._vol_comps1 = jnp.stack(comps1, axis=0).astype(jnp.int32)  # (P,3)
            self._vol_comps2 = jnp.stack(comps2, axis=0).astype(jnp.int32)  # (P,3)
            self._vol_signs = jnp.asarray(signs, dtype=jnp.int32)  # (P,)
        else:
            self._vol_comps1 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._vol_comps2 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._vol_signs = jnp.zeros((0,), dtype=jnp.int32)

        #
        #
        #   build per-term data for this vertex using the model bookkeeping

        # each term stores:
        #   weight[t]               float
        #   seg_k_eidx[t], seg_l_eidx[t]      int (base edge index)
        #   seg_k_inv_shift[t], seg_l_inv_shift[t]  int in {+1,-1} (this is h_s^{-1} action)
        #   loop_edges[t, m]        int with padding -1
        #   loop_shifts[t, m]       int with padding 0  (creation:+1, annihilation:-1, plus orientation flip)
        weights: List[float] = []
        seg_k: List[int] = []
        seg_l: List[int] = []
        seg_k_inv: List[int] = []
        seg_l_inv: List[int] = []
        loop_edges_list: List[jnp.ndarray] = []
        loop_shifts_list: List[jnp.ndarray] = []

        # iterate triangulations stored in lqx (created by _init_triangulation_edges)
        for triang_key, triang in lqx._sum_terms.items():
            terms_at_v = triang["Terms"].get(str(self.vertex), None)
            if terms_at_v is None:
                continue

            # triang_key is str( (min_loop1, min_loop2) )
            min_loop1 = ast.literal_eval(triang_key)[0]
            loop_idx = lqx.hilbert.graph.minimal_loops().index(min_loop1)
            dressed_loop = lqx.hilbert.graph.dressed_minimal_loops()[loop_idx]

            ij_list = terms_at_v["ij"]
            kl_list = terms_at_v["kl"]

            for ij in ij_list:
                for kl in kl_list:
                    eps_term = float(lqx._epsilon[str(ij)] * lqx._epsilon[str(kl)])

                    # loop orientation via eps(ij): eps=-1 means traverse loop backwards => flip shifts
                    loop_flip = -1 if (lqx._epsilon[str(ij)] == -1) else +1

                    # segment edges k,l are the two edges in kl (keyed edges)
                    ek = kl[0]
                    el = kl[1]

                    eidx_k = int(lqx.hilbert.graph.edge_to_index(ek))
                    eidx_l = int(lqx.hilbert.graph.edge_to_index(el))

                    # inverse shift direction:
                    # if edge emanates from v => inverse is adjoint => shift -1
                    # else inverse is non-adjoint => shift +1
                    inv_k = (
                        -1 if _emanates_from(lqx.hilbert.graph, self.vertex, ek) else +1
                    )
                    inv_l = (
                        -1 if _emanates_from(lqx.hilbert.graph, self.vertex, el) else +1
                    )

                    # loop edges and their creation/annihilation shifts, copy-resolved later by offsets
                    le = []
                    ls = []
                    for edge, data in dressed_loop:
                        eidx = int(lqx.hilbert.graph.edge_to_index(edge))
                        s = +1 if data["type"] == "creation" else -1
                        le.append(eidx)
                        ls.append(loop_flip * s)

                    loop_edges_list.append(jnp.asarray(le, dtype=jnp.int32))
                    loop_shifts_list.append(jnp.asarray(ls, dtype=jnp.int32))

                    weights.append(eps_term)
                    seg_k.append(eidx_k)
                    seg_l.append(eidx_l)
                    seg_k_inv.append(inv_k)
                    seg_l_inv.append(inv_l)

        self._n_terms = len(weights)

        # pad loops to fixed width
        Lmax = int(max((arr.size for arr in loop_edges_list), default=0))
        if Lmax == 0:
            self._loop_edges = jnp.zeros((self._n_terms, 0), dtype=jnp.int32)
            self._loop_shifts = jnp.zeros((self._n_terms, 0), dtype=jnp.int32)
        else:
            self._loop_edges = jnp.stack(
                [
                    jnp.pad(a, (0, Lmax - a.size), constant_values=-1)
                    for a in loop_edges_list
                ],
                axis=0,
            ).astype(jnp.int32)
            self._loop_shifts = jnp.stack(
                [
                    jnp.pad(a, (0, Lmax - a.size), constant_values=0)
                    for a in loop_shifts_list
                ],
                axis=0,
            ).astype(jnp.int32)

        self._weight = (
            jnp.asarray(weights, dtype=jnp.float64)
            if self._n_terms
            else jnp.zeros((0,), dtype=jnp.float64)
        )
        self._seg_k = (
            jnp.asarray(seg_k, dtype=jnp.int32)
            if self._n_terms
            else jnp.zeros((0,), dtype=jnp.int32)
        )
        self._seg_l = (
            jnp.asarray(seg_l, dtype=jnp.int32)
            if self._n_terms
            else jnp.zeros((0,), dtype=jnp.int32)
        )
        self._seg_k_inv = (
            jnp.asarray(seg_k_inv, dtype=jnp.int32)
            if self._n_terms
            else jnp.zeros((0,), dtype=jnp.int32)
        )
        self._seg_l_inv = (
            jnp.asarray(seg_l_inv, dtype=jnp.int32)
            if self._n_terms
            else jnp.zeros((0,), dtype=jnp.int32)
        )

    @property
    def max_conn_size(self) -> int:
        # 1 connection per (term, copy)
        return 3 * self._n_terms if self._n_terms else 1

    def _wrap_mod(self, vals: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        return ((vals + delta - self._state_min) % self._mod_span) + self._state_min

    def _shift_one(
        self, sigma: jnp.ndarray, comp_idx: jnp.ndarray, delta: jnp.ndarray
    ) -> jnp.ndarray:
        # sigma: (M,D), comp_idx scalar, delta scalar
        sel = sigma[:, comp_idx]
        new_sel = self._wrap_mod(sel, delta)
        return sigma.at[:, comp_idx].set(new_sel)

    def _apply_loop(
        self, sigma: jnp.ndarray, comps_row: jnp.ndarray, shifts_row: jnp.ndarray
    ) -> jnp.ndarray:
        # sigma: (M,D); comps_row: (L,), shifts_row: (L,)
        sig = sigma
        L = comps_row.shape[0]
        for m in range(L):
            idx = comps_row[m]
            sh = shifts_row[m]
            sig = jax.lax.cond(
                idx >= 0,
                lambda s: self._shift_one(s, idx, sh),
                lambda s: s,
                sig,
            )
        return sig

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # σ: (..., D) int
        σ = jnp.asarray(σ, dtype=jnp.int32)

        if self._n_terms == 0:
            return σ[..., None, :], jnp.zeros(σ.shape[:-1] + (1,), dtype=jnp.float64)

        leading = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading))) if leading else 1
        sig0 = σ.reshape((M, D))

        # base sqrtV(σ)
        s0 = _sqrt_volume_kernel(
            σ, self._vol_comps1, self._vol_comps2, self._vol_signs
        ).reshape((M,))

        n_conn = self.gauge_dim * self._n_terms
        sigp = jnp.broadcast_to(sig0[:, None, :], (M, n_conn, D)).copy()
        mels = jnp.zeros((M, n_conn), dtype=jnp.float64)

        idx = 0
        for t in range(self._n_terms):
            w = self._weight[t]

            # loop base edges/shifts (length Lmax, padded)
            loop_e = self._loop_edges[t]  # (Lmax,)
            loop_s = self._loop_shifts[t]  # (Lmax,)

            # segment base edges and inverse shifts
            ek = self._seg_k[t]
            el = self._seg_l[t]
            invk = self._seg_k_inv[t]
            invl = self._seg_l_inv[t]

            for c in range(self.gauge_dim):
                off = int(self._offsets[c])

                # Δ_k(σ) = sqrtV(σ) - sqrtV(h_{k}^{-1} σ)
                sig_k = self._shift_one(sig0, jnp.int32(ek + off), jnp.int32(invk))
                sk = _sqrt_volume_kernel(
                    sig_k.reshape(leading + (D,)),
                    self._vol_comps1,
                    self._vol_comps2,
                    self._vol_signs,
                ).reshape((M,))
                dk = s0 - sk

                # Δ_l(σ)
                sig_l = self._shift_one(sig0, jnp.int32(el + off), jnp.int32(invl))
                sl = _sqrt_volume_kernel(
                    sig_l.reshape(leading + (D,)),
                    self._vol_comps1,
                    self._vol_comps2,
                    self._vol_signs,
                ).reshape((M,))
                dl = s0 - sl

                # final state: apply loop holonomy in copy c only
                loop_comps_c = jnp.where(loop_e >= 0, loop_e + off, loop_e).astype(
                    jnp.int32
                )
                loop_shifts_c = loop_s.astype(jnp.int32)

                sig_final = self._apply_loop(sig0, loop_comps_c, loop_shifts_c)

                sigp = sigp.at[:, idx, :].set(sig_final)
                mels = mels.at[:, idx].set((self._Nv * w) * dk * dl)
                idx += 1

        # reshape back
        sigp = sigp.reshape(leading + (n_conn, D))
        mels = mels.reshape(leading + (n_conn,)).astype(jnp.float64)
        return sigp, mels


class Euclidean3dVertexTRCAdjointOperator(Euclidean3dVertexTRCOperator):
    r"""
    Adjoint of Euclidean3dVertexTRCOperator in the same computational convention.

    Differences vs non-adjoint:
      - segment difference uses h_s (forward) instead of h_s^{-1}:
          Δ_s^†(σ) = sqrtV(σ) - sqrtV(h_s σ)
      - loop holonomy is daggered: loop shifts are negated.
    """

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):
        σ = jnp.asarray(σ, dtype=jnp.int32)

        if self._n_terms == 0:
            return σ[..., None, :], jnp.zeros(σ.shape[:-1] + (1,), dtype=jnp.float64)

        leading = σ.shape[:-1]
        D = σ.shape[-1]
        M = int(jnp.prod(jnp.array(leading))) if leading else 1
        sig0 = σ.reshape((M, D))

        s0 = _sqrt_volume_kernel(
            σ, self._vol_comps1, self._vol_comps2, self._vol_signs
        ).reshape((M,))

        n_conn = self.gauge_dim * self._n_terms
        sigp = jnp.broadcast_to(sig0[:, None, :], (M, n_conn, D)).copy()
        mels = jnp.zeros((M, n_conn), dtype=jnp.float64)

        idx = 0
        for t in range(self._n_terms):
            w = self._weight[t]
            loop_e = self._loop_edges[t]

            # dagger the loop
            loop_s = -self._loop_shifts[t]

            ek = self._seg_k[t]
            el = self._seg_l[t]
            invk = self._seg_k_inv[t]
            invl = self._seg_l_inv[t]

            # forward shift is minus the inverse shift
            fwdk = -invk
            fwdl = -invl

            for c in range(self.gauge_dim):
                off = int(self._offsets[c])

                sig_k = self._shift_one(sig0, jnp.int32(ek + off), jnp.int32(fwdk))
                sk = _sqrt_volume_kernel(
                    sig_k.reshape(leading + (D,)),
                    self._vol_comps1,
                    self._vol_comps2,
                    self._vol_signs,
                ).reshape((M,))
                dk = s0 - sk

                sig_l = self._shift_one(sig0, jnp.int32(el + off), jnp.int32(fwdl))
                sl = _sqrt_volume_kernel(
                    sig_l.reshape(leading + (D,)),
                    self._vol_comps1,
                    self._vol_comps2,
                    self._vol_signs,
                ).reshape((M,))
                dl = s0 - sl

                loop_comps_c = jnp.where(loop_e >= 0, loop_e + off, loop_e).astype(
                    jnp.int32
                )
                loop_shifts_c = loop_s.astype(jnp.int32)

                sig_final = self._apply_loop(sig0, loop_comps_c, loop_shifts_c)

                sigp = sigp.at[:, idx, :].set(sig_final)
                mels = mels.at[:, idx].set((self._Nv * w) * dk * dl)
                idx += 1

        sigp = sigp.reshape(leading + (n_conn, D))
        mels = mels.reshape(leading + (n_conn,)).astype(jnp.float64)
        return sigp, mels


def ThiemannRegularisedVertexConstraint3d(
    lqx: "LqxWCL3D",
    vertex: Union[int, str],
    *,
    apply_lapse: bool = False,
    adjoint: bool = False,
):
    if adjoint:
        return Euclidean3dVertexTRCAdjointOperator(lqx, vertex, apply_lapse=apply_lapse)
    return Euclidean3dVertexTRCOperator(lqx, vertex, apply_lapse=apply_lapse)
