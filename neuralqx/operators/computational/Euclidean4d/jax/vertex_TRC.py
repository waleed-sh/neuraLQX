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

################### WARNING WARNING WARNING ###################
# The implementation below implements Thiemann's quantum
# Hamilton constraint acting on a KET. However, NetKet
# internally uses a row (bra) based implementation, which
# effectively means that the NON-ADJOINT CODE BELOW ACTS,
# MATHEMATICALLY, AS AN ADJOINT and the ADJOINT CODE BELOW
# ACTS AS A NON-ADJOINT, MATHEMATICALLY.
#
# Since the operator action is EASY TO SEE with the ket-based
# implementation, we have kept the naming convention.
################### WARNING WARNING WARNING ###################


from __future__ import annotations

from typing import Any
from typing import Tuple
from typing import Union
from typing import List

import jax
from jax import numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator

from neuralqx.utils.misc.graph import get_true_edge_triplet
from neuralqx.operators.types.computational_operator import ComputationalJaxOperator

Array = jnp.ndarray


#
#
#   low-level helpers (pure functions)


def _wrap_mod(vals: Array, delta: Array, mod_span: Array, state_min: Array) -> Array:
    # all scalars are JAX arrays to avoid static recompiles
    return ((vals + delta - state_min) % mod_span) + state_min


def _shift_one_component(
    sigma: Array,
    comp_idx: Array,
    delta: Array,
    mod_span: Array,
    state_min: Array,
) -> Array:
    # comp_idx is a scalar int64 as JAX array
    sel = jnp.take(sigma, comp_idx, axis=-1)
    new_sel = _wrap_mod(sel, delta, mod_span, state_min)
    return sigma.at[..., comp_idx].set(new_sel)


def _apply_loop_row(
    sigma: Array,
    comps_row: Array,
    shifts_row: Array,
    mult: int,
    mod_span: Array,
    state_min: Array,
) -> Array:
    M = comps_row.shape[0]
    mult_scalar = jnp.asarray(mult, dtype=shifts_row.dtype)

    def body(m, cur):
        idx = comps_row[m]
        sh = shifts_row[m] * mult_scalar
        return lax.cond(
            idx >= 0,
            lambda s: _shift_one_component(s, idx, sh, mod_span, state_min),
            lambda s: s,
            cur,
        )

    return lax.fori_loop(0, M, body, sigma)


def _volume_batch(
    sigma: Array,
    comps1: Array,
    comps2: Array,
    comps3: Array,
    signs: Array,
    fdtype: Any,
) -> Array:
    """Compute V_v(σ) for shape (B, N, D)."""
    B, N, D = sigma.shape
    sig = sigma.reshape(B * N, D)

    m1 = jnp.take(sig, comps1, axis=-1).astype(fdtype)
    m2 = jnp.take(sig, comps2, axis=-1).astype(fdtype)
    m3 = jnp.take(sig, comps3, axis=-1).astype(fdtype)

    cr = jnp.cross(m2, m3, axis=-1)
    dots = jnp.einsum("btc,btc->bt", m1, cr)

    Vr = jnp.sum(dots * signs[None, :], axis=1)
    V = jnp.sqrt(jnp.abs(Vr)).reshape(B, N)
    return V


#
#
#   top-level jitted kernel


def _evtrc_kernel_impl(
    *,
    # static
    gdim: int,
    n_terms: int,
    # dynamic arrays/scalars
    sigma: Array,
    comps1: Array,
    comps2: Array,
    comps3: Array,
    signs: Array,
    seg_comp_k: Array,
    loop_comps: Array,
    loop_shifts: Array,
    coeff: Array,
    mod_span: Array,
    state_min: Array,
    Nv: Array,
    T: Array,
    apply_lapse: Array,
) -> Tuple[Array, Array]:
    """
    Pure-JAX implementation. gdim and n_terms are static (Python ints).
    All other inputs are JAX arrays. Dtypes are inferred from arrays.
    """

    # infer dtypes from inputs (avoid passing dtype objects)
    fdtype = coeff.dtype
    idtype = seg_comp_k.dtype

    # promote (N,D) -> (1,N,D)
    single_input = sigma.ndim == 2
    if single_input:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape

    # early exit: no terms
    if n_terms == 0 or coeff.size == 0:
        sigma_p = sigma[:, :, None, :]
        mels = jnp.zeros((B, N, 1), dtype=fdtype)
        return (sigma_p[0], mels[0]) if single_input else (sigma_p, mels)

    # base volume
    v0 = _volume_batch(sigma, comps1, comps2, comps3, signs, fdtype)

    # deltaV over all (k,t)
    deltaV = jnp.zeros((gdim, n_terms, B, N), dtype=fdtype)

    def body_delta(i, dv):
        k = i // n_terms
        t = i - k * n_terms

        # scalar int64
        comp_idx = seg_comp_k[k, t]
        sig_shift = _shift_one_component(
            sigma,
            comp_idx,
            jnp.asarray(-1, dtype=idtype),
            mod_span,
            state_min,
        )

        vSi = _volume_batch(sig_shift, comps1, comps2, comps3, signs, fdtype)
        return dv.at[k, t].set((v0 - vSi).astype(fdtype))

    deltaV = lax.fori_loop(0, gdim * n_terms, body_delta, deltaV)

    # build output states
    sigma_alpha = jnp.zeros((gdim, n_terms, B, N, D), dtype=sigma.dtype)
    sigma_adag = jnp.zeros((gdim, n_terms, B, N, D), dtype=sigma.dtype)

    def body_states(i, carry):
        sa, sd = carry
        k = i // n_terms
        t = i - k * n_terms
        comps_row = loop_comps[k, t, :]
        shifts_row = loop_shifts[k, t, :]
        sig_plus = _apply_loop_row(
            sigma, comps_row, shifts_row, +1, mod_span, state_min
        )
        sig_minus = _apply_loop_row(
            sigma, comps_row, shifts_row, -1, mod_span, state_min
        )
        sa = sa.at[k, t].set(sig_plus)
        sd = sd.at[k, t].set(sig_minus)
        return (sa, sd)

    sigma_alpha, sigma_adag = lax.fori_loop(
        0, gdim * n_terms, body_states, (sigma_alpha, sigma_adag)
    )

    # matrix elements
    coeff_bt = coeff[None, :, None, None].astype(fdtype)
    mels_alpha = +coeff_bt * deltaV
    mels_adag = -coeff_bt * deltaV

    # pack connections interleaving (+, -)
    n_conn = 2 * gdim * n_terms
    sigmap = jnp.empty((B, N, n_conn, D), dtype=sigma.dtype)
    mels = jnp.empty((B, N, n_conn), dtype=fdtype)

    def write_conn(i, carry):
        Xp, Ml = carry
        k = i // n_terms
        t = i - k * n_terms
        base = 2 * i
        Xp = Xp.at[:, :, base, :].set(sigma_alpha[k, t])
        Xp = Xp.at[:, :, base + 1, :].set(sigma_adag[k, t])
        Ml = Ml.at[:, :, base].set(mels_alpha[k, t])
        Ml = Ml.at[:, :, base + 1].set(mels_adag[k, t])
        return (Xp, Ml)

    sigmap, mels = lax.fori_loop(0, gdim * n_terms, write_conn, (sigmap, mels))

    # global scale
    scale = jnp.where(apply_lapse.astype(jnp.int64) == 1, Nv / T, 1.0 / T).astype(
        fdtype
    )
    mels = (scale * mels).astype(fdtype)

    if single_input:
        return sigmap[0], mels[0]
    return sigmap, mels


# jit with static arguments for the loop sizes
_evtrc_kernel = jax.jit(_evtrc_kernel_impl, static_argnames=("gdim", "n_terms"))


@register_pytree_node_class
class EuclideanVertexTRCOperatorJax(ComputationalJaxOperator):
    """
    JAX-native version of EuclideanVertexTRCOperatorFast that:
      - is a PyTree (works inside jit/vmap/pmap)
      - uses a top-level jitted kernel (method itself is not jitted)
      - keeps all arrays in JAX
    """

    def __init__(
        self,
        lqx: "lqx",
        vertex: Union[str, int],
        *,
        apply_lapse: bool = True,
    ) -> None:

        # DiscreteJaxOperator requires hilbert
        super().__init__(lqx.H.hilbert_netket)

        H = lqx.H
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex

        # constants (as JAX arrays to avoid static recompiles)

        gauge_dim = int(H.gauge_dimensions)
        assert (
            gauge_dim == 3
        ), f"This operator assumes gauge_dim == 3 (U(1)^3), found {gauge_dim} instead."

        self._gauge_dim: int = int(lqx.H.gauge_dimensions)

        self._cutoff = jnp.asarray(int(H.cutoff), dtype=jnp.int64)

        self._state_min = jnp.asarray(
            int(H.allowed_basis_states.start), dtype=jnp.int64
        )
        self._mod_span = jnp.asarray(
            int(H.allowed_basis_states.length), dtype=jnp.int64
        )
        self._state_max = jnp.asarray(
            int(
                self._state_min
                + (self._mod_span - 1) * int(H.allowed_basis_states.step)
            ),
            dtype=jnp.int64,
        )

        self._n_edges_total = jnp.asarray(int(H.size // gauge_dim), dtype=jnp.int64)

        # admissibility -> encode by n_terms == 0 if false
        is_admissible = self.vertex in set(lqx._admissible_vertices)

        # lapse/T
        self._apply_lapse = jnp.asarray(1 if apply_lapse else 0, dtype=jnp.int64)
        if apply_lapse and hasattr(lqx, "lapses"):
            self._Nv = jnp.asarray(float(lqx.lapses[self.vertex]), dtype=jnp.float64)
        else:
            self._Nv = jnp.asarray(1.0, dtype=jnp.float64)

        level_terms = lqx._constraint_terms["0"][str(self.vertex)]
        self._T = jnp.asarray(float(level_terms["T"]), dtype=jnp.float64)

        # volume geometry
        triplets = H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = H.graph.signs[str(self.vertex)]

        offs = jnp.arange(gauge_dim, dtype=jnp.int64) * self._n_edges_total

        comps1, comps2, comps3, signs = [], [], [], []
        for triple in triplets:
            e1, e2, e3 = map(H.graph.edge_to_index, triple)
            comps1.append((jnp.asarray(e1, dtype=jnp.int64) + offs).astype(jnp.int64))
            comps2.append((jnp.asarray(e2, dtype=jnp.int64) + offs).astype(jnp.int64))
            comps3.append((jnp.asarray(e3, dtype=jnp.int64) + offs).astype(jnp.int64))
            signs.append(int(signs_dict[get_true_edge_triplet(triple, H.graph)]))

        self._comps1 = jnp.stack(comps1, axis=0).astype(jnp.int64)
        self._comps2 = jnp.stack(comps2, axis=0).astype(jnp.int64)
        self._comps3 = jnp.stack(comps3, axis=0).astype(jnp.int64)
        self._signs = jnp.asarray(signs, dtype=jnp.int64)

        # per-term loop structures
        reord_list: List[float] = []
        eps_over_L_list: List[float] = []
        seg_comp_all_list: List[jnp.ndarray] = []
        loop_comps_k_lists: List[List[jnp.ndarray]] = [[], [], []]
        loop_shifts_k_lists: List[List[jnp.ndarray]] = [[], [], []]

        for contrib in level_terms["triplets"]:
            eps = float(contrib["ε"])
            L = float(contrib["L"])
            eps_over_L = eps / L

            for term in contrib["terms"]:
                reord = float(term.reordering_sign)

                seg_edge = tuple(term.segment_edge)
                seg_eidx = int(H.graph.edge_to_index(seg_edge))
                seg_comps_all = (jnp.asarray(seg_eidx, dtype=jnp.int64) + offs).astype(
                    jnp.int64
                )

                edges_flat_all, shifts_flat_all = [], []
                for edge, data in term.dressed_minimal_loop:
                    eidx = int(H.graph.edge_to_index(edge))
                    comps_all = (jnp.asarray(eidx, dtype=jnp.int64) + offs).astype(
                        jnp.int64
                    )
                    s = +1 if data["type"] == "creation" else -1
                    edges_flat_all.append(comps_all)
                    shifts_flat_all.append(jnp.full((gauge_dim,), s, dtype=jnp.int64))

                if edges_flat_all:
                    loop_comps_all = jnp.concatenate(edges_flat_all, axis=0)
                    loop_shifts_all = jnp.concatenate(shifts_flat_all, axis=0)
                else:
                    loop_comps_all = jnp.zeros((0,), dtype=jnp.int64)
                    loop_shifts_all = jnp.zeros((0,), dtype=jnp.int64)

                for k in range(gauge_dim):
                    loop_comps_k_lists[k].append(loop_comps_all[k::gauge_dim])
                    loop_shifts_k_lists[k].append(loop_shifts_all[k::gauge_dim])

                seg_comp_all_list.append(seg_comps_all)
                reord_list.append(reord)
                eps_over_L_list.append(eps_over_L)

        n_terms = len(reord_list) if is_admissible else 0
        self._n_terms: int = int(n_terms)

        if n_terms == 0:
            # make minimal, valid tensors
            self._coeff = jnp.zeros((0,), dtype=jnp.float64)
            self._seg_comp_k = jnp.zeros((3, 0), dtype=jnp.int64)
            self._loop_comps = jnp.zeros((3, 0, 0), dtype=jnp.int64)
            self._loop_shifts = jnp.zeros((3, 0, 0), dtype=jnp.int64)
        else:
            reord = jnp.asarray(reord_list, dtype=jnp.float64)
            epsL = jnp.asarray(eps_over_L_list, dtype=jnp.float64)
            self._coeff = (reord * epsL).astype(jnp.float64)

            # segment indices per copy: (3, n_terms)
            seg_k = []
            for k in range(gauge_dim):
                seg_k.append(jnp.stack([s[k] for s in seg_comp_all_list], axis=0))
            self._seg_comp_k = jnp.stack(seg_k, axis=0).astype(jnp.int64)

            # loop arrays padded
            loop_comps_k_padded, loop_shifts_k_padded, Mk_list = [], [], []

            for k in range(gauge_dim):
                lengths = [int(a.size) for a in loop_comps_k_lists[k]]
                Mk = int(max(lengths) if lengths else 0)
                Mk_list.append(Mk)
                if Mk == 0:
                    loop_comps_k_padded.append(jnp.zeros((n_terms, 0), dtype=jnp.int64))
                    loop_shifts_k_padded.append(
                        jnp.zeros((n_terms, 0), dtype=jnp.int64)
                    )
                else:
                    comps_pad = jnp.stack(
                        [
                            jnp.pad(a, (0, Mk - a.size), constant_values=-1)
                            for a in loop_comps_k_lists[k]
                        ],
                        axis=0,
                    ).astype(jnp.int64)

                    shifts_pad = jnp.stack(
                        [
                            jnp.pad(
                                s.astype(jnp.int64), (0, Mk - s.size), constant_values=0
                            )
                            for s in loop_shifts_k_lists[k]
                        ],
                        axis=0,
                    ).astype(jnp.int64)

                    loop_comps_k_padded.append(comps_pad)
                    loop_shifts_k_padded.append(shifts_pad)

            M_max = int(max(Mk_list) if Mk_list else 0)

            if M_max == 0:
                self._loop_comps = jnp.zeros((gauge_dim, n_terms, 0), dtype=jnp.int64)
                self._loop_shifts = jnp.zeros((gauge_dim, n_terms, 0), dtype=jnp.int64)
            else:
                comps_3d, shifts_3d = [], []
                for k in range(gauge_dim):
                    curM = loop_comps_k_padded[k].shape[1]
                    if curM < M_max:
                        pad_w = ((0, 0), (0, M_max - curM))
                        comps_3d.append(
                            jnp.pad(loop_comps_k_padded[k], pad_w, constant_values=-1)
                        )
                        shifts_3d.append(
                            jnp.pad(loop_shifts_k_padded[k], pad_w, constant_values=0)
                        )
                    else:
                        comps_3d.append(loop_comps_k_padded[k])
                        shifts_3d.append(loop_shifts_k_padded[k])
                self._loop_comps = jnp.stack(comps_3d, axis=0).astype(jnp.int64)
                self._loop_shifts = jnp.stack(shifts_3d, axis=0).astype(jnp.int64)

        # dtypes (kept as Python attributes, used to cast)
        self._fdtype = jnp.float64
        self._idtype = jnp.int64

        self._mod_span_py: int = int(H.allowed_basis_states.length)
        self._state_min_py: int = int(H.allowed_basis_states.start)
        self._Nv_py: float = float(self._Nv)
        self._T_py: float = float(self._T)
        self._apply_lapse_py: bool = bool(apply_lapse)

        # device versions used in math:
        self._Nv = jnp.asarray(self._Nv_py, dtype=jnp.float64)
        self._T = jnp.asarray(self._T_py, dtype=jnp.float64)
        self._apply_lapse = jnp.asarray(
            1 if self._apply_lapse_py else 0, dtype=jnp.int64
        )

    @property
    def max_conn_size(self) -> int:
        # 2 * 3 * n_terms connections (+/- for each copy and term)
        return int(2 * int(self._gauge_dim) * int(self._n_terms))

    def _get_conn_padded(self, x):
        x = jnp.asarray(x, dtype=jnp.int64)

        σp, mels = _evtrc_kernel(
            # static
            gdim=self._gauge_dim,
            n_terms=self._n_terms,
            # dynamic (arrays)
            sigma=x,
            comps1=self._comps1,
            comps2=self._comps2,
            comps3=self._comps3,
            signs=self._signs,
            seg_comp_k=self._seg_comp_k,
            loop_comps=self._loop_comps,
            loop_shifts=self._loop_shifts,
            coeff=self._coeff,
            mod_span=self._mod_span,
            state_min=self._state_min,
            Nv=self._Nv,
            T=self._T,
            apply_lapse=self._apply_lapse,
        )
        return σp, mels

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def adjoint(self) -> "EuclideanVertexTRCAdjointOperatorJax":
        """Return the adjoint operator."""
        obj = EuclideanVertexTRCAdjointOperatorJax.__new__(
            EuclideanVertexTRCAdjointOperatorJax
        )
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def dtype(self):
        return self._fdtype

    #
    #
    #   pytree plumbing
    #   put arrays in the leaves, keep nothing static so recompiles are minimised

    def tree_flatten(self):
        leaves = (
            self._comps1,
            self._comps2,
            self._comps3,
            self._signs,
            self._seg_comp_k,
            self._loop_comps,
            self._loop_shifts,
            self._coeff,
            self._mod_span,
            self._state_min,
            self._Nv,
            self._T,
            self._apply_lapse,
        )
        aux = dict(
            hilbert=self.hilbert,
            gauge_dim=self._gauge_dim,
            n_terms=self._n_terms,
            fdtype=str(self._fdtype),
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (
            comps1,
            comps2,
            comps3,
            signs,
            seg_comp_k,
            loop_comps,
            loop_shifts,
            coeff,
            mod_span,
            state_min,
            Nv,
            T,
            apply_lapse,
        ) = leaves

        self = object.__new__(cls)
        DiscreteJaxOperator.__init__(self, aux["hilbert"])
        # restore arrays
        self._comps1, self._comps2, self._comps3, self._signs = (
            comps1,
            comps2,
            comps3,
            signs,
        )
        self._seg_comp_k, self._loop_comps, self._loop_shifts = (
            seg_comp_k,
            loop_comps,
            loop_shifts,
        )
        self._coeff = coeff
        self._mod_span, self._state_min, self._Nv, self._T, self._apply_lapse = (
            mod_span,
            state_min,
            Nv,
            T,
            apply_lapse,
        )
        # restore Python scalars
        self._gauge_dim = int(aux["gauge_dim"])
        self._n_terms = int(aux["n_terms"])
        # set dtype from coeff
        self._fdtype = jnp.float64
        self._idtype = jnp.int64
        return self


def _evtrc_adjoint_kernel_impl(
    *,
    gdim: int,
    n_terms: int,
    sigma: Array,
    comps1: Array,
    comps2: Array,
    comps3: Array,
    signs: Array,
    seg_comp_k: Array,
    loop_comps: Array,
    loop_shifts: Array,
    coeff: Array,
    mod_span: Array,
    state_min: Array,
    Nv: Array,
    T: Array,
    apply_lapse: Array,
) -> Tuple[Array, Array]:

    fdtype = coeff.dtype
    idtype = seg_comp_k.dtype

    single_input = sigma.ndim == 2
    if single_input:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape

    if n_terms == 0 or coeff.size == 0:
        sigma_p = sigma[:, :, None, :]
        mels = jnp.zeros((B, N, 1), dtype=fdtype)
        return (sigma_p[0], mels[0]) if single_input else (sigma_p, mels)

    # build branch states σ_α and σ_α† for all (k,t)
    sigma_alpha = jnp.zeros((gdim, n_terms, B, N, D), dtype=sigma.dtype)
    sigma_adag = jnp.zeros((gdim, n_terms, B, N, D), dtype=sigma.dtype)

    def body_states(i, carry):
        sa, sd = carry
        k = i // n_terms
        t = i - k * n_terms
        comps_row = loop_comps[k, t, :]
        shifts_row = loop_shifts[k, t, :]
        sig_plus = _apply_loop_row(
            sigma, comps_row, shifts_row, +1, mod_span, state_min
        )
        sig_minus = _apply_loop_row(
            sigma, comps_row, shifts_row, -1, mod_span, state_min
        )
        sa = sa.at[k, t].set(sig_plus)
        sd = sd.at[k, t].set(sig_minus)
        return (sa, sd)

    sigma_alpha, sigma_adag = lax.fori_loop(
        0, gdim * n_terms, body_states, (sigma_alpha, sigma_adag)
    )

    # ΔV on branch states (adjoint)
    deltaV_alpha = jnp.zeros((gdim, n_terms, B, N), dtype=fdtype)
    deltaV_adag = jnp.zeros((gdim, n_terms, B, N), dtype=fdtype)

    def body_delta(i, carry):
        dA, dD = carry
        k = i // n_terms
        t = i - k * n_terms
        comp_idx = seg_comp_k[k, t]

        # α branch
        sigA = sigma_alpha[k, t]
        vA = _volume_batch(sigA, comps1, comps2, comps3, signs, fdtype)
        sigA_sinv = _shift_one_component(
            sigA, comp_idx, jnp.asarray(-1, dtype=idtype), mod_span, state_min
        )
        vA_sinv = _volume_batch(sigA_sinv, comps1, comps2, comps3, signs, fdtype)
        dA = dA.at[k, t].set((vA - vA_sinv).astype(fdtype))

        # α† branch
        sigD = sigma_adag[k, t]
        vD = _volume_batch(sigD, comps1, comps2, comps3, signs, fdtype)
        sigD_sinv = _shift_one_component(
            sigD, comp_idx, jnp.asarray(-1, dtype=idtype), mod_span, state_min
        )
        vD_sinv = _volume_batch(sigD_sinv, comps1, comps2, comps3, signs, fdtype)
        dD = dD.at[k, t].set((vD - vD_sinv).astype(fdtype))

        return (dA, dD)

    deltaV_alpha, deltaV_adag = lax.fori_loop(
        0, gdim * n_terms, body_delta, (deltaV_alpha, deltaV_adag)
    )

    # signs for adjoint
    coeff_bt = coeff[None, :, None, None].astype(fdtype)
    mels_alpha = -coeff_bt * deltaV_alpha
    mels_adag = +coeff_bt * deltaV_adag

    # pack outputs interleaving (α, α†)
    n_conn = 2 * gdim * n_terms
    sigmap = jnp.empty((B, N, n_conn, D), dtype=sigma.dtype)
    mels = jnp.empty((B, N, n_conn), dtype=fdtype)

    def write_conn(i, carry):
        Xp, Ml = carry
        k = i // n_terms
        t = i - k * n_terms
        base = 2 * i
        Xp = Xp.at[:, :, base, :].set(sigma_alpha[k, t])
        Xp = Xp.at[:, :, base + 1, :].set(sigma_adag[k, t])
        Ml = Ml.at[:, :, base].set(mels_alpha[k, t])
        Ml = Ml.at[:, :, base + 1].set(mels_adag[k, t])
        return (Xp, Ml)

    sigmap, mels = lax.fori_loop(0, gdim * n_terms, write_conn, (sigmap, mels))

    scale = jnp.where(apply_lapse.astype(jnp.int64) == 1, Nv / T, 1.0 / T).astype(
        fdtype
    )
    mels = (scale * mels).astype(fdtype)

    if single_input:
        return sigmap[0], mels[0]
    return sigmap, mels


_evtrc_adjoint_kernel = jax.jit(
    _evtrc_adjoint_kernel_impl, static_argnames=("gdim", "n_terms")
)


@register_pytree_node_class
class EuclideanVertexTRCAdjointOperatorJax(EuclideanVertexTRCOperatorJax):
    """
    JAX-native adjoint of EuclideanVertexTRCOperatorJax.
    Only the top-level kernel changes to evaluate ΔV on branch states and flip signs.
    """

    def _get_conn_padded(self, x):
        x = jnp.asarray(x, dtype=jnp.int64)
        σp, mels = _evtrc_adjoint_kernel(
            gdim=self._gauge_dim,
            n_terms=self._n_terms,
            sigma=x,
            comps1=self._comps1,
            comps2=self._comps2,
            comps3=self._comps3,
            signs=self._signs,
            seg_comp_k=self._seg_comp_k,
            loop_comps=self._loop_comps,
            loop_shifts=self._loop_shifts,
            coeff=self._coeff,
            mod_span=self._mod_span,
            state_min=self._state_min,
            Nv=self._Nv,
            T=self._T,
            apply_lapse=self._apply_lapse,
        )
        return σp, mels

    @property
    def adjoint(self) -> "EuclideanVertexTRCOperatorJax":
        """Return the non-adjoint operator."""
        obj = EuclideanVertexTRCOperatorJax.__new__(EuclideanVertexTRCOperatorJax)
        obj.__dict__.update(self.__dict__)
        return obj


def ThiemannRegularisedVertexConstraintJax(
    lqx: "lqx",
    vertex: Union[str, int],
    *,
    apply_lapse: bool = True,
    adjoint: bool = False,
):
    if adjoint:
        return EuclideanVertexTRCAdjointOperatorJax(
            lqx=lqx, vertex=vertex, apply_lapse=apply_lapse
        )
    return EuclideanVertexTRCOperatorJax(
        lqx=lqx, vertex=vertex, apply_lapse=apply_lapse
    )
