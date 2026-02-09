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
from jax import lax
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


@jax.jit
def _sqrt_volume_2p1(
    sigma: jnp.ndarray,  # (M, D) int
    comps1: jnp.ndarray,  # (P, 3) int
    comps2: jnp.ndarray,  # (P, 3) int
    signs: jnp.ndarray,  # (P,) int
) -> jnp.ndarray:
    """
    sqrt(V_v)(sigma) for the 2+1 U(1)^3 vertex volume (area) eigenvalue:
        V_v(sigma) = || sum_pairs eps(p) * (m_{e1} x m_{e2}) ||
    and we return sqrt(V_v).
    """
    P = signs.shape[0]
    if P == 0:
        return jnp.zeros((sigma.shape[0],), dtype=jnp.float64)

    m1 = jnp.take(sigma, comps1, axis=-1)
    m2 = jnp.take(sigma, comps2, axis=-1)

    cr = jnp.cross(m1, m2, axis=-1)  # (M, P, 3)
    vec = jnp.sum(cr * signs[None, :, None], axis=1)  # (M, 3)

    V = jnp.linalg.norm(vec, axis=-1).astype(jnp.float64)  # (M,)
    return jnp.sqrt(V).astype(jnp.float64)


def _wrap_mod(
    vals: jnp.ndarray, delta: jnp.ndarray, mod_span: int, state_min: int
) -> jnp.ndarray:
    # maps into [state_min, state_min + mod_span - 1]
    return ((vals + delta - state_min) % mod_span) + state_min


def _shift_one_component(
    sig: jnp.ndarray,
    comp_idx: jnp.ndarray,
    delta: jnp.ndarray,
    mod_span: int,
    state_min: int,
) -> jnp.ndarray:
    comp_idx = jnp.asarray(comp_idx, dtype=jnp.int32)
    delta = jnp.asarray(delta, dtype=sig.dtype)

    sel = jnp.take(sig, comp_idx, axis=-1)
    new_sel = _wrap_mod(sel, delta, mod_span=mod_span, state_min=state_min)
    return sig.at[:, comp_idx].set(new_sel)


def _apply_loop_row(
    sig: jnp.ndarray,
    comps_row: jnp.ndarray,
    shifts_row: jnp.ndarray,
    mod_span: int,
    state_min: int,
) -> jnp.ndarray:
    Lmax = comps_row.shape[0]

    def body(i, cur):
        idx = comps_row[i]
        sh = shifts_row[i]
        return lax.cond(
            idx >= 0,
            lambda s: _shift_one_component(
                s, idx, sh, mod_span=mod_span, state_min=state_min
            ),
            lambda s: s,
            cur,
        )

    return lax.fori_loop(0, Lmax, body, sig)


@jax.jit
def _hv_kernel(
    sigma: jnp.ndarray,
    vol_c1: jnp.ndarray,
    vol_c2: jnp.ndarray,
    vol_s: jnp.ndarray,
    weight: jnp.ndarray,
    seg_k_comp: jnp.ndarray,
    seg_l_comp: jnp.ndarray,
    seg_k_shift: jnp.ndarray,
    seg_l_shift: jnp.ndarray,
    loop_comps: jnp.ndarray,
    loop_shifts: jnp.ndarray,
    mod_span: int,
    state_min: int,
    Nv: float,
):
    sigma = jnp.asarray(sigma, dtype=jnp.int32)

    leading = sigma.shape[:-1]
    D = sigma.shape[-1]

    sig0 = sigma.reshape((-1, D))

    T = weight.shape[0]
    C = 3 * T

    s0 = _sqrt_volume_2p1(sig0, vol_c1, vol_c2, vol_s)

    # outputs
    sigp = jnp.broadcast_to(sig0[:, None, :], (sig0.shape[0], C, D)).copy()
    mels = jnp.zeros((sig0.shape[0], C), dtype=jnp.float64)

    def body(i, carry):
        sigp_, mels_ = carry
        c = i // T
        t = i - c * T

        k_idx = seg_k_comp[c, t]
        l_idx = seg_l_comp[c, t]

        dk_shift = seg_k_shift[t]
        dl_shift = seg_l_shift[t]

        sig_k = _shift_one_component(
            sig0, k_idx, dk_shift, mod_span=mod_span, state_min=state_min
        )
        sk = _sqrt_volume_2p1(sig_k, vol_c1, vol_c2, vol_s)
        dk = s0 - sk

        sig_l = _shift_one_component(
            sig0, l_idx, dl_shift, mod_span=mod_span, state_min=state_min
        )
        sl = _sqrt_volume_2p1(sig_l, vol_c1, vol_c2, vol_s)
        dl = s0 - sl

        comps_row = loop_comps[c, t, :]
        shifts_row = loop_shifts[c, t, :]
        sig_fin = _apply_loop_row(
            sig0, comps_row, shifts_row, mod_span=mod_span, state_min=state_min
        )

        mel = (Nv * weight[t]) * dk * dl

        sigp_ = sigp_.at[:, i, :].set(sig_fin)
        mels_ = mels_.at[:, i].set(mel)
        return sigp_, mels_

    sigp, mels = lax.fori_loop(0, C, body, (sigp, mels))

    # reshape back to padded NetKet format
    sigp = sigp.reshape(leading + (C, D))
    mels = mels.reshape(leading + (C,))
    return sigp, mels


def _emanates_from(graph, v: int, edge: tuple) -> bool:
    """
    True if the directed (keyed) edge is oriented in the stored graph such that it emanates from v.
    """
    u, w = edge[0], edge[1]
    if v not in (u, w):
        return False
    rev = (edge[1], edge[0], edge[2])
    return (edge in graph.edges and edge[0] == v) or (
        rev in graph.edges and rev[0] == v
    )


def _pad_2d(rows: List[jnp.ndarray], pad_value: int, dtype) -> jnp.ndarray:
    """Stack 1D arrays into (T, Lmax) with padding."""
    T = len(rows)
    Lmax = max((int(r.size) for r in rows), default=0)
    if Lmax == 0:
        return jnp.zeros((T, 0), dtype=dtype)
    return jnp.stack(
        [
            jnp.pad(r.astype(dtype), (0, Lmax - r.size), constant_values=pad_value)
            for r in rows
        ],
        axis=0,
    )


@register_pytree_node_class
class Euclidean3dVertexTRCOperatorJax(ComputationalJaxOperator):
    r"""
    Single-vertex Thiemann-regularised Euclidean Hamiltonian constraint H_v for the 2+1 U(1)^3 WCL model
    as a JAX-compatible ComputationalOperator.

    For each term t (constructed from LqxWCL3D._sum_terms and _epsilon) and each U(1) copy c=0,1,2, this emits
    one connection:
        σ'_{t,c} = h_{α_t}^{(c)} σ

    with matrix element:
        <σ'_{t,c}|H_v|σ> = N(v) w_t Δ_k^{(c)}(σ) Δ_l^{(c)}(σ)

    where:
        Δ_s^{(c)}(σ) = sqrt(V_v(σ)) - sqrt(V_v(h_{s}^{(c)-1} σ)).

    No Taylor expansions are used. sqrt(V_v) is evaluated directly from the 2+1 vertex volume kernel.
    """

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return jnp.float64

    @property
    def max_conn_size(self) -> int:
        return self._max_conn

    def __init__(
        self,
        lqx: "LqxWCL3D",
        vertex: Union[int, str],
        *,
        apply_lapse: bool = False,
    ):
        super().__init__(lqx.hilbert.hilbert)

        self._H = lqx.hilbert
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex

        self.gauge_dim = int(lqx.hilbert.gauge_dimensions)
        assert (
            self.gauge_dim == 3
        ), f"{type(self).__name__} assumes gauge_dim == 3 (U(1)^3)."

        # D = 3 * n_edges_total
        self.n_edges_total = int(lqx.hilbert.size // self.gauge_dim)
        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        # modular arithmetic range
        self._mod_span = int(lqx.hilbert.allowed_basis_states.length)
        self._state_min = int(lqx.hilbert.allowed_basis_states.start)

        # lapse
        self._apply_lapse = bool(apply_lapse)
        self._Nv = (
            float(lqx.lapses[self.vertex])
            if (apply_lapse and hasattr(lqx, "lapses"))
            else 1.0
        )

        pairs = lqx.hilbert.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = lqx.hilbert.graph.signs[str(self.vertex)]

        comps1, comps2, signs = [], [], []
        for pair in pairs:
            e1, e2 = map(lqx.hilbert.graph.edge_to_index, pair)
            comps1.append((jnp.int32(e1) + offsets).astype(jnp.int32))
            comps2.append((jnp.int32(e2) + offsets).astype(jnp.int32))
            signs.append(int(signs_dict[str(pair)]))

        if comps1:
            self._vol_c1 = jnp.stack(comps1, axis=0).astype(jnp.int32)  # (P,3)
            self._vol_c2 = jnp.stack(comps2, axis=0).astype(jnp.int32)  # (P,3)
            self._vol_s = jnp.asarray(signs, dtype=jnp.int32)  # (P,)
        else:
            self._vol_c1 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._vol_c2 = jnp.zeros((0, 3), dtype=jnp.int32)
            self._vol_s = jnp.zeros((0,), dtype=jnp.int32)

        #
        #
        #   build term list at v

        weight: List[float] = []
        segk_base: List[int] = []
        segl_base: List[int] = []
        segk_shift: List[int] = []
        segl_shift: List[int] = []
        loop_edges_rows: List[jnp.ndarray] = []
        loop_shifts_rows: List[jnp.ndarray] = []

        for triang_key, triang in lqx._sum_terms.items():
            terms_at_v = triang["Terms"].get(str(self.vertex), None)
            if terms_at_v is None:
                continue

            # pick the first minimal loop of the triangulation pair as the loop label
            min_loop1 = ast.literal_eval(triang_key)[0]
            loop_idx = lqx.hilbert.graph.minimal_loops().index(min_loop1)
            dressed_loop = lqx.hilbert.graph.dressed_minimal_loops()[loop_idx]

            ij_list = terms_at_v["ij"]
            kl_list = terms_at_v["kl"]

            for ij in ij_list:
                for kl in kl_list:
                    w = float(lqx._epsilon[str(ij)] * lqx._epsilon[str(kl)])

                    # eps(ij) sets loop orientation: eps=-1 -> reverse loop
                    loop_flip = -1 if (lqx._epsilon[str(ij)] == -1) else +1

                    # segment edges k,l are the two edges in kl (keyed)
                    ek = kl[0]
                    el = kl[1]

                    eidx_k = int(lqx.hilbert.graph.edge_to_index(ek))
                    eidx_l = int(lqx.hilbert.graph.edge_to_index(el))

                    # ket operator uses h_s^{-1} for the Δ's
                    inv_k = (
                        -1 if _emanates_from(lqx.hilbert.graph, self.vertex, ek) else +1
                    )
                    inv_l = (
                        -1 if _emanates_from(lqx.hilbert.graph, self.vertex, el) else +1
                    )

                    # loop shifts in base-edge index space (copy resolved later)
                    le, ls = [], []
                    for edge, data in dressed_loop:
                        eidx = int(lqx.hilbert.graph.edge_to_index(edge))
                        s = +1 if data["type"] == "creation" else -1
                        le.append(eidx)
                        ls.append(loop_flip * s)

                    weight.append(w)
                    segk_base.append(eidx_k)
                    segl_base.append(eidx_l)
                    segk_shift.append(inv_k)
                    segl_shift.append(inv_l)
                    loop_edges_rows.append(jnp.asarray(le, dtype=jnp.int32))
                    loop_shifts_rows.append(jnp.asarray(ls, dtype=jnp.int32))

        self._Tn = int(len(weight))
        self._max_conn = max(1, self.gauge_dim * self._Tn)

        # store term arrays
        self._w = (
            jnp.asarray(weight, dtype=jnp.float64)
            if self._Tn
            else jnp.zeros((0,), dtype=jnp.float64)
        )

        # segment comps per copy: (3, T)
        if self._Tn:
            segk_base = jnp.asarray(segk_base, dtype=jnp.int32)
            segl_base = jnp.asarray(segl_base, dtype=jnp.int32)
            self._segk_comp = (offsets[:, None] + segk_base[None, :]).astype(jnp.int32)
            self._segl_comp = (offsets[:, None] + segl_base[None, :]).astype(jnp.int32)
            self._segk_shift = jnp.asarray(segk_shift, dtype=jnp.int32)
            self._segl_shift = jnp.asarray(segl_shift, dtype=jnp.int32)
        else:
            self._segk_comp = jnp.zeros((3, 0), dtype=jnp.int32)
            self._segl_comp = jnp.zeros((3, 0), dtype=jnp.int32)
            self._segk_shift = jnp.zeros((0,), dtype=jnp.int32)
            self._segl_shift = jnp.zeros((0,), dtype=jnp.int32)

        # pad loop rows to (T, Lmax), then lift to (3, T, Lmax) by adding offsets
        loop_edges_2d = (
            _pad_2d(loop_edges_rows, pad_value=-1, dtype=jnp.int32)
            if self._Tn
            else jnp.zeros((0, 0), dtype=jnp.int32)
        )
        loop_shifts_2d = (
            _pad_2d(loop_shifts_rows, pad_value=0, dtype=jnp.int32)
            if self._Tn
            else jnp.zeros((0, 0), dtype=jnp.int32)
        )

        # copy-resolved indices: (3,T,Lmax)
        if self._Tn:
            Lmax = loop_edges_2d.shape[1]
            # add offsets only where edge != -1
            loop_comps = []
            for c in range(3):
                off = int(offsets[c])
                row = jnp.where(
                    loop_edges_2d >= 0, loop_edges_2d + off, loop_edges_2d
                ).astype(jnp.int32)
                loop_comps.append(row)
            self._loop_comps = jnp.stack(loop_comps, axis=0)  # (3,T,Lmax)
            self._loop_shifts = jnp.broadcast_to(
                loop_shifts_2d[None, :, :], (3, self._Tn, Lmax)
            ).astype(jnp.int32)
        else:
            self._loop_comps = jnp.zeros((3, 0, 0), dtype=jnp.int32)
            self._loop_shifts = jnp.zeros((3, 0, 0), dtype=jnp.int32)

    #
    #
    #   pytree support

    def tree_flatten(self):
        leaves = (
            self._vol_c1,
            self._vol_c2,
            self._vol_s,
            self._w,
            self._segk_comp,
            self._segl_comp,
            self._segk_shift,
            self._segl_shift,
            self._loop_comps,
            self._loop_shifts,
        )

        aux = {
            "hilbert": self.hilbert,
            "vertex": self.vertex,
            "gauge_dim": self.gauge_dim,
            "n_edges_total": self.n_edges_total,
            "mod_span": self._mod_span,
            "state_min": self._state_min,
            "apply_lapse": self._apply_lapse,
            "Nv": self._Nv,
            "max_conn": self._max_conn,
            "Tn": int(self._Tn),
        }
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (
            vol_c1,
            vol_c2,
            vol_s,
            weight,
            segk_comp,
            segl_comp,
            segk_shift,
            segl_shift,
            loop_comps,
            loop_shifts,
        ) = leaves

        obj = cls.__new__(cls)

        object.__setattr__(obj, "_hilbert", aux["hilbert"])

        obj.vertex = aux["vertex"]
        obj.gauge_dim = aux["gauge_dim"]
        obj.n_edges_total = aux["n_edges_total"]
        obj._mod_span = aux["mod_span"]
        obj._state_min = aux["state_min"]
        obj._apply_lapse = aux["apply_lapse"]
        obj._Nv = aux["Nv"]
        obj._max_conn = aux["max_conn"]
        obj._Tn = aux["Tn"]

        obj._vol_c1 = vol_c1
        obj._vol_c2 = vol_c2
        obj._vol_s = vol_s
        obj._w = weight
        obj._segk_comp = segk_comp
        obj._segl_comp = segl_comp
        obj._segk_shift = segk_shift
        obj._segl_shift = segl_shift
        obj._loop_comps = loop_comps
        obj._loop_shifts = loop_shifts

        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int32)
        if self._Tn == 0:
            return x[..., None, :], jnp.zeros(x.shape[:-1] + (1,), dtype=jnp.float64)

        σp, mels = _hv_kernel(
            sigma=x,
            vol_c1=self._vol_c1,
            vol_c2=self._vol_c2,
            vol_s=self._vol_s,
            weight=self._w,
            seg_k_comp=self._segk_comp,
            seg_l_comp=self._segl_comp,
            seg_k_shift=self._segk_shift,
            seg_l_shift=self._segl_shift,
            loop_comps=self._loop_comps,
            loop_shifts=self._loop_shifts,
            mod_span=self._mod_span,
            state_min=self._state_min,
            Nv=self._Nv,
        )
        return σp, mels


@register_pytree_node_class
class Euclidean3dVertexTRCAdjointOperatorJax(Euclidean3dVertexTRCOperatorJax):
    r"""
    Adjoint of Euclidean3dVertexTRCOperatorJax.

    Differences vs the ket-ordered operator:
      - segment differences use forward shifts (h_s) instead of inverse shifts (h_s^{-1})
        i.e. we flip seg shifts: seg_shift -> -seg_shift
      - loop holonomy is daggered, i.e. loop shifts negated.
    """

    def __init__(
        self,
        lqx: "LqxWCL3D",
        vertex: Union[int, str],
        *,
        apply_lapse: bool = False,
    ):
        super().__init__(lqx, vertex, apply_lapse=apply_lapse)

        # flip segment shifts: h_s^{-1} -> h_s
        self._segk_shift = (-self._segk_shift).astype(jnp.int32)
        self._segl_shift = (-self._segl_shift).astype(jnp.int32)

        # dagger loop: h_alpha -> h_alpha^\dagger
        self._loop_shifts = (-self._loop_shifts).astype(jnp.int32)


def ThiemannRegularisedVertexConstraint3dJax(
    lqx: "LqxWCL3D",
    vertex: Union[int, str],
    *,
    apply_lapse: bool = False,
    adjoint: bool = False,
):
    if adjoint:
        return Euclidean3dVertexTRCAdjointOperatorJax(
            lqx, vertex, apply_lapse=apply_lapse
        )
    return Euclidean3dVertexTRCOperatorJax(lqx, vertex, apply_lapse=apply_lapse)
