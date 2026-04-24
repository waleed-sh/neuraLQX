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

"""
An implementation of the vertex Thiemann regularised Euclidean quantum Hamilton constraint of
the 4-d model of Euclidean LQG in the WCL as a lightweight ComputationalOperator
"""

from functools import partial

from typing import Union
from typing import List


import jax
import jax.numpy as jnp
from jax import lax
from netket.utils.types import DType

from neuralqx.utils.misc.graph import get_true_edge_triplet
from neuralqx.operators.types.computational_operator import ComputationalOperator


class EuclideanVertexTRCOperator(ComputationalOperator):
    r"""
    Single-vertex Euclidean Thiemann-regularised constraint :math:`H_v` in the 3+1-d U(1)^3 WCL
    model, copy-resolved on the U(1)^3 factors.

    Mathematically:
      .. math::

        H_v
        =
        \frac{1}{T_v}\;
        \sum_{\text{triplets } j\ni v}\frac{\varepsilon_j}{L_j}
        \sum_{t\in j}\text{reord}_t
        \sum_{k=1}^{3}
        \Big(h_{\alpha_t}^{(k)} - h_{\alpha_t}^{(k)\dagger}\Big)\,
        \Big(V_v - h_{s_t}^{(k)} V_v h_{s_t}^{(k)-1}\Big)\,,

      where the superscript :math:`(k)` means "act only on the k-th U(1) copy".

    Action on a configuration :math:`\sigma` (suppressing indices :math:`j,t,k` for brevity):

      .. math::

        \langle h_{\alpha}^{(k)}\sigma | H_v | \sigma \rangle
        = \frac{1}{T_v}\,\frac{\varepsilon}{L}\,\text{reord}\;
          \bigl[V_v(\sigma)-V_v(h_{s}^{(k)-1}\sigma)\bigr],

        \qquad
        \langle h_{\alpha}^{(k)\dagger}\sigma | H_v | \sigma \rangle
        = - \frac{1}{T_v}\,\frac{\varepsilon}{L}\,\text{reord}\;
          \bigl[V_v(\sigma)-V_v(h_{s}^{(k)-1}\sigma)\bigr].

    Computationally:
      - We precompute static index/shifts for each term and for each copy k=0,1,2
      - We emit 6 connections per term (2 signs x 3 copies)
      - Shapes are kept static for JAX jit: lists -> tuples, fixed loop lengths
    """

    def __init__(
        self,
        lqx: "lqx",
        vertex: str,
        *,
        apply_lapse: bool = True,
    ):
        #
        #
        #   High-level plan of __init__
        #       We precompute all static data so the later kernel has only numeric loops with
        #       fixed bounds. We keep the volume machinery identical to the original above
        #       implementation, but we refine loop/segment holonomy data so that we have
        #       "per-copy" index sets for k=0,1,2. This lets us implement h^{(k)} actions.

        super().__init__(lqx.H.hilbert_netket)

        # store a short-handle to H and the vertex id, coercing to int if needed
        self._H = lqx.H
        self.vertex = int(vertex) if not isinstance(vertex, int) else vertex

        # store the integer cutoff used for modular addition on charges (±1 shifts)
        self.cutoff = int(lqx.H.cutoff)

        # get the allowed quantum numbers

        # a span for the modded addition
        self._mod_span = int(lqx.H.allowed_basis_states.length)

        # lowest
        self._state_min = int(lqx.H.allowed_basis_states.start)

        # highest
        self._state_max = int(
            self._state_min
            + (self._mod_span - 1) * int(lqx.H.allowed_basis_states.step)
        )

        # read and verify that the gauge structure is U(1)^3 (i.e., 3 components per edge)
        self.gauge_dim = int(lqx.H.gauge_dimensions)

        # Dtypes
        # matrix elements and volume computations
        self._fdtype: DType = jnp.float64
        # indices/charges
        self._idtype = jnp.int64

        assert (
            self.gauge_dim == 3
        ), f"This operator assumes gauge_dim == 3 (U(1)^3), found {self.gauge_dim} instead."

        # compute the total number of edges at a single copy level (so D = n_edges_total * 3)
        self.n_edges_total = int(lqx.H.size // self.gauge_dim)

        # cache whether this vertex is admissible (compile-time boolean)
        self._is_admissible = self.vertex in set(lqx._admissible_vertices)

        # whether to multiply by the lapse N(v) (otherwise we return a bare H_v)
        self._apply_lapse = bool(apply_lapse)

        # if lapse is enabled and available, fetch N(v), else default to 1.0.
        if self._apply_lapse and hasattr(lqx, "lapses"):
            self._Nv = float(lqx.lapses[self.vertex])
        else:
            self._Nv = 1.0

        #
        #
        #   precompute volume geometry
        #       we collect oriented edge triplets (e1,e2,e3) adjacent to v and their orientation
        #       signs
        triplets = lqx.H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = lqx.H.graph.signs[str(self.vertex)]

        # prepare containers for the component indices (for m_{e1}, m_{e2}, m_{e3}) and signs
        comps1, comps2, comps3, signs = [], [], [], []

        # offsets to reach the 3 copies: [0, n_edges_total, 2*n_edges_total]
        offs = jnp.arange(self.gauge_dim) * self.n_edges_total

        # loop over oriented triplets adjacent to the vertex
        for triple in triplets:
            # map each (directed) edge to its base (level-0) integer index
            e1, e2, e3 = map(lqx.H.graph.edge_to_index, triple)

            # record the 3 component indices for each edge: (e + offs) gives the U(1)^3 slots
            comps1.append((e1 + offs).astype(jnp.int32))
            comps2.append((e2 + offs).astype(jnp.int32))
            comps3.append((e3 + offs).astype(jnp.int32))

            # recover the orientation (ε_or) for this embedded triplet
            signs.append(int(signs_dict[get_true_edge_triplet(triple, lqx.H.graph)]))

        # stack lists into arrays with static shapes (JAX friendly)
        self._comps1 = jnp.stack(comps1, axis=0).astype(jnp.int32)
        self._comps2 = jnp.stack(comps2, axis=0).astype(jnp.int32)
        self._comps3 = jnp.stack(comps3, axis=0).astype(jnp.int32)
        self._signs = jnp.array(signs, dtype=jnp.int32)

        # read precomputed constraint terms at graph level "0"
        level_terms = lqx._constraint_terms["0"][str(self.vertex)]

        # global normalization T_v, applied once at the very end
        self._T = float(level_terms["T"])

        #
        #
        #   we will build per-term data. For each contribution (triplet j) we store:
        #       - eps_over_L: (ε_j / L_j), constant within the triplet (replicated to terms)
        #       - reordering: reord_t ∈ {±1}
        #       - seg_comps: (3,) indices for the segment s_t across copies (we
        #         will pick copy k later)
        #       - loop_comps_k: tuple [k=0...2] of index arrays for h_{α_t}^{(k)}
        #       - loop_shifts_k: tuple [k=0...2] of ±1 shifts matching loop_comps_k

        # (n_terms,) float64
        reord_list = []

        # (n_terms,) float64
        eps_over_L_list = []

        # tuple of (3,) int32 per term
        seg_comp_list = []

        # three parallel lists, one per copy k
        loop_comps_k_list = [[], [], []]

        # three parallel lists, one per copy k
        loop_shifts_k_list = [[], [], []]

        # iterate over all triplets contributing to this vertex
        for contrib in level_terms["triplets"]:
            # extract ε and L for this triplet and form ε/L
            eps = float(contrib["ε"])
            L = float(contrib["L"])
            eps_over_L = eps / L

            # loop over all terms contributed by this triplet
            for term in contrib["terms"]:
                # reordering sign for this term (depends on loop orientation conventions)
                reord = float(term.reordering_sign)

                # segment s_t: get its base edge index and the 3 component positions (all copies)
                seg_edge = tuple(term.segment_edge)
                seg_eidx = lqx.H.graph.edge_to_index(seg_edge)

                # shape (3,)
                seg_comps_all = (seg_eidx + offs).astype(jnp.int32)

                # minimal loop α_t is a sequence of (edge, creation/annihilation) along dressed loop
                # we first collect the "all-copies" flattened layout, then split by copy k
                edges_flat_all, shifts_flat_all = [], []

                for edge, data in term.dressed_minimal_loop:
                    # map edge to base index
                    eidx = lqx.H.graph.edge_to_index(edge)

                    # indices for all three copies of this edge, shape (3,)
                    comps_all = (eidx + offs).astype(jnp.int32)

                    # shift sign (+1 for creation, -1 for annihilation)
                    s = +1 if data["type"] == "creation" else -1

                    # append three indices (for the three copies) and three identical shifts

                    # (3,)
                    edges_flat_all.append(comps_all)

                    # (3,)
                    shifts_flat_all.append(
                        jnp.full((self.gauge_dim,), s, dtype=self._idtype)
                    )

                # concatenate across loop edges, handle degenerate loops gracefully
                if edges_flat_all:

                    # shape (3*M,)
                    loop_comps_all = jnp.concatenate(edges_flat_all, axis=0)

                    # shape (3*M,)
                    loop_shifts_all = jnp.concatenate(shifts_flat_all, axis=0)
                else:
                    loop_comps_all = jnp.zeros((0,), dtype=jnp.int32)
                    loop_shifts_all = jnp.zeros((0,), dtype=self._idtype)

                # split the (3*M,) all-copies arrays into three copy-resolved arrays:
                #   copy k picks items k, k+3, k+6, ... i.e. slicing with step 3.
                for k in range(self.gauge_dim):

                    # extract copy-k indices/shifts, shapes remain static per term instance
                    loop_comps_k = loop_comps_all[k :: self.gauge_dim]
                    loop_shifts_k = loop_shifts_all[k :: self.gauge_dim]

                    # append to the per-k containers
                    loop_comps_k_list[k].append(loop_comps_k)
                    loop_shifts_k_list[k].append(loop_shifts_k)

                # append shared per-term data
                # keep all 3, we will pick k at runtime
                seg_comp_list.append(seg_comps_all)
                reord_list.append(reord)
                eps_over_L_list.append(eps_over_L)

        # freeze containers into tuples/arrays for JAX static shapes

        # number of terms contributing at this vertex
        self._n_terms = len(reord_list)

        # per-term arrays: reordering sign and ε/L factor

        # shape (n_terms,)
        self._reord = jnp.asarray(reord_list, dtype=jnp.float64)

        # shape (n_terms,)
        self._eps_over_L = jnp.asarray(eps_over_L_list, dtype=jnp.float64)

        # per-term segment components (all three copies)
        # each entry is shape (3,)
        # this is basically a tuple length n_terms of (3,) int32 arrays
        self._seg_comps_all = tuple(seg_comp_list)

        # per-copy loop data: for each k in {0,1,2}, we store a tuple of arrays (one per term)
        # each per-term array has shape (M_t_k,) possibly zero-length for degenerate loops

        # 3-tuples of tuples
        self._loop_comps_k = tuple(tuple(lst) for lst in loop_comps_k_list)

        # 3-tuples of tuples
        self._loop_shifts_k = tuple(tuple(lst) for lst in loop_shifts_k_list)

    #
    #
    #   Operator properties

    @property
    def is_hermitian(self):
        return False

    @property
    def adjoint(self) -> "EuclideanVertexTRCAdjointOperator":
        """Return the adjoint operator."""
        obj = EuclideanVertexTRCAdjointOperator.__new__(
            EuclideanVertexTRCAdjointOperator
        )
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def dtype(self):
        return jnp.float64

    #
    #
    #   Low-level helpers

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

    def _apply_shift_by_comps(
        self,
        sigma: jnp.ndarray,
        comps: jnp.ndarray,
        shifts: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply ±1 shifts to selected component indices for the entire batch. This realises
        h_γ |σ⟩ = |σ + shift(γ)⟩ for any γ (segment or loop).

        If K == 0 (degenerate case), return sigma unchanged.

        :param sigma: (B, N, D) batch of configurations
        :param comps: (K,) indices to update (can be K=0)
        :param shifts: (K,) ±1 shifts corresponding to comps
        """

        # fast path: nothing to shift
        if comps.size == 0:
            return sigma

        # unpack batch shape, D is the full number of dofs
        B, N, D = sigma.shape

        # flatten the (B,N) axes for vectorised gather/scatter
        sig = sigma.reshape(-1, D)

        # gather selected components across the batch
        sel = sig[:, comps]

        # broadcast shifts to match (batch_size, K), cast to int type
        delta = jnp.broadcast_to(shifts[None, :], sel.shape).astype(sig.dtype)

        # apply modular wrapping after shifting
        # new_sel = self._wrap_mod(sel, delta, self.cutoff, self._mod_span)
        # dev: fix the scattering shapes, sel and delta potentially int8, rest is float64
        #      will give errors in new JAX + silent overflows
        new_sel = self._wrap_mod(
            sel, delta, mod_span=self._mod_span, state_min=self._state_min
        )

        # scatter updated values back into the flattened array
        sig = sig.at[:, comps].set(new_sel)

        # restore original (B, N, D) shape
        return sig.reshape(B, N, D)

    def _volume_batch(self, sigma: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the vertex volume V_v(σ) for a batch of configurations:

            V_v(σ) = sqrt( | Σ_triplets ε_or * m_{e1}(σ) · ( m_{e2}(σ) x m_{e3}(σ) ) | ).

        Here m_{ei}(σ) ∈ ℤ^3 are the U(1)^3 charges read from σ at the 3 component indices
        of edge e_i (unchanged from the product-action implementation).
        """

        # read shapes and flatten the batch (B*N, D) for vectorised indexing
        B, N, D = sigma.shape
        sig = sigma.reshape(B * N, D)

        # gather 3-vectors (m_{e1}, m_{e2}, m_{e3}) per triplet across the flattened batch
        m1 = jnp.take(sig, self._comps1, axis=-1).astype(self._fdtype)
        m2 = jnp.take(sig, self._comps2, axis=-1).astype(self._fdtype)
        m3 = jnp.take(sig, self._comps3, axis=-1).astype(self._fdtype)

        # compute cross product m_{e2} x m_{e3} and dot with m_{e1}
        cr = jnp.cross(m2, m3, axis=-1)
        dots = jnp.einsum("mtc,mtc->mt", m1, cr)

        # sum over triplets with their oriented sign ε_or, take sqrt(|.|), and reshape to (B, N)
        Vr = jnp.sum(dots * self._signs[None, :], axis=1)
        V = jnp.sqrt(jnp.abs(Vr)).reshape(B, N)

        return V

    #
    #
    #   Main kernel

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        # if the input is a single chain (N,D), temporarily promote to (B=1, N, D) for uniformity
        single_input = σ.ndim == 2
        if single_input:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        # early exit: if vertex is inadmissible or no terms, return an identity-padded stub
        if (not self._is_admissible) or (self._n_terms == 0):
            σp = σ[:, :, None, :]
            mels = jnp.zeros((B, N, 1), dtype=self.dtype)
            if single_input:
                σp, mels = σp[0], mels[0]
            return σp, mels

        # precompute the base volume V_v(σ) once (used to form every ΔV_t^(k))
        # shape (B, N)
        v0 = self._volume_batch(σ)

        # we now emit exactly 6 connections per term: k=0,1,2 and (h, h^†)
        # 2 × n_terms × 3
        n_conn = 2 * self._n_terms * self.gauge_dim

        # preallocate output buffers with static shapes for jax.jit stability
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, n_conn, D)).copy()
        mels = jnp.zeros((B, N, n_conn), dtype=self.dtype)

        # running index into the 3rd dimension of σp/mels where we place connections
        idx = 0

        # fixed-length outer loop over terms, inner fixed-length loop over copies k=0...2
        for t in range(self._n_terms):

            # per-term constant factor inside the sum: coeff_t = reord_t * (ε/L)_t
            coeff_t = (self._reord[t] * self._eps_over_L[t]).astype(jnp.float64)

            # get the 3 segment indices (all copies) and the per-copy loop buffers for this term
            # shape (3,)
            seg_comps_all = self._seg_comps_all[t]
            loop_comps_k = self._loop_comps_k
            loop_shifts_k = self._loop_shifts_k

            # sum over the three U(1) copies, k = 0,1,2
            for k in range(self.gauge_dim):

                #
                #
                #   build ΔV_t^(k)(σ) = V(σ) - V(h_{s_t}^{(k)-1} σ)

                # select the single segment component for copy k (shape (1,) to keep vectorised api)
                # slice gives (1,) int32
                seg_comp_k = seg_comps_all[k : k + 1]

                # shift "down" only the k-th copy on that segment (dagger on h_s)
                seg_shift_k = jnp.array([-1], dtype=σ.dtype)

                # apply the copy-resolved segment shift to build h_{s_t}^{(k)-1} σ
                σ_s_inv_k = self._apply_shift_by_comps(σ, seg_comp_k, seg_shift_k)

                # evaluate volume on the shifted batch
                vSi_k = self._volume_batch(σ_s_inv_k)

                # form the volume difference for this (t,k)
                deltaV_k = (v0 - vSi_k).astype(jnp.float64)

                #
                #
                #   build the two final states: h_{α_t}^{(k)}|σ⟩ and h_{α_t}^{(k)†}|σ⟩

                # get the copy-k loop indices and shifts (may be empty for degenerate loops)
                # shape (M_tk,)
                loop_comps_tk = loop_comps_k[k][t]
                # shape (M_tk,)
                loop_shifts_tk = loop_shifts_k[k][t]

                if loop_comps_tk.size > 0:
                    # apply +shift for h_{α}^{(k)} and -shift for h_{α}^{(k)†}
                    σ_alpha_k = self._apply_shift_by_comps(
                        σ, loop_comps_tk, loop_shifts_tk.astype(σ.dtype)
                    )

                    σ_adag_k = self._apply_shift_by_comps(
                        σ, loop_comps_tk, (-loop_shifts_tk).astype(σ.dtype)
                    )

                else:
                    # degenerate loop: σ is unchanged by loop holonomy
                    σ_alpha_k = σ
                    σ_adag_k = σ

                #
                #
                #   matrix elements for the two branches:
                #       <h_{α}^{(k)}σ|H_v|σ> = (+coeff_t) * ΔV_t^(k)(σ)
                #       <h_{α}^{(k)†}σ|H_v|σ> = (-coeff_t) * ΔV_t^(k)(σ)
                mel_alpha_k = (+coeff_t) * deltaV_k
                mel_adag_k = (-coeff_t) * deltaV_k

                # write both connections into the preallocated output buffers
                σp = σp.at[:, :, idx, :].set(σ_alpha_k)
                mels = mels.at[:, :, idx].set(mel_alpha_k.astype(self.dtype))
                idx += 1

                σp = σp.at[:, :, idx, :].set(σ_adag_k)
                mels = mels.at[:, :, idx].set(mel_adag_k.astype(self.dtype))
                idx += 1

        #
        #
        #   apply the overall normalization once:
        #       H_v <- (N(v)/T_v) * [ Σ_j (ε_j/L_j) Σ_t reord_t Σ_k (...) ]  if apply_lapse
        #           or (1/T_v) * [ ... ] otherwise.
        scale = (self._Nv / self._T) if self._apply_lapse else (1.0 / self._T)
        mels = scale * mels

        # if we promoted (N,D) to (1,N,D) earlier, drop that leading dimension now
        if single_input:
            σp = σp[0]
            mels = mels[0]

        # return the connected configurations and matrix elements in NetKet padded format.
        return σp, mels


class EuclideanVertexTRCOperatorFast(ComputationalOperator):
    r"""
    This implementation is numerically identical to the EuclideanVertexTRCOperator implementation,
    except it is faster.

    The implementation is not self-documented, as it aims to mirror the well-documented
    implementation in the EuclideanVertexTRCOperator class.
    """

    def __init__(
        self,
        lqx: "lqx",
        vertex: Union[str, int],
        *,
        apply_lapse: bool = True,
    ) -> None:
        super().__init__(lqx.H.hilbert_netket)

        # short-handle to H and vertex id
        self._H = lqx.H

        self.vertex: int = int(vertex) if not isinstance(vertex, int) else vertex

        # gauge dimension must be 3 for U(1)^3
        self.gauge_dim: int = int(lqx.H.gauge_dimensions)
        assert (
            self.gauge_dim == 3
        ), f"This operator assumes gauge_dim == 3 (U(1)^3), found {self.gauge_dim} instead."

        # modular arithmetic parameters
        self.cutoff: int = int(lqx.H.cutoff)

        # a span for the modded addition
        # works for symmetric/positive spaces
        self._mod_span = int(lqx.H.allowed_basis_states.length)

        # lowest
        self._state_min = int(lqx.H.allowed_basis_states.start)

        # highest
        self._state_max = int(
            self._state_min
            + (self._mod_span - 1) * int(lqx.H.allowed_basis_states.step)
        )

        # number of edges at one copy level (D = n_edges_total * 3)
        self.n_edges_total: int = int(lqx.H.size // self.gauge_dim)

        # compile-time admissible vertex flag
        self._is_admissible: bool = self.vertex in set(lqx._admissible_vertices)

        # lapse
        self._apply_lapse: bool = bool(apply_lapse)
        if self._apply_lapse and hasattr(lqx, "lapses"):
            self._Nv: float = float(lqx.lapses[self.vertex])
        else:
            self._Nv = 1.0

        # volume geometry: oriented triplets (e1,e2,e3) and signs
        triplets = lqx.H.graph.handler.graph_edges_data["graph"]["connectivities"][
            str(self.vertex)
        ]["edges"]
        signs_dict = lqx.H.graph.signs[str(self.vertex)]

        comps1: List[jnp.ndarray] = []
        comps2: List[jnp.ndarray] = []
        comps3: List[jnp.ndarray] = []
        signs: List[int] = []

        # offsets to reach copies: [0, nE, 2*nE]
        offs = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        for triple in triplets:
            e1, e2, e3 = map(lqx.H.graph.edge_to_index, triple)
            comps1.append((jnp.asarray(e1, dtype=jnp.int32) + offs).astype(jnp.int32))
            comps2.append((jnp.asarray(e2, dtype=jnp.int32) + offs).astype(jnp.int32))
            comps3.append((jnp.asarray(e3, dtype=jnp.int32) + offs).astype(jnp.int32))
            signs.append(int(signs_dict[get_true_edge_triplet(triple, lqx.H.graph)]))

        # static arrays for volume

        # shapes (T_v, 3)
        self._comps1 = jnp.stack(comps1, axis=0).astype(jnp.int32)
        self._comps2 = jnp.stack(comps2, axis=0).astype(jnp.int32)
        self._comps3 = jnp.stack(comps3, axis=0).astype(jnp.int32)

        # shape (T_v,)
        self._signs = jnp.asarray(signs, dtype=jnp.int32)

        # graph-level terms (level 0)
        level_terms = lqx._constraint_terms["0"][str(self.vertex)]
        self._T: float = float(level_terms["T"])

        # per-term structures
        reord_list: List[float] = []
        eps_over_L_list: List[float] = []

        # each (3,)
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
                seg_eidx = int(lqx.H.graph.edge_to_index(seg_edge))
                seg_comps_all = (jnp.asarray(seg_eidx, dtype=jnp.int32) + offs).astype(
                    jnp.int32
                )

                edges_flat_all: List[jnp.ndarray] = []
                shifts_flat_all: List[jnp.ndarray] = []

                for edge, data in term.dressed_minimal_loop:
                    eidx = int(lqx.H.graph.edge_to_index(edge))
                    comps_all = (jnp.asarray(eidx, dtype=jnp.int32) + offs).astype(
                        jnp.int32
                    )
                    s = +1 if data["type"] == "creation" else -1
                    edges_flat_all.append(comps_all)
                    shifts_flat_all.append(
                        jnp.full((self.gauge_dim,), s, dtype=jnp.int32)
                    )

                if edges_flat_all:
                    loop_comps_all = jnp.concatenate(edges_flat_all, axis=0)
                    loop_shifts_all = jnp.concatenate(shifts_flat_all, axis=0)
                else:
                    loop_comps_all = jnp.zeros((0,), dtype=jnp.int32)
                    loop_shifts_all = jnp.zeros((0,), dtype=jnp.int32)

                # split into copies
                for k in range(self.gauge_dim):
                    loop_comps_k = loop_comps_all[k :: self.gauge_dim]
                    loop_shifts_k = loop_shifts_all[k :: self.gauge_dim]
                    loop_comps_k_lists[k].append(loop_comps_k)
                    loop_shifts_k_lists[k].append(loop_shifts_k)

                seg_comp_all_list.append(seg_comps_all)
                reord_list.append(reord)
                eps_over_L_list.append(eps_over_L)

        # final per-term arrays
        self._n_terms: int = len(reord_list)

        # shapes (n_terms,)
        self._reord = jnp.asarray(reord_list, dtype=jnp.float64)
        self._eps_over_L = jnp.asarray(eps_over_L_list, dtype=jnp.float64)
        self._coeff = (self._reord * self._eps_over_L).astype(jnp.float64)

        # segment indices per copy: (3, n_terms)
        seg_k = []
        for k in range(self.gauge_dim):
            seg_k.append(jnp.stack([s[k] for s in seg_comp_all_list], axis=0))
        self._seg_comp_k = jnp.stack(seg_k, axis=0).astype(jnp.int32)

        # loop data per copy, first pad to width M_k, then unify across copies to M_max
        loop_comps_k_padded: List[jnp.ndarray] = []
        loop_shifts_k_padded: List[jnp.ndarray] = []
        Mk_list: List[int] = []

        for k in range(self.gauge_dim):
            lengths = [int(arr.size) for arr in loop_comps_k_lists[k]]
            M_k = int(max(lengths) if lengths else 0)
            Mk_list.append(M_k)
            if M_k == 0:
                loop_comps_k_padded.append(
                    jnp.zeros((self._n_terms, 0), dtype=jnp.int32)
                )
                loop_shifts_k_padded.append(
                    jnp.zeros((self._n_terms, 0), dtype=jnp.int32)
                )
            else:
                comps_pad = jnp.stack(
                    [
                        jnp.pad(a, (0, M_k - a.size), constant_values=-1)
                        for a in loop_comps_k_lists[k]
                    ],
                    axis=0,
                ).astype(jnp.int32)
                shifts_pad = jnp.stack(
                    [
                        jnp.pad(
                            s.astype(jnp.int32), (0, M_k - s.size), constant_values=0
                        )
                        for s in loop_shifts_k_lists[k]
                    ],
                    axis=0,
                ).astype(jnp.int32)
                loop_comps_k_padded.append(comps_pad)
                loop_shifts_k_padded.append(shifts_pad)

        M_max = int(max(Mk_list) if Mk_list else 0)

        if M_max == 0:
            self._loop_comps = jnp.zeros(
                (self.gauge_dim, self._n_terms, 0), dtype=jnp.int32
            )
            self._loop_shifts = jnp.zeros(
                (self.gauge_dim, self._n_terms, 0), dtype=jnp.int32
            )
        else:
            comps_3d = []
            shifts_3d = []
            for k in range(self.gauge_dim):
                M_k = loop_comps_k_padded[k].shape[1]
                if M_k < M_max:
                    pad_w = (0, 0), (0, M_max - M_k)
                    comps_3d.append(
                        jnp.pad(loop_comps_k_padded[k], pad_w, constant_values=-1)
                    )
                    shifts_3d.append(
                        jnp.pad(loop_shifts_k_padded[k], pad_w, constant_values=0)
                    )
                else:
                    comps_3d.append(loop_comps_k_padded[k])
                    shifts_3d.append(loop_shifts_k_padded[k])
            self._loop_comps = jnp.stack(comps_3d, axis=0).astype(jnp.int32)
            self._loop_shifts = jnp.stack(shifts_3d, axis=0).astype(jnp.int32)

        # Dtypes
        # matrix elements and volume computations
        self._fdtype: DType = jnp.float64
        # indices/charges
        self._idtype = jnp.int64

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def adjoint(self) -> "EuclideanVertexTRCAdjointOperatorFast":
        """Return the adjoint operator."""
        obj = EuclideanVertexTRCAdjointOperatorFast.__new__(
            EuclideanVertexTRCAdjointOperatorFast
        )
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def dtype(self) -> DType:
        return self._fdtype

    @staticmethod
    def _wrap_mod(
        vals: jnp.ndarray, delta: jnp.ndarray, mod_span: int, state_min: int
    ) -> jnp.ndarray:
        return ((vals + delta - state_min) % mod_span) + state_min

    def _volume_batch(self, sigma: jnp.ndarray) -> jnp.ndarray:
        """Compute V_v(σ) for a batch sigma of shape (B, N, D)."""
        B, N, D = sigma.shape

        sig = sigma.reshape(B * N, D)

        m1 = jnp.take(sig, self._comps1, axis=-1).astype(self._fdtype)
        m2 = jnp.take(sig, self._comps2, axis=-1).astype(self._fdtype)
        m3 = jnp.take(sig, self._comps3, axis=-1).astype(self._fdtype)

        cr = jnp.cross(m2, m3, axis=-1)
        dots = jnp.einsum("btc,btc->bt", m1, cr)

        Vr = jnp.sum(dots * self._signs[None, :], axis=1)
        V = jnp.sqrt(jnp.abs(Vr)).reshape(B, N)

        return V

    def _shift_one_component(
        self, sigma: jax.Array, comp_idx: jax.Array, delta: int
    ) -> jax.Array:
        """
        Return sigma with one component (scalar index) shifted by delta (with modular wrap).
        `comp_idx` can be a traced int32 scalar.
        """

        comp_idx = jnp.asarray(comp_idx, dtype=self._idtype)
        sel = jnp.take(sigma, comp_idx, axis=-1)

        new_sel = self._wrap_mod(
            sel, jnp.asarray(delta, dtype=self._idtype), self._mod_span, self._state_min
        )

        return sigma.at[..., comp_idx].set(new_sel)

    def _apply_loop_row(
        self, sigma: jax.Array, comps_row: jax.Array, shifts_row: jax.Array, mult: int
    ) -> jax.Array:
        """
        Apply a row of loop updates (M entries) onto sigma, in-order. comps_row has -1 padding.
        """

        M = comps_row.shape[0]

        def body(m, cur):
            idx = comps_row[m]
            sh = shifts_row[m] * mult
            return lax.cond(
                idx >= 0,
                lambda s: self._shift_one_component(s, idx, sh),
                lambda s: s,
                cur,
            )

        return lax.fori_loop(0, M, body, sigma)

    @partial(jax.jit, static_argnums=0)
    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        # promote (N,D) -> (1,N,D)
        single_input = σ.ndim == 2
        if single_input:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        # early exit
        if (not self._is_admissible) or (self._n_terms == 0):
            σp = σ[:, :, None, :]
            mels = jnp.zeros((B, N, 1), dtype=self._fdtype)
            return (σp[0], mels[0]) if single_input else (σp, mels)

        # base volume
        v0 = self._volume_batch(σ)

        # compute deltaV for all (k,t)
        deltaV = jnp.zeros((self.gauge_dim, self._n_terms, B, N), dtype=self._fdtype)

        def body_delta(i, dv):
            k = i // self._n_terms
            t = i - k * self._n_terms
            comp_idx = self._seg_comp_k[k, t]
            sig_shift = self._shift_one_component(σ, comp_idx, -1)
            vSi = self._volume_batch(sig_shift)
            dv = dv.at[k, t].set((v0 - vSi).astype(self._fdtype))
            return dv

        deltaV = lax.fori_loop(0, self.gauge_dim * self._n_terms, body_delta, deltaV)

        # prepare output buffers for states
        sigma_alpha = jnp.zeros((self.gauge_dim, self._n_terms, B, N, D), dtype=σ.dtype)
        sigma_adag = jnp.zeros((self.gauge_dim, self._n_terms, B, N, D), dtype=σ.dtype)

        def body_states(i, carry):
            sa, sd = carry
            k = i // self._n_terms
            t = i - k * self._n_terms
            comps_row = self._loop_comps[k, t, :]
            shifts_row = self._loop_shifts[k, t, :]
            sig_plus = self._apply_loop_row(σ, comps_row, shifts_row, +1)
            sig_minus = self._apply_loop_row(σ, comps_row, shifts_row, -1)
            sa = sa.at[k, t].set(sig_plus)
            sd = sd.at[k, t].set(sig_minus)
            return (sa, sd)

        sigma_alpha, sigma_adag = lax.fori_loop(
            0, self.gauge_dim * self._n_terms, body_states, (sigma_alpha, sigma_adag)
        )

        # matrix elements for alpha/adag
        coeff = self._coeff[None, :, None, None].astype(self._fdtype)
        mels_alpha = +coeff * deltaV
        mels_adag = -coeff * deltaV

        # pack connections: interleave alpha, adag per (k,t)
        n_conn = 2 * self.gauge_dim * self._n_terms
        σp = jnp.empty((B, N, n_conn, D), dtype=σ.dtype)
        mels = jnp.empty((B, N, n_conn), dtype=self._fdtype)

        def write_conn(i, carry):
            Xp, Ml = carry
            k = i // self._n_terms
            t = i - k * self._n_terms
            base = 2 * i
            Xp = Xp.at[:, :, base, :].set(sigma_alpha[k, t])
            Xp = Xp.at[:, :, base + 1, :].set(sigma_adag[k, t])
            Ml = Ml.at[:, :, base].set(mels_alpha[k, t])
            Ml = Ml.at[:, :, base + 1].set(mels_adag[k, t])
            return (Xp, Ml)

        σp, mels = lax.fori_loop(
            0, self.gauge_dim * self._n_terms, write_conn, (σp, mels)
        )

        # global scale
        scale = (self._Nv / self._T) if self._apply_lapse else (1.0 / self._T)
        mels = (scale * mels).astype(self._fdtype)

        if single_input:
            return σp[0], mels[0]
        return σp, mels


#
#
# adjoint implementations


class EuclideanVertexTRCAdjointOperator(EuclideanVertexTRCOperator):
    r"""
    Adjoint of EuclideanVertexTRCOperator.

    Core action (per term t, copy k):
        (V_v - h_{s_t}^{(k)} V_v h_{s_t}^{(k)\dagger}) (h_{α_t}^{(k)\dagger} - h_{α_t}^{(k)})

    Computationally:
      For each branch σ' ∈ { h_{α}^{(k)}σ, h_{α}^{(k)†}σ }:
        ΔV_adj(σ') = V(σ') - V(h_{s}^{(k)-1} σ')
      mels:
        for σ' = h_{α}σ  ->  - coeff_t * ΔV_adj(σ')
        for σ' = h_{α}†σ ->  + coeff_t * ΔV_adj(σ')
    """

    @property
    def adjoint(self) -> "EuclideanVertexTRCOperator":
        """Return the non-adjoint operator."""
        obj = EuclideanVertexTRCOperator.__new__(EuclideanVertexTRCOperator)
        obj.__dict__.update(self.__dict__)
        return obj

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):
        single_input = σ.ndim == 2
        if single_input:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        if (not self._is_admissible) or (self._n_terms == 0):
            σp = σ[:, :, None, :]
            mels = jnp.zeros((B, N, 1), dtype=self.dtype)
            if single_input:
                σp, mels = σp[0], mels[0]
            return σp, mels

        # preallocate outputs (same connection topology as original: interleave α, α†)
        n_conn = 2 * self._n_terms * self.gauge_dim
        σp = jnp.broadcast_to(σ[:, :, None, :], (B, N, n_conn, D)).copy()
        mels = jnp.zeros((B, N, n_conn), dtype=self.dtype)

        idx = 0
        for t in range(self._n_terms):
            coeff_t = (self._reord[t] * self._eps_over_L[t]).astype(jnp.float64)

            seg_comps_all = self._seg_comps_all[t]
            loop_comps_k = self._loop_comps_k
            loop_shifts_k = self._loop_shifts_k

            for k in range(self.gauge_dim):
                # build the two branch states first: σ_α, σ_α†
                loop_comps_tk = loop_comps_k[k][t]
                loop_shifts_tk = loop_shifts_k[k][t]

                if loop_comps_tk.size > 0:
                    σ_alpha_k = self._apply_shift_by_comps(
                        σ, loop_comps_tk, loop_shifts_tk.astype(σ.dtype)
                    )
                    σ_adag_k = self._apply_shift_by_comps(
                        σ, loop_comps_tk, (-loop_shifts_tk).astype(σ.dtype)
                    )
                else:
                    σ_alpha_k = σ
                    σ_adag_k = σ

                # for the adjoint, ΔV is evaluated ON EACH BRANCH STATE:
                seg_comp_k = seg_comps_all[k : k + 1]
                seg_shift_inv = jnp.array([-1], dtype=σ.dtype)

                # α branch
                v_alpha = self._volume_batch(σ_alpha_k)
                σ_alpha_sinv = self._apply_shift_by_comps(
                    σ_alpha_k, seg_comp_k, seg_shift_inv
                )
                v_alpha_sinv = self._volume_batch(σ_alpha_sinv)
                deltaV_alpha = (v_alpha - v_alpha_sinv).astype(jnp.float64)

                # α† branch
                v_adag = self._volume_batch(σ_adag_k)
                σ_adag_sinv = self._apply_shift_by_comps(
                    σ_adag_k, seg_comp_k, seg_shift_inv
                )
                v_adag_sinv = self._volume_batch(σ_adag_sinv)
                deltaV_adag = (v_adag - v_adag_sinv).astype(jnp.float64)

                # signs flipped relative to original
                mel_alpha_k = (-coeff_t) * deltaV_alpha  # for |h_α σ>
                mel_adag_k = (+coeff_t) * deltaV_adag  # for |h_α† σ>

                σp = σp.at[:, :, idx, :].set(σ_alpha_k)
                mels = mels.at[:, :, idx].set(mel_alpha_k.astype(self.dtype))
                idx += 1

                σp = σp.at[:, :, idx, :].set(σ_adag_k)
                mels = mels.at[:, :, idx].set(mel_adag_k.astype(self.dtype))
                idx += 1

        scale = (self._Nv / self._T) if self._apply_lapse else (1.0 / self._T)
        mels = scale * mels

        if single_input:
            σp = σp[0]
            mels = mels[0]
        return σp, mels


class EuclideanVertexTRCAdjointOperatorFast(EuclideanVertexTRCOperatorFast):
    r"""
    Fast adjoint version of EuclideanVertexTRCOperatorFast.
    """

    @property
    def adjoint(self) -> "EuclideanVertexTRCOperatorFast":
        """Return the non-adjoint operator."""
        obj = EuclideanVertexTRCOperatorFast.__new__(EuclideanVertexTRCOperatorFast)
        obj.__dict__.update(self.__dict__)
        return obj

    @partial(jax.jit, static_argnums=0)
    def _get_conn_padded_kernel(self, σ: jnp.ndarray):
        single_input = σ.ndim == 2
        if single_input:
            σ = σ[None, :, :]
        B, N, D = σ.shape

        if (not self._is_admissible) or (self._n_terms == 0):
            σp = σ[:, :, None, :]
            mels = jnp.zeros((B, N, 1), dtype=self._fdtype)
            return (σp[0], mels[0]) if single_input else (σp, mels)

        # build branch states for all (k,t): σ_α and σ_α† (same as original)
        sigma_alpha = jnp.zeros((self.gauge_dim, self._n_terms, B, N, D), dtype=σ.dtype)
        sigma_adag = jnp.zeros((self.gauge_dim, self._n_terms, B, N, D), dtype=σ.dtype)

        def body_states(i, carry):
            sa, sd = carry
            k = i // self._n_terms
            t = i - k * self._n_terms
            comps_row = self._loop_comps[k, t, :]
            shifts_row = self._loop_shifts[k, t, :]
            sig_plus = self._apply_loop_row(σ, comps_row, shifts_row, +1)
            sig_minus = self._apply_loop_row(σ, comps_row, shifts_row, -1)
            sa = sa.at[k, t].set(sig_plus)
            sd = sd.at[k, t].set(sig_minus)
            return (sa, sd)

        sigma_alpha, sigma_adag = lax.fori_loop(
            0, self.gauge_dim * self._n_terms, body_states, (sigma_alpha, sigma_adag)
        )

        # for the adjoint, ΔV is computed on each branch state
        deltaV_alpha = jnp.zeros(
            (self.gauge_dim, self._n_terms, B, N), dtype=self._fdtype
        )
        deltaV_adag = jnp.zeros(
            (self.gauge_dim, self._n_terms, B, N), dtype=self._fdtype
        )

        def body_delta(i, carry):
            dA, dD = carry
            k = i // self._n_terms
            t = i - k * self._n_terms

            comp_idx = self._seg_comp_k[k, t]

            # α branch
            sigA = sigma_alpha[k, t]
            vA = self._volume_batch(sigA)
            sigA_sinv = self._shift_one_component(sigA, comp_idx, -1)
            vA_sinv = self._volume_batch(sigA_sinv)
            dA = dA.at[k, t].set((vA - vA_sinv).astype(self._fdtype))

            # α† branch
            sigD = sigma_adag[k, t]
            vD = self._volume_batch(sigD)
            sigD_sinv = self._shift_one_component(sigD, comp_idx, -1)
            vD_sinv = self._volume_batch(sigD_sinv)
            dD = dD.at[k, t].set((vD - vD_sinv).astype(self._fdtype))

            return (dA, dD)

        deltaV_alpha, deltaV_adag = lax.fori_loop(
            0, self.gauge_dim * self._n_terms, body_delta, (deltaV_alpha, deltaV_adag)
        )

        # 3) Signs per branch (adjoint)
        coeff = self._coeff[None, :, None, None].astype(self._fdtype)
        mels_alpha = -coeff * deltaV_alpha
        mels_adag = +coeff * deltaV_adag

        # 4) Pack connections (α, α†) interleaved
        n_conn = 2 * self.gauge_dim * self._n_terms
        σp = jnp.empty((B, N, n_conn, D), dtype=σ.dtype)
        mels = jnp.empty((B, N, n_conn), dtype=self._fdtype)

        def write_conn(i, carry):
            Xp, Ml = carry
            k = i // self._n_terms
            t = i - k * self._n_terms
            base = 2 * i
            Xp = Xp.at[:, :, base, :].set(sigma_alpha[k, t])
            Xp = Xp.at[:, :, base + 1, :].set(sigma_adag[k, t])
            Ml = Ml.at[:, :, base].set(mels_alpha[k, t])
            Ml = Ml.at[:, :, base + 1].set(mels_adag[k, t])
            return (Xp, Ml)

        σp, mels = lax.fori_loop(
            0, self.gauge_dim * self._n_terms, write_conn, (σp, mels)
        )

        scale = (self._Nv / self._T) if self._apply_lapse else (1.0 / self._T)
        mels = (scale * mels).astype(self._fdtype)

        if single_input:
            return σp[0], mels[0]
        return σp, mels


#
#
#   dispatchers


def ThiemannRegularisedVertexConstraint(
    lqx: "lqx",
    vertex: Union[str, int],
    *,
    apply_lapse: bool = True,
    adjoint: bool = False,
):
    if adjoint:
        return EuclideanVertexTRCAdjointOperator(
            lqx=lqx, vertex=vertex, apply_lapse=apply_lapse
        )
    return EuclideanVertexTRCOperator(lqx=lqx, vertex=vertex, apply_lapse=apply_lapse)


def ThiemannRegularisedVertexConstraintFast(
    lqx: "lqx",
    vertex: Union[str, int],
    *,
    apply_lapse: bool = True,
    adjoint: bool = False,
):
    if adjoint:
        return EuclideanVertexTRCAdjointOperatorFast(
            lqx=lqx, vertex=vertex, apply_lapse=apply_lapse
        )
    return EuclideanVertexTRCOperatorFast(
        lqx=lqx, vertex=vertex, apply_lapse=apply_lapse
    )
