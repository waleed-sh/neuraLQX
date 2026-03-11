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


from __future__ import annotations

from dataclasses import field

from typing import Dict
from typing import List
from typing import Tuple
from typing import Any
from typing import Optional
from typing import Union

import jax.numpy as jnp
import flax.linen as nn

from neuralqx.graph import Graph
from neuralqx.hilbert import AbstractHilbertInterface

Edge = Tuple[int, int, int]
"""(u, v, key) oriented edge type."""

LoopElem = Tuple[Edge, Dict[str, Any]]
"""((u, v, key), {'type': 'creation'|'annihilation', ...}) type for loop elements."""


#
#
#   helpers


def _softplus_pos(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically-szable positive parameterisation via softplus."""

    # the epsilon avoids exact zeros, which helps downstream divisions or multiplicative scalings
    # remain well-defined
    return nn.softplus(x) + 1e-8


def _frequencies(
    n_f: int,
    cutoff: int,
    mode: str = "harmonic",
    base: float = 2.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Construct the frequency multipliers used in the loop-Fourier features.

    The loop charges live in Z_M, thus we use periodic features cos(ω, q) and sin(ω, q) and expose
    the spacing of ω (harmonic or logarithmic) as a modeling choice.

    :param n_f: number of Fourier harmonics to include
    :param cutoff: charge cutoff
    :param mode: the type of frequencies to generate, can be:
        - 'harmonic': frequencies are 1, 2, 3, ..., n_f (equally spaced)
        - 'log': frequences are base**k, for k = 0, ..., n_f - 1 (log-spaced)
    :param base:  log base if `mode == 'log'`
    :param dtype: output dtype of the frequency vector

    :return: a JAX array of the angular frequencies ω_j = (2π / M) * multiplier_j of shape (n_f,)
    """

    # the order of the cyclic group for loop charges
    M = float(2 * cutoff + 1)

    if mode == "harmonic":

        # choose equally spaced integer multiples
        multipliers = jnp.arange(1, n_f + 1, dtype=dtype)
    elif mode == "log":

        # choose logarithmically spaced powers of base
        ks = jnp.arange(n_f, dtype=dtype)
        multipliers = jnp.power(jnp.array(base, dtype=dtype), ks)
    else:

        # guard against typos
        raise ValueError(f"Unknown freq_mode: {mode!r}")

    # convert to angular frequencies
    return (2.0 * jnp.pi / M) * multipliers


def _lookup_edge_index(edge_key: Edge, edge_to_index: Dict[Any, int]) -> Optional[int]:
    """
    Helper function to resolve a flat edge index for a given edge key, allowing small format
    variations.

    :param edge_key: the canonical edge description (u, v, key) at the 0-th gauge level
    :param edge_to_index: user-supplied mapping from edges to indices (provided from the Graph API)

    :return: the resolved index if found, else None
    """

    if edge_key in edge_to_index:
        # exact tuple match
        return int(edge_to_index[edge_key])

    rev = (edge_key[1], edge_key[0], edge_key[2])
    if rev in edge_to_index:
        # reversed orientation
        return int(edge_to_index[rev])

    s = str(edge_key)
    if s in edge_to_index:
        # stringified
        return int(edge_to_index[s])

    s_rev = str(rev)
    if s_rev in edge_to_index:
        # reversed stringified
        return int(edge_to_index[s_rev])

    # welp, couldnt find anything...
    return None


def _edge_index_for_copy(
    base_edge: Edge, edge_to_index: Dict[Any, int], n_edges: int, k: int
) -> int:
    """
    Resolve the flat sigma index for a given base edge at a particular gauge level.

    :param base_edge: edge (u, v, key) at gauge level 0
    :param edge_to_index: mapping from edges to flat indices
    :param n_edges: number of oriented edges per gauge level
    :param k: gauge level index (0...K-1)

    :return: flat index into sigma for the edge (u + k*n_edges, v + k*n_edges, key) if present,
      otherwise falls back to base_index + k*n_edges
    """

    # base edge coordinates and key
    u, v, key = base_edge

    # shifted edge for level k
    shifted = (u + k * n_edges, v + k * n_edges, key)

    # try to resolve shifted edge directly
    idx = _lookup_edge_index(shifted, edge_to_index)

    if idx is not None:
        # present, so use it
        return idx

    # otherwise try base at level 0
    base_idx = _lookup_edge_index((u, v, key), edge_to_index)

    if base_idx is not None:
        # if known, stride by n_edges to get copy k
        return int(base_idx) + int(k) * int(n_edges)

    # nothing matched: we cannot resolve the flat index
    raise KeyError(
        f"Edge {shifted} not found, and base {(u, v, key)} not in edge_to_index. "
        "Provide either all copies in edge_to_index OR at least copy-0 with contiguous-per-copy layout."
    )


#
#
#   const builders


def _build_loop_const(
    loops_edges_typed: Optional[List[List[LoopElem]]],
    loops_indices: Optional[List[List[int]]],
    edge_to_index: Dict[Any, int],
    gauge_dim: int,
    n_edges: int,
):
    """
    Precompute padded loop tensors for fast just-in-time gathering.

    This converts Python lists of variable-length loops into fixed-size, padded arrays. The forward
    pass then uses a single take_along_axis gather per batch to compute loop charges efficiently as:
    sum(mask * sign * sigma[idx]).

    :param loops_edges_typed: minimal loops given as a list of loops, each loop is a list of
      (edge, meta) where meta['type'] in {'creation', 'annihilation'} sets the sign
    :param loops_indices: alternate representation: minimal loops given as flat indices at level 0.
      If provided, copy-k indices are computed by adding k*n_edges
    :param edge_to_index: a mapping from edges to flat sigma indices
    :param gauge_dim: number of gauge levels (K)
    :param n_edges: number of oriented edges per gauge lavel

    :return: a tuple (idxs, signs, mask, L) where
        - idxs is an int32 JAX array of padded indices of shape (K, L, Smax)
        - signs is an int8 JAX array of aligned signs (+1 or -1) of shape (K, L, Smax)
        - mask is a float32 JAX zeroes array (but with 1.0 for valid entries) of shape (K, L, Smax)
        - L is an int representing the number of minimal loops per gauge level
    """

    if gauge_dim == 0:
        # degenerate case, no levels (K = 0)
        z = jnp.zeros((0, 0, 0), dtype=jnp.int32)
        return z, z.astype(jnp.int8), z.astype(jnp.float32), 0

    # prepare Python lists for each level before padding to a common length

    # indices per (level, loop)
    per_copy_idxs: List[List[List[int]]] = [[] for _ in range(gauge_dim)]

    # signs per (level, loop)
    per_copy_sgns: List[List[List[int]]] = [[] for _ in range(gauge_dim)]

    if loops_edges_typed is not None and len(loops_edges_typed) > 0:
        # prefer typed loops if given
        for loop in loops_edges_typed:

            # to collect the level-0 edges and signs from `type` metadata
            base_edges: List[Edge] = []
            base_sgns: List[int] = []
            for edge, meta in loop:
                sgn = +1 if str(meta.get("type", "creation")) == "creation" else -1
                base_edges.append(edge)
                base_sgns.append(sgn)

            # we now replicate for each gauge level
            for k in range(gauge_dim):
                idxs_k: List[int] = []
                sgns_k: List[int] = []

                for be, s in zip(base_edges, base_sgns):
                    # resolve each edge for copy k
                    flat = _edge_index_for_copy(be, edge_to_index, n_edges, k)
                    idxs_k.append(flat)
                    sgns_k.append(s)

                per_copy_idxs[k].append(idxs_k)
                per_copy_sgns[k].append(sgns_k)

    elif loops_indices is not None and len(loops_indices) > 0:

        # fallback: loops by flat indices
        # do the same business but now it is simpler as we are already at flat indices
        for loop in loops_indices:
            loop0 = [int(i) for i in loop]

            # replicate for all gauge levels
            for k in range(gauge_dim):
                idxs = [i + k * int(n_edges) for i in loop0]
                sgns = [1] * len(loop0)
                per_copy_idxs[k].append(idxs)
                per_copy_sgns[k].append(sgns)
    else:
        # no loops provided
        return (
            jnp.zeros((gauge_dim, 0, 0), dtype=jnp.int32),
            jnp.zeros((gauge_dim, 0, 0), dtype=jnp.int8),
            jnp.zeros((gauge_dim, 0, 0), dtype=jnp.float32),
            0,
        )

    # number of minimal loops per level
    L = len(per_copy_idxs[0]) if gauge_dim > 0 else 0

    # max loop length
    Smax = max((len(x) for k in range(gauge_dim) for x in per_copy_idxs[k]), default=0)

    # create padded arrays of indices, signs and the mask
    idxs_arr = jnp.zeros((gauge_dim, L, Smax), dtype=jnp.int32)
    sgns_arr = jnp.zeros((gauge_dim, L, Smax), dtype=jnp.int8)
    mask_arr = jnp.zeros((gauge_dim, L, Smax), dtype=jnp.float32)

    # loop through each gauge level
    for k in range(gauge_dim):

        # and in each level loop through each loop
        for l in range(L):

            # variable length indices and corresponding signs
            cur_idx = per_copy_idxs[k][l]
            cur_sgn = per_copy_sgns[k][l]

            if len(cur_idx) == 0:
                # empty loop, unlikely
                continue

            # pad_len = Smax - len(cur_idx)
            # write slice for valid region
            idxs_arr = idxs_arr.at[k, l, : len(cur_idx)].set(
                jnp.array(cur_idx, dtype=jnp.int32)
            )
            sgns_arr = sgns_arr.at[k, l, : len(cur_idx)].set(
                jnp.array(cur_sgn, dtype=jnp.int8)
            )
            mask_arr = mask_arr.at[k, l, : len(cur_idx)].set(1.0)

    return idxs_arr, sgns_arr, mask_arr, L


def _build_triplet_const(
    triplets_by_vertex: Optional[Dict[str, List[Tuple[Edge, Edge, Edge]]]],
    triplet_signs_by_vertex: Optional[Dict[str, Dict[Any, int]]],
    edge_to_index: Dict[Any, int],
    gauge_dim: int,
    n_edges: int,
):
    """
    Precompute triplet gather indices for all levels in one tensor. For every triplet edges
    (e1, e2, e3) provided at level-0, we construct indices for levels k = 0...K-1 by shifting
    vertices (u, v) -> (u + k*n_edges, v + k*n_edges) using the same parallel-edge key.

    :param triplets_by_vertex: mapping from vertex (as string) to a list of triplets of edges
    :param triplet_signs_by_vertex: optional per-vertex dict of signs for given triplets
    :param edge_to_index: mapping from edges to flat sigma indices
    :param gauge_dim: number of gauge levels
    :param n_edges: number of oriented edges per gauge level

    :return: a tuple (idx_mat, eps, T) where
        - idx_max is an int32 JAX array of shape (T, 3, K) for flat indices for each edge component
            over gauge levels
        - eps is an int32 JAX array of shape (T,) of triplet signs
        - T is an int representing the number of triplets
    """

    if not triplets_by_vertex:
        # no triplets provided
        return (
            jnp.zeros((0, 3, gauge_dim), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.int32),
            0,
        )

    # default empty dict if not provided
    signs_by_vertex = triplet_signs_by_vertex or {}

    # storage for (vertex, triplet, sign)
    flat: List[Tuple[int, Tuple[Edge, Edge, Edge], int]] = []

    for v_str, tris in triplets_by_vertex.items():

        # sign mapping for this vertex
        local = signs_by_vertex.get(v_str, {})

        def _sign_for(tri: Tuple[Edge, Edge, Edge]) -> int:
            """Helper to resolve sign for this triplet."""

            if tri in local:
                # tuple match
                return int(local[tri])

            # stringified tri, and return 0 if nothing found
            return int(local.get(str(tri), 0))

        for tri in tris:
            # unpack the three gauge level 0 edges
            e1, e2, e3 = tri
            # resolve orientation sign
            eps = _sign_for(tri)
            # push to flat list
            flat.append((int(v_str), (e1, e2, e3), eps))

    # number of triplets
    T = len(flat)
    if T == 0:
        # degenerate case, fast exit
        return (
            jnp.zeros((0, 3, gauge_dim), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.int32),
            0,
        )

    # fill the data into the arrays
    idx_mat = jnp.zeros((T, 3, gauge_dim), dtype=jnp.int32)
    eps_arr = jnp.zeros((T,), dtype=jnp.int32)

    for t, (_v, (e1, e2, e3), eps) in enumerate(flat):
        eps_arr = eps_arr.at[t].set(int(eps))
        for j_edge, (u, v, key) in enumerate((e1, e2, e3)):
            for k in range(gauge_dim):
                idx = _edge_index_for_copy((u, v, key), edge_to_index, n_edges, k)
                idx_mat = idx_mat.at[t, j_edge, k].set(idx)
    return idx_mat, eps_arr, T


class CFVTNet(nn.Module):
    r"""
    A fast gauge-equivariant ansatz for lattice gauge states.

    This module implements a composite ansatz for (log-) wavefunction amplitudes on a tensor-product
    Hilbert space built over K gauge levels (dimensions, e.g. for U(1)^3 this would be 3) of a base
    graph. Each level corresponds to a block of `n_edges` oriented edges, the total visible width is
    `n_edges * gauge_dimensions`.

    The architecture fuses three ingredients:

        1) Loop Fourier features
            - we extract loop charges q_{ℓ,k} by summing ±σ_e over the oriented edges e that form
              each minimal loop ℓ, for each level k
            - we then compute periodic features cos(ω_j q_{ℓ,k}), sin(ω_j q_{ℓ,k}) where ω_j are
              harmonics over the cyclic group Z_M (M = 2 * cutoff + 1)
            - trainable weights A, B (and optionally A_phase, B_phase for complex phase) linearly
              combine these features into a scalar contribution to log|ψ| (and arg ψ)

        2) Local, volume-like triplet term (optional, requires K = 3)
            - at each vertex, we may provide oriented triplets of incident edges (e1, e2, e)
            - for each triplet, we form a pseudoscalar Tv := ε · (σ_{e1} · (σ_{e2} × σ_{e3})), this
              couples the three gauge levels as a triple product
            - odd (tanh with learned scale λ) and even (quadratic penalty) parts contribute to the
              amplitude, an optional linear part contributes to the phase

        3) Bias terms (optional)
            - per-edge linear bias (RBM style, Σ_e b_e σ_e) adds directly to log|ψ|, with an
              optional phase counterpart
            - a per-loop constant bias (or per (loop, level) if levels are not tied) that shifts
              log|ψ| independently of input, useful for calibrating baselines

        Level tying:
            - if `tie_copies = True`, loop Fourier weights are shared across levels by averaging
              features over the level axis before applying (A, B, ...). Otherwise, we learn separate
              parameters per level

        Dtypes:
            - `param_dtype` sets the dtype for all learnable parameters
            - `output_dtype` chooses whether the network outputs real or complex log amplitudes. If
              complex output and `use_phase=True`, we emit `log|ψ| + i * phase`. Otherwise we emit
              a real `log|ψ|`

        Regularisation:
            - `regularization_terms()` returns a dict of L2 penalties for different parameter blocks
              (Fourier, vertex, edge bias, loop bias) controlled by l2_* scalars. You sum these into
              your loss externally.

    All index tensors (loop edges, triplet edges) and frequencies are built once and stored in the
    non-trainable "cache" collection to avoid Python loops and recompilations during training.
    """

    n_edges: int
    n_vertices: int
    gauge_dim: int
    cutoff: int
    q_min: Union[float, int]
    q_max: Union[float, int]
    q_step: Union[float, int]

    edge_to_index: Dict[Any, int] = field(repr=False, metadata={"pytree_node": False})
    loops_edges_typed: Optional[List[List[LoopElem]]] = field(
        default=None, repr=False, metadata={"pytree_node": False}
    )
    loops_indices: Optional[List[List[int]]] = field(
        default=None, repr=False, metadata={"pytree_node": False}
    )
    triplets_by_vertex: Optional[Dict[str, List[Tuple[Edge, Edge, Edge]]]] = field(
        default=None, repr=False, metadata={"pytree_node": False}
    )
    triplet_signs_by_vertex: Optional[Dict[str, Dict[Any, int]]] = field(
        default=None, repr=False, metadata={"pytree_node": False}
    )

    N_F: int = 1
    freq_mode: str = "harmonic"
    freq_log_base: float = 2.0
    tie_copies: bool = False
    use_vertex_odd: bool = True
    use_vertex_even: bool = True
    use_phase: bool = True
    use_edge_bias: bool = False
    use_loop_bias: bool = False

    param_dtype: Any = jnp.float64
    output_dtype: Any = jnp.complex128

    l2_fourier: float = 0.0
    l2_vertex: float = 0.0
    l2_edge: float = 0.0
    l2_loop: float = 0.0

    init_std_A: float = 0.1
    init_std_B: float = 0.1
    init_std_A_phase: float = 0.1
    init_std_B_phase: float = 0.1
    init_std_alpha: float = 0.1
    init_std_lambda: float = 0.1
    init_std_beta: float = 0.1
    init_std_alpha_phase: float = 0.1
    init_std_visible_bias: float = 0.4
    init_std_visible_bias_phase: float = 0.5
    init_std_loop_bias: float = 0.0

    def __hash__(self):
        return id(self)

    @nn.compact
    def __call__(self, sigma: jnp.ndarray) -> jnp.ndarray:
        B, Ntot = sigma.shape
        expected = int(self.n_edges) * int(self.gauge_dim)
        if int(Ntot) != expected:
            raise ValueError(
                f"Input has width {Ntot}, expected {expected} (= n_edges * gauge_dim)."
            )

        # cache: build once, reuse
        const = self.variable(
            "cache",
            "const",
            lambda: {
                "omega": _frequencies(
                    self.N_F,
                    self.cutoff,
                    self.freq_mode,
                    self.freq_log_base,
                    dtype=jnp.float32,
                ),
                **(
                    lambda idxs, sgns, mask, L: {
                        "loop_idxs": idxs,
                        "loop_sgns": sgns,
                        "loop_mask": mask,
                        "loop_sgnmask": (sgns.astype(jnp.float32) * mask),
                        "L": L,
                    }
                )(
                    *_build_loop_const(
                        self.loops_edges_typed,
                        self.loops_indices,
                        self.edge_to_index,
                        self.gauge_dim,
                        self.n_edges,
                    )
                ),
                **(
                    lambda idx_mat, eps, T: {
                        "trip_idx": idx_mat,
                        "trip_eps": eps,
                        "T": T,
                    }
                )(
                    *_build_triplet_const(
                        self.triplets_by_vertex,
                        self.triplet_signs_by_vertex,
                        self.edge_to_index,
                        self.gauge_dim,
                        self.n_edges,
                    )
                ),
            },
        )

        # (N_F,)
        omega: jnp.ndarray = const.value["omega"]

        # (K, L, Smax)
        loop_idxs: jnp.ndarray = const.value["loop_idxs"]

        # (K, L, Smax) float32
        loop_sgnmask: jnp.ndarray = const.value["loop_sgnmask"]
        L: int = int(loop_idxs.shape[1])

        # (T, 3, K)
        trip_idx: jnp.ndarray = const.value["trip_idx"]

        # (T,)
        trip_eps: jnp.ndarray = const.value["trip_eps"]
        T: int = int(trip_idx.shape[0])

        K = int(self.gauge_dim)
        M = 2 * int(self.cutoff) + 1

        # loops: q[b,l,k]
        if L > 0:
            # (B,K,L,S)
            idxs_exp = jnp.broadcast_to(
                loop_idxs[None, ...],
                (B, loop_idxs.shape[0], loop_idxs.shape[1], loop_idxs.shape[2]),
            )
            # (B,K,L,S)
            gathered = jnp.take_along_axis(sigma[:, None, None, :], idxs_exp, axis=3)
            # pre-multiplied sign*mask avoids an extra mul in hot path
            # (1,K,L,S)
            sgnmask = loop_sgnmask[None, ...].astype(gathered.dtype)
            q = jnp.sum(gathered * sgnmask, axis=-1).transpose(0, 2, 1)
        else:
            q = jnp.zeros((B, 0, K), dtype=sigma.dtype)

        # modular features
        # Switchable wrapping, default is physically aligned [q_min, q_max] with step q_step.
        # (B,L,K) -> float32
        q_mod = self.mod_between(q=q, dtype=jnp.float32)

        # (B,L,K,N_F)
        angles = q_mod[..., None] * omega[None, None, None, :]
        cos_feat = jnp.cos(angles)
        sin_feat = jnp.sin(angles)

        K_eff = 1 if self.tie_copies else K

        # Fuse cos/sin heads to halve passes and memory traffic
        def fuse_linear(feat_cos, feat_sin, nameA, nameB, initA, initB):
            Wc = self.param(
                nameA,
                nn.initializers.normal(initA),
                (L, K_eff, self.N_F),
                self.param_dtype,
            )
            Ws = self.param(
                nameB,
                nn.initializers.normal(initB),
                (L, K_eff, self.N_F),
                self.param_dtype,
            )
            if self.tie_copies:
                # avg over K -> (B,L,NF)
                cos_red = jnp.mean(feat_cos, axis=2)
                sin_red = jnp.mean(feat_sin, axis=2)

                # (L,NF)
                Wc_eff = Wc[:, 0, :]
                # (L,NF)
                Ws_eff = Ws[:, 0, :]
                # einsum once per branch, but half as many total
                out = jnp.einsum("bln,ln->b", cos_red, Wc_eff) + jnp.einsum(
                    "bln,ln->b", sin_red, Ws_eff
                )
            else:
                out = jnp.einsum("blkn,lkn->b", feat_cos, Wc) + jnp.einsum(
                    "blkn,lkn->b", feat_sin, Ws
                )
            return out

        if L == 0:
            fourier_amp = jnp.zeros((B,), dtype=self.param_dtype)
            fourier_phase = jnp.zeros((B,), dtype=self.param_dtype)
        else:
            fourier_amp = fuse_linear(
                cos_feat, sin_feat, "A", "B", self.init_std_A, self.init_std_B
            )
            if self.use_phase and jnp.issubdtype(
                self.output_dtype, jnp.complexfloating
            ):
                fourier_phase = fuse_linear(
                    cos_feat,
                    sin_feat,
                    "A_phase",
                    "B_phase",
                    self.init_std_A_phase,
                    self.init_std_B_phase,
                )
            else:
                fourier_phase = jnp.zeros((B,), dtype=self.param_dtype)

        # triplets (K must be 3 if present)
        triple_amp = jnp.zeros((B,), dtype=self.param_dtype)
        triple_phase = jnp.zeros((B,), dtype=self.param_dtype)
        if T > 0:
            if K != 3:
                raise ValueError("Triplet volume term requires gauge_dim == 3.")

            # (B,T,3,K)
            idx_exp = jnp.broadcast_to(trip_idx[None, ...], (B,) + trip_idx.shape)

            # (B,1,1,N)
            sig_exp = sigma[:, None, None, :]

            # (B,T,3,K)
            gathered = jnp.take_along_axis(sig_exp, idx_exp, axis=3)
            m1 = gathered[:, :, 0, :]
            m2 = gathered[:, :, 1, :]
            m3 = gathered[:, :, 2, :]

            # (B,T,3)
            cross = jnp.cross(m2, m3, axis=-1)

            # (B,T)
            dots = jnp.einsum("btk,btk->bt", m1, cross)

            # (B,T)
            Tv = dots * trip_eps[None, :].astype(dots.dtype)

            if self.use_vertex_odd:
                alpha = self.param(
                    "alpha",
                    nn.initializers.normal(self.init_std_alpha),
                    (T,),
                    self.param_dtype,
                )
                lam_raw = self.param(
                    "lambda",
                    nn.initializers.normal(self.init_std_lambda),
                    (T,),
                    self.param_dtype,
                )
                lam = _softplus_pos(lam_raw)
                triple_amp = triple_amp + jnp.sum(
                    alpha[None, :] * jnp.tanh(lam[None, :] * Tv), axis=-1
                )

            if self.use_vertex_even:
                beta_raw = self.param(
                    "beta",
                    nn.initializers.normal(self.init_std_beta),
                    (T,),
                    self.param_dtype,
                )
                beta = _softplus_pos(beta_raw)
                triple_amp = triple_amp - jnp.sum(beta[None, :] * (Tv**2), axis=-1)

            if self.use_phase and jnp.issubdtype(
                self.output_dtype, jnp.complexfloating
            ):
                alpha_p = self.param(
                    "alpha_phase",
                    nn.initializers.normal(self.init_std_alpha_phase),
                    (T,),
                    self.param_dtype,
                )
                triple_phase = triple_phase + jnp.sum(alpha_p[None, :] * Tv, axis=-1)

        # per-edge bias
        bias_amp = 0.0
        bias_phase = 0.0
        if self.use_edge_bias:
            b = self.param(
                "visible_bias",
                nn.initializers.normal(self.init_std_visible_bias),
                (self.n_edges * self.gauge_dim,),
                self.param_dtype,
            )
            bias_amp = jnp.einsum("bd,d->b", sigma, b)
            if self.use_phase and jnp.issubdtype(
                self.output_dtype, jnp.complexfloating
            ):
                b_phi = self.param(
                    "visible_bias_phase",
                    nn.initializers.normal(self.init_std_visible_bias_phase),
                    (self.n_edges * self.gauge_dim,),
                    self.param_dtype,
                )
                bias_phase = jnp.einsum("bd,d->b", sigma, b_phi)

        # per-loop bias
        loop_bias_term = 0.0
        if self.use_loop_bias and L > 0:
            if self.tie_copies:
                loop_bias = self.param(
                    "loop_bias",
                    nn.initializers.normal(self.init_std_loop_bias),
                    (L,),
                    self.param_dtype,
                )
                loop_bias_term = jnp.broadcast_to(jnp.sum(loop_bias), (B,))
            else:
                loop_bias = self.param(
                    "loop_bias",
                    nn.initializers.normal(self.init_std_loop_bias),
                    (L, K),
                    self.param_dtype,
                )
                # sum over (L,K), then broadcast to batch — equivalent to previous einsum with ones
                loop_bias_term = jnp.broadcast_to(jnp.sum(loop_bias), (B,))

        # output assembly
        logabs = (fourier_amp + triple_amp + bias_amp + loop_bias_term).astype(
            self.output_dtype
        )
        if jnp.issubdtype(self.output_dtype, jnp.complexfloating) and self.use_phase:
            phase = (fourier_phase + triple_phase + bias_phase).astype(
                self.output_dtype
            )
            out = logabs + 1j * phase
        else:
            out = logabs.real
        return out

    def mod_between(self, q, dtype):
        """Modulo-wrap q into [q_min, q_max] with step size dq."""
        M = (self.q_max - self.q_min) / self.q_step + 1
        return (self.q_min + jnp.mod((q - self.q_min), M * self.q_step)).astype(dtype)

    def regularization_terms(self, params: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        reg = {}
        if self.l2_fourier > 0.0:
            for name in ["A", "B", "A_phase", "B_phase"]:
                if name in params:
                    w = params[name]
                    reg[name] = self.l2_fourier * jnp.sum(jnp.real(jnp.conj(w) * w))
        if self.l2_vertex > 0.0:
            for name in ["alpha", "beta", "lambda", "alpha_phase"]:
                if name in params:
                    w = params[name]
                    reg[name] = self.l2_vertex * jnp.sum(jnp.real(jnp.conj(w) * w))
        if self.l2_edge > 0.0:
            for name in ["visible_bias", "visible_bias_phase"]:
                if name in params:
                    w = params[name]
                    reg[name] = self.l2_edge * jnp.sum(jnp.real(jnp.conj(w) * w))
        if self.l2_loop > 0.0 and "loop_bias" in params:
            w = params["loop_bias"]
            reg["loop_bias"] = self.l2_loop * jnp.sum(jnp.real(jnp.conj(w) * w))
        return reg


def make_cfvtnet_kwargs(graph: Graph, H: AbstractHilbertInterface):
    """Construct graph/hamiltonian dependent kwargs for CFVTNet."""

    def stringify_keys(obj):
        if isinstance(obj, dict):
            return {str(key): stringify_keys(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [stringify_keys(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(stringify_keys(item) for item in obj)
        elif isinstance(obj, set):
            return {stringify_keys(item) for item in obj}
        else:
            return obj

    def flatten_edges(d):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict) and "edges" in value:
                new_dict[key] = value["edges"]
            else:
                new_dict[key] = value
        return new_dict

    return dict(
        n_edges=graph.n_edges,
        n_vertices=graph.n_vertices,
        gauge_dim=H.gauge_dimensions,
        cutoff=H.cutoff,
        edge_to_index=stringify_keys(graph.mapping),
        loops_edges_typed=graph.dressed_minimal_loops(),
        loops_indices=None,
        triplets_by_vertex=stringify_keys(
            flatten_edges(graph.handler.graph_edges_data["graph"]["connectivities"])
        ),
        triplet_signs_by_vertex=stringify_keys(graph.signs),
        q_step=H.hilbert.allowed_basis_states.step,
        q_min=H.hilbert.allowed_basis_states.start,
        q_max=max(H.hilbert.allowed_basis_states.all_states()),
    )
