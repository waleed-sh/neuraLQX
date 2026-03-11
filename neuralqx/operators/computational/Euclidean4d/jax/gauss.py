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

from functools import partial
from typing import List
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.tree_util import register_pytree_node_class

from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


def wrap_mod(
    vals_i32: jnp.ndarray,
    delta_i32: jnp.ndarray,
    state_min: int,
    mod_span: int,
    state_step: int,
) -> jnp.ndarray:
    # map to indices on the discrete lattice, add, wrap, map back
    idx_vals = (vals_i32 - state_min) // state_step
    idx_delta = delta_i32 // state_step
    idx_sum = (idx_vals + idx_delta) % mod_span
    return idx_sum * state_step + state_min


@partial(
    jax.jit,
    static_argnames=(
        "gauge_dim",
        "n_vertices",
        "k_max",
        "modular",
        "mod_span",
        "state_min",
        "state_step",
    ),
)
def gauss_constraint_kernel(
    sigma: jnp.ndarray,
    comps_pad: jnp.ndarray,
    signs_pad: jnp.ndarray,
    *,
    gauge_dim: int,
    n_vertices: int,
    k_max: int,
    modular: bool,
    mod_span: int,
    state_min: int,
    state_step: int,
):
    # promote (N,D) -> (1,N,D)
    single_input = sigma.ndim == 2
    if single_input:
        sigma = sigma[None, :, :]

    B, N, D = sigma.shape
    M = B * N

    if n_vertices == 0:
        sigmap = sigma[:, :, None, :]
        zeros = jnp.zeros((B, N, 1), dtype=jnp.float64)
        return (sigmap[0], zeros[0]) if single_input else (sigmap, zeros)

    sig = sigma.reshape((M, D)).astype(jnp.int64)
    total = jnp.zeros((M,), dtype=jnp.float64)

    def body_vertex(v, tot_cur):
        comps_v = comps_pad[v]
        signs_v = signs_pad[v]

        # flatten indices and safe-gather (map -1 -> 0, then mask by sign == 0)
        comp_flat = comps_v.reshape((-1,))
        safe_idx = jnp.where(comp_flat >= 0, comp_flat, 0)
        gathered = jnp.take(sig, safe_idx, axis=-1)
        gathered = gathered.reshape((M, k_max, gauge_dim))

        if modular:
            # accumulate with modular wrapping, skipping padded rows (sign == 0)
            Gv = jnp.zeros((M, gauge_dim), dtype=jnp.int64)

            def body_edge(k, G_cur):
                s = signs_v[k]
                valid = s != 0
                term = gathered[:, k, :]
                add_term = jnp.where(s > 0, term, -term)

                return lax.cond(
                    valid,
                    lambda t: wrap_mod(G_cur, t, state_min, mod_span, state_step),
                    lambda _: G_cur,
                    add_term,
                )

            Gv = lax.fori_loop(0, k_max, body_edge, Gv)
            add_val = jnp.sum((Gv.astype(jnp.float64)) ** 2, axis=-1)
            return tot_cur + add_val
        else:
            signed = gathered.astype(jnp.float64) * signs_v[None, :, None].astype(
                jnp.float64
            )
            Gv = jnp.sum(signed, axis=1)
            add_val = jnp.sum(Gv * Gv, axis=-1)
            return tot_cur + add_val

    total = lax.fori_loop(0, n_vertices, body_vertex, total)

    vals = total.reshape((B, N))
    sigmap = sigma[:, :, None, :]
    mels = vals[:, :, None]

    if single_input:
        return sigmap[0], mels[0]
    return sigmap, mels


@register_pytree_node_class
class GaussConstraintOperatorJax(ComputationalJaxOperator):
    """
    JAX-compiled diagonal Gauß constraint operator:
        sum_v || sum_{e->v} m_e - sum_{e<-v} m_e ||_2^2
    Returns exactly one connection (the input σ) and its matrix element.
    """

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(
        self,
        H: AbstractHilbertInterface,
        gauge_dim: Union[int, None] = None,
        *,
        modded: bool = False,
    ):
        super().__init__(H.hilbert_netket)

        self._H = H

        # static ints
        self.gauge_dim = int(gauge_dim if gauge_dim is not None else H.gauge_dimensions)

        self.n_edges_total = int(H.size // self.gauge_dim)

        # modular arithmetic parameters
        self.modular = bool(modded)

        self.state_min = int(H.allowed_basis_states.start)
        self.state_step = int(H.allowed_basis_states.step)
        self.mod_span = int(H.allowed_basis_states.length)
        self.state_max = int(self.state_min + (self.mod_span - 1) * self.state_step)

        # build padded per-vertex structures
        node_conn = H.graph.handler.list_of_node_connectivity
        offsets = (
            jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total
        ).astype(jnp.int64)

        per_v_comps: List[jnp.ndarray] = []
        per_v_signs: List[jnp.ndarray] = []

        for _, attr in node_conn.items():
            comps_list = []
            signs_list = []

            for edge in attr["incoming"]:
                eidx = int(H.graph.edge_to_index(edge))
                comps_list.append((jnp.int64(eidx) + offsets).astype(jnp.int64))
                signs_list.append(jnp.int8(1))

            for edge in attr["outgoing"]:
                eidx = int(H.graph.edge_to_index(edge))
                comps_list.append((jnp.int64(eidx) + offsets).astype(jnp.int64))
                signs_list.append(jnp.int8(-1))

            if comps_list:
                comps_v = jnp.stack(comps_list, axis=0)
                signs_v = jnp.stack(signs_list, axis=0)
            else:
                comps_v = jnp.zeros((0, self.gauge_dim), dtype=jnp.int64)
                signs_v = jnp.zeros((0,), dtype=jnp.int8)

            per_v_comps.append(comps_v)
            per_v_signs.append(signs_v)

        V = len(per_v_comps)
        if V == 0:
            self.comps_pad = jnp.zeros((0, 0, self.gauge_dim), dtype=jnp.int64)
            self.signs_pad = jnp.zeros((0, 0), dtype=jnp.int8)
            self.n_vertices = 0
            self.k_max = 0
        else:
            K_max = max(int(x.shape[0]) for x in per_v_comps)
            self.n_vertices = V
            self.k_max = K_max

            if K_max == 0:
                self.comps_pad = jnp.zeros((V, 0, self.gauge_dim), dtype=jnp.int64)
                self.signs_pad = jnp.zeros((V, 0), dtype=jnp.int8)
            else:
                comps_pad = []
                signs_pad = []
                for comps_v, signs_v in zip(per_v_comps, per_v_signs):
                    k = int(comps_v.shape[0])
                    if k < K_max:
                        pad_c = ((0, K_max - k), (0, 0))
                        pad_s = ((0, K_max - k),)
                        comps_pad.append(jnp.pad(comps_v, pad_c, constant_values=-1))
                        signs_pad.append(jnp.pad(signs_v, pad_s, constant_values=0))
                    else:
                        comps_pad.append(comps_v)
                        signs_pad.append(signs_v)
                self.comps_pad = jnp.stack(comps_pad, axis=0).astype(jnp.int64)
                self.signs_pad = jnp.stack(signs_pad, axis=0).astype(jnp.int8)

    @property
    def max_conn_size(self) -> int:
        return 1

    def _get_conn_padded(self, x_local: np.ndarray | jnp.ndarray):

        x = jnp.asarray(x_local, dtype=jnp.int64)

        sigma_p, mels = gauss_constraint_kernel(
            x,
            self.comps_pad,
            self.signs_pad,
            gauge_dim=self.gauge_dim,
            n_vertices=self.n_vertices,
            k_max=self.k_max,
            modular=self.modular,
            mod_span=self.mod_span,
            state_min=self.state_min,
            state_step=self.state_step,
        )

        return sigma_p, mels

    def tree_flatten(self):

        children = (self.comps_pad, self.signs_pad)
        aux = dict(
            hilbert=self.hilbert,
            gauge_dim=self.gauge_dim,
            n_edges_total=self.n_edges_total,
            modular=self.modular,
            state_min=self.state_min,
            state_max=self.state_max,
            mod_span=self.mod_span,
            state_step=self.state_step,
            n_vertices=self.n_vertices,
            k_max=self.k_max,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        comps_pad, signs_pad = children

        # rebuild a minimal instance, we bypass geometry rebuild and just stuff the precomputed
        # arrays and metadata back in
        obj = cls.__new__(cls)
        super(cls, obj).__init__(aux["hilbert"])

        obj._H = None

        obj.gauge_dim = int(aux["gauge_dim"])
        obj.n_edges_total = int(aux["n_edges_total"])
        obj.modular = bool(aux["modular"])

        obj.state_min = int(aux["state_min"])
        obj.state_max = int(aux["state_max"])
        obj.mod_span = int(aux["mod_span"])
        obj.state_step = int(aux["state_step"])

        obj.n_vertices = int(aux["n_vertices"])
        obj.k_max = int(aux["k_max"])

        obj.comps_pad = comps_pad
        obj.signs_pad = signs_pad

        return obj
