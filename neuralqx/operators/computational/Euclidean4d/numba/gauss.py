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


import jax.numpy as jnp
from typing import List
from typing import Tuple

from neuralqx.utils.misc.arithmetic import mod_add
from neuralqx.operators.types.computational_operator import ComputationalOperator


class GaussConstraintOperator(ComputationalOperator):
    r"""
    Diagonal Gauß constraint operator enforcing charge-vector conservation at each vertex.

    For a gauge of dimension ``gauge_dim`` (e.g. U(1)^g), let each edge ``e`` carry a charge vector
    :math:`\vec m_e \in \mathbb{Z}^{g}`. For each vertex :math:`v`, define the Gauss vector
    :math:`\vec G_v(\sigma) = \sum_{e\to v}\vec m_e - \sum_{e\leftarrow v}\vec m_e`,
    i.e. incoming minus outgoing charge vectors.

    The constraint value on a configuration :math:`\sigma` is the sum of squared norms
    (L2) of all Gauss vectors:
    .. math::
        \mathcal{G}(\sigma) = \sum_{v} \|\vec G_v(\sigma)\|_2^2
        = \sum_v \sum_{a=1}^{g} \bigl(G_v^{(a)}(\sigma)\bigr)^2.

    The operator is diagonal in the charge basis and returns exactly one connection: the state
    itself with matrix element :math:`\mathcal{G}(\sigma)`.
    """

    def __init__(
        self,
        H,
        gauge_dim: int = None,
        modded: bool = False,
    ):
        super().__init__(H.hilbert_netket)

        # static layout
        self._H = H
        self.gauge_dim = int(gauge_dim if gauge_dim is not None else H.gauge_dimensions)

        # quick sanity check
        assert (
            self.gauge_dim >= 1
        ), f"Invalid number of gauge dimensions {self.gauge_dim}"

        # get the number of edges per gauge level
        self.n_edges_total = int(H.size // self.gauge_dim)

        # modular or not
        self.modular = modded
        # self.modulus = H.cutoff
        self.mod_sum = mod_add

        # get the allowed quantum numbers

        # step between allowed DoFs
        self._state_step = int(H.allowed_basis_states.step)

        # a span of allowed DoFs for the modded addition
        self._mod_span = int(H.allowed_basis_states.length)

        # lowest
        self._state_min = int(H.allowed_basis_states.start)

        # highest
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)

        # graph connectivity with orientation
        # expect H.g.handler.list_of_node_connectivity[v] to have an "incoming" and "outgoing"
        # edge lists
        node_conn = H.graph.handler.list_of_node_connectivity

        # precompute, for each vertex v:
        #   - comps_v: (K_v, gauge_dim) component indices for all incident edges (stacked)
        #   - signs_v: (K_v,) with +1 for incoming, -1 for outgoing, repeated per edge
        per_v_comps: List[jnp.ndarray] = []
        per_v_signs: List[jnp.ndarray] = []

        offsets = jnp.arange(self.gauge_dim, dtype=jnp.int32) * self.n_edges_total

        # keep a stable vertex ordering (string keys? use the iteration order of the dict)
        self._vertices: List[int] = []

        # loop through every vertex
        for v_str, attr in node_conn.items():

            # append the vertex into the list of vertices saved as an instance attribute
            v = int(v_str) if not isinstance(v_str, int) else v_str
            self._vertices.append(v)

            # we now collect incoming and outgoing edges
            comps_list = []
            signs_list = []

            # incoming edges: +1
            for edge in attr["incoming"]:
                eidx = H.graph.edge_to_index(edge)
                comps_e = (jnp.int32(eidx) + offsets).astype(jnp.int32)
                comps_list.append(comps_e)
                signs_list.append(jnp.int8(1))

            # outgoing edges: -1
            for edge in attr["outgoing"]:
                eidx = H.graph.edge_to_index(edge)
                comps_e = (jnp.int32(eidx) + offsets).astype(jnp.int32)
                comps_list.append(comps_e)
                signs_list.append(jnp.int8(-1))

            if comps_list:
                comps_v = jnp.stack(comps_list, axis=0)
                signs_v = jnp.stack(signs_list, axis=0)
            else:
                # isolated vertex: zero-size, handled gracefully in kernel
                comps_v = jnp.zeros((0, self.gauge_dim), dtype=jnp.int32)
                signs_v = jnp.zeros((0,), dtype=jnp.int8)

            per_v_comps.append(comps_v)
            per_v_signs.append(signs_v)

        # pack static tuples (one entry per vertex)
        # shapes are fixed for the life of this instance
        self._per_v_comps: Tuple[jnp.ndarray, ...] = tuple(per_v_comps)
        self._per_v_signs: Tuple[jnp.ndarray, ...] = tuple(per_v_signs)
        self._n_vertices: int = len(self._vertices)

    @property
    def is_hermitian(self) -> bool:
        # the Gauss constraint is diagonal/Hermitian in the charge basis
        return True

    @property
    def dtype(self):
        # matrix elements are real, use float64
        return jnp.float64

    def _sum_or_mod(self, a, b):
        """Perform modular or normal addition depending on flag."""
        if self.modular:
            return self.mod_sum(
                a,
                b,
                q_min=self._state_min,
                q_max=self._state_max,
                step=self._state_step,
            )
        else:
            return a + b

    def _sub_or_mod(self, a, b):
        """Perform modular or normal subtraction depending on flag."""
        if self.modular:
            # subtraction is just modular addition with negative b
            return self.mod_sum(
                a,
                -b,
                q_min=self._state_min,
                q_max=self._state_max,
                step=self._state_step,
            )
        else:
            return a - b

    def _get_conn_padded_kernel(self, σ: jnp.ndarray):

        # shapes as usual
        # B, N, D = σ.shape

        if σ.ndim == 2:
            # treat as (B = 1, N, D)
            σ = σ[None, :, :]
        B, N, D = σ.shape

        M = B * N
        sig = σ.reshape(M, D)

        # accumulate sum_v ||G_v||^2 in a numerically stable way
        total = jnp.zeros((M,), dtype=self.dtype)

        # loop over vertices, number of vertices is fixed per instance (jit-stable)
        for v in range(self._n_vertices):
            comps_v = self._per_v_comps[v]
            signs_v = self._per_v_signs[v]

            if comps_v.shape[0] == 0:
                # isolated vertex, contributes 0
                continue

            # gather all edge charge vectors incident at v, will have shape (M, K_v, g)
            # we can use take with 2D indices via vmap-like trick: take per column then stack along
            # last axis
            # we can also do the simpler following: flatten comps_v to (K_v*g,) and then reshape
            # after gather
            comp_flat = comps_v.reshape(-1)
            gathered = jnp.take(sig, comp_flat, axis=-1)
            gathered = gathered.reshape(M, comps_v.shape[0], self.gauge_dim)

            # apply signs (+1 incoming, -1 outgoing) along the edges axis
            # signed = gathered * signs_v[None, :, None]
            if self.modular:
                Gv = jnp.zeros((M, self.gauge_dim), dtype=self.dtype)
                for k in range(comps_v.shape[0]):
                    term = gathered[:, k, :]
                    s = signs_v[k]
                    if s == 1:
                        Gv = self._sum_or_mod(Gv, term)
                    else:
                        Gv = self._sub_or_mod(Gv, term)
            else:
                signed = gathered * signs_v[None, :, None]
                Gv = jnp.sum(signed, axis=1)

            # sum over incident edges -> Gauss vector per vertex with shape (M, g)
            # Gv = jnp.sum(signed, axis=1)

            # add squared norm to total: ||Gv||^2 = sum_a Gv_a^2
            # total = total + jnp.sum(Gv * Gv, axis=-1)
            if self.modular:
                total = self._sum_or_mod(total, jnp.sum(Gv * Gv, axis=-1))
            else:
                total = total + jnp.sum(Gv * Gv, axis=-1)

        # reshape back to (B, N) and pad as diagonal operator output
        vals = total.reshape(B, N).astype(self.dtype)
        σp = σ[:, :, None, :]
        mels = vals[:, :, None]

        return σp, mels
