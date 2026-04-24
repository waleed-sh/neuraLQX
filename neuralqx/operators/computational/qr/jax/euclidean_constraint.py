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

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


def _flux_eval_step(
    vals: jnp.ndarray,
    power: jnp.ndarray,
    inverse: bool,
) -> jnp.ndarray:
    """
    Implements the QRLG flux eigenvalue evaluator, which is diagonal. If `inverse` is set to False,
    then the computed flux is `v^power`. If set to True, then the computed flux is (1/v)^power, with
    0 -> 0 (no divergence).
    """
    v = vals.astype(jnp.float64)

    if inverse:
        # 0 -> 0, else (1/v)^power
        safe = jnp.where(v == 0.0, 0.0, jnp.power(1.0 / v, power))
        return safe
    else:
        return jnp.power(v, power)


@jax.jit
def _euclidean_constraint_kernel(
    sigma: jnp.ndarray,
    state_min: jnp.ndarray,
    state_max: jnp.ndarray,
    state_step: jnp.ndarray,
    flux_power: jnp.ndarray,
    lapse: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX-jitted kernel for the Euclidean constraint on a single-vertex, 3-edge model.
    """

    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N

    sig = sigma.reshape(BN, D).astype(jnp.int64)

    # 3 terms, 4 symmetric-holonomy combos per term -> 12 connections
    K = 12
    sigma_p = jnp.broadcast_to(sigma[:, :, None, :], (B, N, K, D))
    mels = jnp.zeros((B, N, K), dtype=jnp.float64)

    lo = state_min
    hi = state_max
    step = state_step
    p = flux_power
    lapse_f = lapse.astype(jnp.float64)

    # fixed edge triplets (a, b, z) and symmetric holonomy combinations
    triplets = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    comb_deltas = ((+1, +1), (+1, -1), (-1, +1), (-1, -1))
    comb_signs = (+1.0, -1.0, -1.0, +1.0)  # (+,+), (+,-), (-,+), (-,-)

    for term_idx, (a, b, z) in enumerate(triplets):
        # components at edges a, b, z
        va = sig[:, a : a + 1]
        vb = sig[:, b : b + 1]
        vz = sig[:, z : z + 1]

        # right diagonal prefactor F_right(σ)
        F_right = (
            _flux_eval_step(va, p, inverse=False)
            * _flux_eval_step(vb, p, inverse=False)
            * _flux_eval_step(vz, p, inverse=True)
        ).reshape(BN)

        for comb_idx, ((da, db), sign) in enumerate(zip(comb_deltas, comb_signs)):

            # connection index in [0, 11]
            k = term_idx * 4 + comb_idx

            # proposed new values for a, b
            new_a = va + da * step
            new_b = vb + db * step

            # non-cyclic bounds check
            mask = (
                (new_a >= lo) & (new_a <= hi) & (new_b >= lo) & (new_b <= hi)
            ).reshape(BN)

            # build σ' by updating only a and b where valid, otherwise keep original
            sig_k = sig
            sig_k = sig_k.at[:, a : a + 1].set(jnp.where(mask[:, None], new_a, va))
            sig_k = sig_k.at[:, b : b + 1].set(jnp.where(mask[:, None], new_b, vb))

            # left diagonal prefactor F_left(σ')
            va_p = sig_k[:, a : a + 1]
            vb_p = sig_k[:, b : b + 1]

            F_left = (
                _flux_eval_step(va_p, p, inverse=False)
                * _flux_eval_step(vb_p, p, inverse=False)
                * _flux_eval_step(vz, p, inverse=True)
            ).reshape(BN)

            # weight: -(1/4) * sign * lapse * F_left * F_right
            w = -(0.25 * sign) * lapse_f * F_left * F_right
            w = jnp.where(
                mask,
                w.astype(jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
            )

            # scatter into k-th connection
            sig_k_BND = sig_k.reshape(B, N, D)
            sigma_p = sigma_p.at[:, :, k, :].set(sig_k_BND)
            mels = mels.at[:, :, k].set(w.reshape(B, N))

    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class EuclideanConstraintJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        # 3 (edge pairs) * 4 (holonomy combos) = 12
        return 12

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def adjoint(self) -> "EuclideanConstraintJax":
        """The Euclidean constraint is Hermitian, so its adjoint is itself."""
        return self

    @property
    def dtype(self):
        return jnp.float64

    def __init__(
        self,
        H,
        *,
        lapse: float = 1.0,
        power: float = 0.25,
        immirzi: float = 1.0,
    ):

        super().__init__(H.hilbert_netket)

        self._H = H
        self.D = int(H.size)

        # require 3 edges total (single vertex, x/y/z)
        n_edges_total = int(H.tiny_size)
        if n_edges_total != 3:
            raise ValueError(
                f"{type(self).__name__} requires a Single-Vertex graph with 3 edges, "
                f"got {n_edges_total} edges."
            )
        if self.D < 3:
            raise ValueError(f"Unexpected state dimension D={self.D} (< 3).")

        # local state lattice
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

        self._lapse = float(lapse)
        self._flux_power = float(power)
        self._immirzi = float(immirzi)
        if self._immirzi == 0.0:
            raise ValueError("The Immirzi parameter must be non-zero.")

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        # no array leaves, all static scalars
        leaves = tuple()
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            state_min=self._state_min,
            state_max=self._state_max,
            state_step=self._state_step,
            lapse=self._lapse,
            flux_power=self._flux_power,
            immirzi=self._immirzi,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj._state_min = int(struct["state_min"])
        obj._state_max = int(struct["state_max"])
        obj._state_step = int(struct["state_step"])
        obj._lapse = float(struct["lapse"])
        obj._flux_power = float(struct["flux_power"])
        obj._immirzi = float(struct["immirzi"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):

        x = jnp.asarray(x, dtype=jnp.int64)

        sigma_p, mels = _euclidean_constraint_kernel(
            sigma=x,
            state_min=jnp.asarray(self._state_min, dtype=jnp.int64),
            state_max=jnp.asarray(self._state_max, dtype=jnp.int64),
            state_step=jnp.asarray(self._state_step, dtype=jnp.int64),
            flux_power=jnp.asarray(self._flux_power, dtype=jnp.float64),
            lapse=jnp.asarray(
                self._lapse / (self._immirzi * self._immirzi), dtype=jnp.float64
            ),
        )
        return sigma_p, mels
