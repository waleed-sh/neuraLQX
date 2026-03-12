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
    v = vals.astype(jnp.float64)

    if inverse:
        return jnp.where(v == 0.0, 0.0, jnp.power(1.0 / v, power))
    return jnp.power(v, power)


def _directional_g_factor(
    sig: jnp.ndarray,
    *,
    main: int,
    other1: int,
    other2: int,
) -> jnp.ndarray:
    v_main = sig[:, main : main + 1]
    v_o1 = sig[:, other1 : other1 + 1]
    v_o2 = sig[:, other2 : other2 + 1]

    return (
        _flux_eval_step(
            v_main, jnp.asarray(3.0 / 8.0, dtype=jnp.float64), inverse=False
        )
        * _flux_eval_step(v_o1, jnp.asarray(1.0 / 8.0, dtype=jnp.float64), inverse=True)
        * _flux_eval_step(v_o2, jnp.asarray(1.0 / 8.0, dtype=jnp.float64), inverse=True)
    ).reshape(sig.shape[0])


@jax.jit
def _lorentzian_constraint_kernel(
    sigma: jnp.ndarray,
    state_min: jnp.ndarray,
    state_max: jnp.ndarray,
    state_step: jnp.ndarray,
    lorentz_prefactor: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]

    B, N, D = sigma.shape
    BN = B * N

    sig = sigma.reshape(BN, D).astype(jnp.int64)

    # 3 directions * 3 * 3 path combinations
    K = 27
    sigma_p = jnp.broadcast_to(sigma[:, :, None, :], (B, N, K, D))
    mels = jnp.zeros((B, N, K), dtype=jnp.float64)

    lo = state_min
    hi = state_max
    step = state_step

    directions = ((0, 1, 2), (1, 0, 2), (2, 0, 1))
    step_deltas = (0, +1, -1)
    step_coeffs = (+0.5, -0.25, -0.25)

    k = 0
    for main, other1, other2 in directions:
        v_main = sig[:, main : main + 1]
        g_right = _directional_g_factor(
            sig,
            main=main,
            other1=other1,
            other2=other2,
        )

        for d1, c1 in zip(step_deltas, step_coeffs):
            cand_1 = v_main + d1 * step
            mask_1 = ((cand_1 >= lo) & (cand_1 <= hi)).reshape(BN)

            sig_1 = sig.at[:, main : main + 1].set(
                jnp.where(mask_1[:, None], cand_1, v_main)
            )
            g_mid = _directional_g_factor(
                sig_1,
                main=main,
                other1=other1,
                other2=other2,
            )
            d1_w = jnp.where(
                mask_1,
                (c1 * g_mid * g_right).astype(jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
            )

            v_1 = sig_1[:, main : main + 1]

            for d2, c2 in zip(step_deltas, step_coeffs):
                cand_2 = v_1 + d2 * step
                mask_2 = mask_1 & ((cand_2 >= lo) & (cand_2 <= hi)).reshape(BN)

                sig_2 = sig_1.at[:, main : main + 1].set(
                    jnp.where(mask_2[:, None], cand_2, v_1)
                )
                g_left = _directional_g_factor(
                    sig_2,
                    main=main,
                    other1=other1,
                    other2=other2,
                )
                d2_w = jnp.where(
                    mask_2,
                    (c2 * g_left * g_mid).astype(jnp.float64),
                    jnp.asarray(0.0, dtype=jnp.float64),
                )

                w = (lorentz_prefactor * (-16.0 * d2_w * d1_w)).astype(jnp.float64)

                sigma_p = sigma_p.at[:, :, k, :].set(sig_2.reshape(B, N, D))
                mels = mels.at[:, :, k].set(w.reshape(B, N))
                k += 1

    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class LorentzianConstraintJax(ComputationalJaxOperator):
    @property
    def max_conn_size(self) -> int:
        return 27

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, *, immirzi: float = 1.0):
        super().__init__(H.hilbert_netket)

        self._H = H
        self.D = int(H.size)

        n_edges_total = int(H.tiny_size)
        if n_edges_total != 3:
            raise ValueError(
                f"{type(self).__name__} requires a Single-Vertex graph with 3 edges, "
                f"got {n_edges_total} edges."
            )

        if self.D < 3:
            raise ValueError(f"Unexpected state dimension D={self.D} (< 3).")

        self._state_min = int(H.allowed_basis_states.start)
        self._mod_span = int(H.allowed_basis_states.length)
        self._state_step = int(H.allowed_basis_states.step)
        self._state_max = int(self._state_min + (self._mod_span - 1) * self._state_step)
        self._immirzi = float(immirzi)
        if self._immirzi == 0.0:
            raise ValueError("The Immirzi parameter must be non-zero.")

    def tree_flatten(self):
        leaves = tuple()
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            state_min=self._state_min,
            state_max=self._state_max,
            state_step=self._state_step,
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
        obj._immirzi = float(struct["immirzi"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)

        sigma_p, mels = _lorentzian_constraint_kernel(
            sigma=x,
            state_min=jnp.asarray(self._state_min, dtype=jnp.int64),
            state_max=jnp.asarray(self._state_max, dtype=jnp.int64),
            state_step=jnp.asarray(self._state_step, dtype=jnp.int64),
            lorentz_prefactor=jnp.asarray(
                (1.0 + (self._immirzi * self._immirzi))
                / (self._immirzi * self._immirzi),
                dtype=jnp.float64,
            ),
        )
        return sigma_p, mels
