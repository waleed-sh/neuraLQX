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
    v = vals.astype(jnp.float32)

    def true_branch(_):
        safe = jnp.where(v == 0.0, 0.0, jnp.power(1.0 / v, power))
        return safe

    def false_branch(_):
        return jnp.power(v, power)

    # use JAX conditional instead of Python 'if'
    return jax.lax.cond(inverse, true_branch, false_branch, operand=None)


@jax.jit
def _flux_kernel(
    sigma: jnp.ndarray,
    site: jnp.ndarray,
    power: jnp.ndarray,
    inverse: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    single = sigma.ndim == 2
    if single:
        sigma = sigma[None, :, :]
    B, N, D = sigma.shape
    BN = B * N

    sig = sigma.reshape(BN, D).astype(jnp.int64)

    s = site.astype(int)

    vals = jax.lax.dynamic_slice_in_dim(sig, s, slice_size=1, axis=1)
    # vals = sig[:, s:s+1]

    diag = _flux_eval_step(vals, power, inverse).reshape(BN).astype(jnp.float32)

    sigma_p = sigma[:, :, None, :]
    mels = diag.reshape(B, N, 1)

    return (sigma_p[0], mels[0]) if single else (sigma_p, mels)


@register_pytree_node_class
class QRFluxJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float32

    def __init__(
        self,
        H,
        *,
        site: int,
        power: float = 1.0,
        inverse: bool = False,
    ):
        super().__init__(H.hilbert_netket)

        self._H = H
        self.D = int(H.size)

        if not (isinstance(site, int) and 0 <= site < self.D):
            raise ValueError(f"`site` must be in [0, {self.D-1}], got {site} instead.")

        self._site = int(site)
        self._power = float(power)
        self._inverse = bool(inverse)

    def tree_flatten(self):
        leaves = tuple()
        struct = dict(
            hilbert=self.hilbert,
            D=self.D,
            site=self._site,
            power=self._power,
            inverse=self._inverse,
        )
        return leaves, struct

    @classmethod
    def tree_unflatten(cls, struct, leaves):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(struct["hilbert"])
        obj.D = int(struct["D"])
        obj._site = int(struct["site"])
        obj._power = float(struct["power"])
        obj._inverse = bool(struct["inverse"])
        return obj

    def _get_conn_padded(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.int64)

        sigma_p, mels = _flux_kernel(
            sigma=x,
            site=jnp.asarray(self._site, dtype=jnp.int64),
            power=jnp.asarray(self._power, dtype=jnp.float32),
            inverse=self._inverse,
        )
        return sigma_p, mels
