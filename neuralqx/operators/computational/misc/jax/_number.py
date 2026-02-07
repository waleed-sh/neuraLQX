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


@jax.jit
def _number_kernel(
    sigma: jnp.ndarray,
    edge_idx_arr: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # gather along last axis as a (..., 1), then squeeze -> (...,)
    vals = jnp.take(sigma, edge_idx_arr, axis=-1)

    if vals.ndim == sigma.ndim:
        vals = jnp.squeeze(vals, axis=-1)

    # cast to float64 for matrix elements
    vals = vals.astype(jnp.float64)

    # one diagonal connection
    return sigma[..., None, :], vals[..., None]


@register_pytree_node_class
class NumberJax(ComputationalJaxOperator):

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, edge: int):
        super().__init__(H.hilbert)

        self.D = int(H.size)
        e = int(edge)
        if not (0 <= e < self.D):
            raise ValueError(
                f"{type(self).__name__}: edge index {edge} out of bounds for D={self.D}"
            )

        # store as int and as an int64 array for the kernel (to avoid static args)
        self._edge = e
        self._edge_arr = jnp.asarray(e, dtype=jnp.int64)

    #
    #
    #   pytree support

    def tree_flatten(self):
        # keep array leaves minimal, put metadata in the aux dict
        leaves = (self._edge_arr,)
        aux = {
            "hilbert": self.hilbert,
            "D": self.D,
            "edge": self._edge,
        }
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (edge_arr,) = leaves
        obj = cls.__new__(cls)
        # reinit base with hilbert
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.D = int(aux["D"])
        obj._edge = int(aux["edge"])
        obj._edge_arr = edge_arr
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = jnp.asarray(x, dtype=jnp.int64)

        sigma_p, mels = _number_kernel(x, self._edge_arr)

        return sigma_p, mels
