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
def _identity_kernel(
    sigma: jnp.ndarray,
    const_arr: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Padded-connection identity: returns the same sigma with one diagonal connection
    and matrix element equal to `const_arr`.
    """
    mels = jnp.broadcast_to(const_arr, sigma.shape[:-1])[..., None]
    return sigma[..., None, :], mels


@register_pytree_node_class
class IdentityJax(ComputationalJaxOperator):
    r"""
    Diagonal identity operator with optional scalar factor `const` (real or complex).

        (IdentityJax(const=c)) |σ⟩ = c · |σ⟩

    - Exactly one connection (diagonal).
    - Hermitian iff `c` is real.
    """

    def __init__(self, H, const=1.0):
        super().__init__(H)

        # choose dtype from const
        is_complex = jnp.iscomplexobj(const)
        self._dtype = jnp.complex128 if is_complex else jnp.float64
        # keep const as a 0-d JAX array (leaf) to avoid static args in jit
        self._const_arr = jnp.asarray(const, dtype=self._dtype)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        # c·I is Hermitian iff imag(c) == 0
        return bool(jnp.all(jnp.imag(self._const_arr) == 0))

    @property
    def dtype(self):
        return self._dtype

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        leaves = (self._const_arr,)
        aux = {
            "hilbert": self.hilbert,
            "dtype": self._dtype,
        }
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (const_arr,) = leaves
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])

        obj._dtype = jnp.dtype(aux["dtype"])

        # normal leaf (JAX/NumPy scalar array)
        if hasattr(const_arr, "dtype"):
            obj._const_arr = const_arr
            return obj

        # NetKet/JAX tree_map(jnp.iscomplexobj, ...) -> python bool leaves
        # KEEP python bool, otherwise NetKet's `any(tree_leaves(...))` can hit tracer-bool conversion
        if isinstance(const_arr, bool):
            obj._const_arr = const_arr
            return obj

        # JAX internal dummy unflatten uses object() sentinels
        # (e.g. custom_derivatives building keypaths). Do not cast object()
        if type(const_arr) is object:
            obj._const_arr = jnp.zeros((), dtype=obj._dtype)
            return obj

        # fallback: scalar-like python values
        obj._const_arr = jnp.asarray(const_arr, dtype=obj._dtype)
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int64)
        sigma_p, mels = _identity_kernel(x, self._const_arr)
        return sigma_p, mels
