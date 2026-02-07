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

from typing import Tuple
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


@jax.jit
def _charge_coloring_kernel(
    sigma: jnp.ndarray,
    site_arr: jnp.ndarray,
    charge_arr: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    A diagonal indicator at one component:
        mels = 1.0 if sigma[..., site] == charge else 0.0
    """
    vals = jnp.take(sigma, site_arr, axis=-1)

    # if site_arr accidentally ends up as shape (1,), squeeze the extra axis
    if vals.ndim == sigma.ndim:
        vals = jnp.squeeze(vals, axis=-1)

    mask = vals == charge_arr
    mels = mask.astype(jnp.float64)

    return sigma[..., None, :], mels[..., None]


@register_pytree_node_class
class ChargeColoringJax(ComputationalJaxOperator):
    """
    JAX-native ChargeColoring Operator:
        (ChargeColoringJax at site=e, charge=q) |σ⟩ = [σ[e] == q] · |σ⟩
    """

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, site: int, charge: int):
        super().__init__(H.hilbert)

        self.D = int(H.size)
        e = int(site)
        q = int(charge)

        if not (0 <= e < self.D):
            raise ValueError(
                f"{type(self).__name__}: site index {site} out of bounds for D={self.D}"
            )

        # optional: validate charge against H._core.hi.local_states if present
        try:
            states = jnp.asarray(H.tiny_hilbert.local_states, dtype=jnp.int64)
            if (states == q).sum() == 0:
                raise ValueError(
                    f"{type(self).__name__}: charge {q} not in allowed states {states}."
                )
        except Exception:
            pass

        # keep both Python scalars and 0-d arrays
        # arrays are used by the kernel to avoid static args
        self._site = e
        self._charge = q
        self._site_arr = jnp.asarray(e, dtype=jnp.int64)
        self._charge_arr = jnp.asarray(q, dtype=jnp.int64)

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        leaves = (self._site_arr, self._charge_arr)
        aux = dict(
            hilbert=self.hilbert,
            D=int(self.D),
            site=int(self._site),
            charge=int(self._charge),
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        site_arr, charge_arr = leaves
        obj = cls.__new__(cls)
        # re-init base with hilbert only
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.D = int(aux["D"])
        obj._site = int(aux["site"])
        obj._charge = int(aux["charge"])
        obj._site_arr = site_arr
        obj._charge_arr = charge_arr
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int64)
        sigma_p, mels = _charge_coloring_kernel(
            sigma=x,
            site_arr=self._site_arr,
            charge_arr=self._charge_arr,
        )
        return sigma_p, mels


@jax.jit
def _coloring_kernel(
    sigma: jnp.ndarray,
    comps: jnp.ndarray,
    charge_vec: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    A diagonal indicator across gauge copies at a base edge:
        vec = sigma[..., comps]  # (..., gauge_dim)
        mels = 1.0 if (vec == charge_vec).all(axis=-1) else 0.0
    """
    vec = jnp.take(sigma, comps, axis=-1)
    mask = jnp.all(vec == charge_vec[None], axis=-1)
    mels = mask.astype(jnp.float64)
    return sigma[..., None, :], mels[..., None]


@register_pytree_node_class
class ColoringJax(ComputationalJaxOperator):
    """
    JAX-native Coloring operator:
        (ColoringJax at base site=e, charge_vector=(q1,...,q_g)) |σ⟩
            = [σ[e + k*n_edges_total] == q_{k+1} ∀k=0..g-1] · |σ⟩

    Works for generic gauge_dim (default use-case: U(1)^3).
    """

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def __init__(self, H, site: int, charge_vector: Sequence[int]):
        super().__init__(H.hilbert)

        self.D = int(H.size)
        self.gauge_dim = int(getattr(H, "gauge_dimensions", 3))
        if self.D % self.gauge_dim != 0:
            raise ValueError(
                f"{type(self).__name__}: H.size={self.D} not divisible by "
                f"gauge_dim={self.gauge_dim}."
            )
        self.n_edges_total = int(self.D // self.gauge_dim)

        e0 = int(site)
        if not (0 <= e0 < self.n_edges_total):
            raise ValueError(
                f"{type(self).__name__}: base site={site} must lie in [0, {self.n_edges_total-1}]."
            )

        cv = tuple(int(c) for c in charge_vector)
        if len(cv) != self.gauge_dim:
            raise ValueError(
                f"{type(self).__name__}: charge_vector length {len(cv)} != "
                f"gauge_dim={self.gauge_dim}."
            )

        # build component indices across copies: e0 + k*n_edges_total
        offs = jnp.arange(self.gauge_dim, dtype=jnp.int64) * self.n_edges_total
        comps = (jnp.asarray(e0, dtype=jnp.int64) + offs).astype(jnp.int64)

        self._site0 = e0
        self._comps = comps
        self._charge_vec = jnp.asarray(cv, dtype=jnp.int64)

    #
    #
    #   pytree plumbing

    def tree_flatten(self):
        leaves = (self._comps, self._charge_vec)
        aux = dict(
            hilbert=self.hilbert,
            D=int(self.D),
            gauge_dim=int(self.gauge_dim),
            n_edges_total=int(self.n_edges_total),
            site0=int(self._site0),
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        comps, charge_vec = leaves
        obj = cls.__new__(cls)
        ComputationalJaxOperator.__init__(obj, aux["hilbert"])
        obj.D = int(aux["D"])
        obj.gauge_dim = int(aux["gauge_dim"])
        obj.n_edges_total = int(aux["n_edges_total"])
        obj._site0 = int(aux["site0"])
        obj._comps = comps
        obj._charge_vec = charge_vec
        return obj

    def _get_conn_padded(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.int64)
        sigma_p, mels = _coloring_kernel(
            sigma=x,
            comps=self._comps,
            charge_vec=self._charge_vec,
        )
        return sigma_p, mels
