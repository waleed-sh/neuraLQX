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

import numpy as np
import jax.numpy as jnp

from neuralqx.operators.types.computational_operator import ComputationalOperator


class ChargeColoring(ComputationalOperator):
    """
    Single-site charge-coloring: δ_{σ[site], charge}.

    This is a diagonal indicator on one degree of freedom (edge-component):
        (ChargeColoring at site=e, charge=q) |σ⟩ = [σ[e] == q] · |σ⟩

    Returns a single diagonal connection (σ itself) with matrix element 1.0 when
    σ[..., site]==charge, otherwise 0.0.
    """

    def __init__(self, H, site: int, charge: int):
        super().__init__(H.hilbert)

        self._H = H
        self.D = int(H.size)
        self._site = int(site)
        self._charge = int(charge)

        # basic validations
        if not (0 <= self._site < self.D):
            raise ValueError(
                f"{type(self).__name__}: site index {site} out of bounds for D={self.D}"
            )

        # validate charge against allowed local states if available
        # (works for symmetric [-N..N] and [0..Q] encodings alike)
        try:
            states = np.asarray(H.tiny_hilbert.local_states, dtype=np.int64)
            if self._charge not in states.tolist():
                raise ValueError(
                    f"{type(self).__name__}: charge {charge} not in allowed states {states}."
                )
        except Exception:
            # if local_states not present, fall back to cutoff-based sanity check when available
            if hasattr(H, "cutoff"):
                if abs(self._charge) > int(H.cutoff):
                    raise ValueError(
                        f"{type(self).__name__}: |charge|={abs(self._charge)} exceeds cutoff={H.cutoff}."
                    )

        # one diagonal connection
        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mask = σ[..., self._site] == self._charge
        vals = mask.astype(self.dtype)

        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(vals, axis=-1)
        return σp, mels


class Coloring(ComputationalOperator):
    """
    Copy-resolved coloring on U(1)^gdim: δ_{(σ[site+k*nE])_k, charge_vector}

    This is a diagonal indicator on a base edge 'site' across gauge copies:
        (Coloring at site=e, charge_vector=(q1,...,q_g)) |σ⟩
            = [ (σ[e + k*n_edges_total] == q_{k+1}  for all k) ] · |σ⟩

    Works for generic gauge_dim (default use-case: U(1)^3 with gauge_dim=3).
    """

    def __init__(self, H, site: int, charge_vector: Sequence[int]):
        super().__init__(H.hilbert)

        self._H = H
        self.gauge_dim = int(getattr(H, "gauge_dimensions", 3))
        self.D = int(H.size)
        self._site = int(site)

        # normalise and validate charge_vector
        cv = tuple(int(c) for c in charge_vector)
        if len(cv) != self.gauge_dim:
            raise ValueError(
                f"{type(self).__name__}: charge_vector length {len(cv)} "
                f"must equal gauge_dim={self.gauge_dim}."
            )
        self._charge_vec = jnp.asarray(cv, dtype=jnp.int64)

        # edges per copy
        if self.D % self.gauge_dim != 0:
            raise ValueError(
                f"{type(self).__name__}: H.size={self.D} not divisible by "
                f"gauge_dim={self.gauge_dim}."
            )
        self._n_edges_total = int(self.D // self.gauge_dim)

        # site must refer to the base (copy-0) index
        if not (0 <= self._site < self._n_edges_total):
            raise ValueError(
                f"{type(self).__name__}: site={site} must be in [0, {self._n_edges_total-1}] "
                f"(base graph indices only)."
            )

        # optionally validate each component charge vs local_states if present
        try:
            states = np.asarray(H.tiny_hilbert.local_states, dtype=np.int64)
            st = set(states.tolist())
            bad = [c for c in cv if c not in st]
            if bad:
                raise ValueError(
                    f"{type(self).__name__}: charges {bad} not in allowed states {states}."
                )
        except Exception:
            if hasattr(H, "cutoff"):
                cmax = int(H.cutoff)
                for c in cv:
                    if abs(c) > cmax:
                        raise ValueError(
                            f"{type(self).__name__}: |charge|={abs(c)} exceeds cutoff={cmax}."
                        )

        # precompute component indices across copies: site + k*n_edges_total
        offs = jnp.arange(self.gauge_dim, dtype=jnp.int64) * self._n_edges_total
        self._comps = (jnp.asarray(self._site, dtype=jnp.int64) + offs).astype(
            jnp.int64
        )

        # one diagonal connection
        self._nconn = 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return jnp.float64

    def _get_conn_padded_kernel(
        self, σ: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # gather the gdim components of this base edge
        vec = jnp.take(σ, self._comps, axis=-1)
        mask = jnp.all(vec == self._charge_vec, axis=-1)
        vals = mask.astype(self.dtype)

        σp = jnp.expand_dims(σ, axis=-2)
        mels = jnp.expand_dims(vals, axis=-1)
        return σp, mels
