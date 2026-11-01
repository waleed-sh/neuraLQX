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


"""Magnetic-index link basis for gauge-covariant SU(2) operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from neuralqx.hilbert.constraint import IdentityConstraint
from neuralqx.hilbert.local import LocalRange
from neuralqx.hilbert.space import HomogeneousDiscreteHilbert
from neuralqx.utils.struct import field

from .utils import validate_two_j_max


class SU2PeterWeylHilbert(HomogeneousDiscreteHilbert):
    r"""Truncated product of L²(SU(2), Haar), before imposing Gauss law.

    Each edge stores one integer encoding the normalized basis function
    ``sqrt(2*j+1) D^j_mn(g)``. Within each increasing doubled-spin block,
    ``two_m`` then ``two_n`` increase from ``-two_j`` to ``two_j`` in steps
    of two. Use :meth:`encode` and :meth:`decode` for magnetic labels.

    This basis is different from SU2GaugeInvariantHilbert's intertwiner
    coordinates, state arrays must never be interchanged between them.
    No product basis is enumerated during construction.
    """

    local_space: object = field(static=True, init=False, default=None)
    two_j_max: int = field(static=True, default=1)

    def __post_init__(self):
        cutoff = validate_two_j_max(self.two_j_max)
        if (
            not self.edges
            or self.vertices
            or (
                self.constraint is not None
                and not isinstance(self.constraint, IdentityConstraint)
            )
        ):
            raise ValueError(
                "Peter-Weyl coordinates require unconstrained edge sites only."
            )
        dimension = (cutoff + 1) * (cutoff + 2) * (2 * cutoff + 3) // 6
        if dimension > np.iinfo(np.int32).max:
            raise ValueError("The local Peter-Weyl dimension exceeds int32 indexing.")
        object.__setattr__(
            self, "local_space", LocalRange(0, 1, dimension, dtype=np.int32)
        )
        super().__post_init__()

    def replace(self, **updates):
        """Rebuild derived local ranges and layout when the cutoff changes."""
        parameters = {
            name: getattr(self, name)
            for name, spec in self.__struct_fields__.items()
            if spec.init
        }
        if set(updates) - parameters.keys():
            raise TypeError("Only constructor parameters can be replaced.")
        parameters.update(updates)
        return type(self)(**parameters)

    def encode(self, two_j, two_m, two_n):
        """Encode magnetic labels, returning -1 for invalid/out-of-cutoff labels."""
        j, m, n = jnp.broadcast_arrays(
            jnp.asarray(two_j), jnp.asarray(two_m), jnp.asarray(two_n)
        )
        valid = (
            (j >= 0)
            & (j <= self.two_j_max)
            & (j == jnp.floor(j))
            & (jnp.abs(m) <= j)
            & (jnp.abs(n) <= j)
            & ((m + j) % 2 == 0)
            & ((n + j) % 2 == 0)
        )
        # int64 intermediates avoid overflow in the cubic block offset.
        j, m, n = (a.astype(jnp.int64) for a in (j, m, n))
        index = j * (j + 1) * (2 * j + 1) // 6 + (m + j) // 2 * (j + 1) + (n + j) // 2
        return jnp.where(valid, index, -1).astype(jnp.int32)

    def decode(self, indices):
        """Decode edge indices to (two_j, two_m, two_n), invalid inputs give -1."""
        indices = jnp.asarray(indices)
        j = jnp.arange(self.two_j_max + 2, dtype=jnp.int64)
        offsets = j * (j + 1) * (2 * j + 1) // 6
        spin = jnp.searchsorted(offsets, indices, side="right") - 1
        safe_spin = jnp.clip(spin, 0, self.two_j_max)
        remainder = indices - offsets[safe_spin]
        m = 2 * (remainder // (safe_spin + 1)) - safe_spin
        n = 2 * (remainder % (safe_spin + 1)) - safe_spin
        valid = (
            (indices >= 0)
            & (indices < self.local_size)
            & (indices == jnp.floor(indices))
        )
        return tuple(jnp.where(valid, x, -1).astype(jnp.int32) for x in (spin, m, n))


__all__ = ["SU2PeterWeylHilbert"]
