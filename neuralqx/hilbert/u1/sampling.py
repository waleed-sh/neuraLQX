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


"""Sampling API for U(1) Hilbert spaces.

The mixin exposes the public ``random_state`` method for U(1) spaces and
forwards the actual array creation to the jitted sampling helper.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from .random import u1_random_state_jit


class U1SamplingMixin:
    """Random-state generation for homogeneous U(1) charge spaces.

    The mixin implements direct sampling from the finite U(1) local charge
    range. It does not use rejection sampling because every scalar charge is
    drawn from an already valid local domain. Gauge-invariant subclasses may
    call this implementation to sample an unconstrained candidate and then
    impose their constructive relations.

    Concrete spaces must provide ``size``, ``dtype``, ``local_space``, and the
    static metadata consumed by the jitted U(1) sampling helper.
    """

    def random_state(
        self,
        key: jax.Array,
        size: int | tuple[int, ...] | None = None,
        *,
        dtype: Any | None = None,
        max_trials: int = 1024,
    ) -> jax.Array:
        """Samples random U(1) states from the local charge range.

        Args:
            key: JAX pseudo-random key.
            size: Optional leading batch shape.
            dtype: Optional dtype for generated state values.
            max_trials: Ignored for direct U(1) sampling.

        Returns:
            Random state array with trailing dimension ``self.size``.
        """
        del max_trials
        out_dtype = self.dtype if dtype is None else np.dtype(dtype)
        return u1_random_state_jit(self, key, size, out_dtype)
