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


"""Random valid-state generation for SU(2) Hilbert spaces.

SU(2) spaces avoid generic box rejection. Cached exact bases support direct
uniform draws, otherwise a compiled Metropolis warmup generates valid states.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.hilbert.su2.proposals import random_loop_warmup
from neuralqx.hilbert.su2.proposals.metropolis import random_metropolis_warmup
from neuralqx.hilbert.utils import batch_shape


class SU2SamplingMixin:
    """Constructs valid SU(2) spin-network states without box rejection.

    SU(2) vertex admissibility makes naive rectangular rejection sampling
    inefficient and, for some graphs, impractical. This mixin therefore samples
    from the cached exact basis when available, returns the vacuum sector when
    requested, or runs independent uniform-target Metropolis chains. A finite
    warmup produces random initial states, not certified equilibrium samples.

    Concrete spaces must expose ``_exact_states``, ``random_init``,
    ``two_j_max``, ``size``, and ``dtype``. The public return convention
    matches the rest of the Hilbert API: leading axes are the requested batch
    shape and the trailing axis is the flat spin-network state.
    """

    def random_state(
        self,
        key: jax.Array,
        size: int | tuple[int, ...] | None = None,
        *,
        dtype: Any | None = None,
        max_trials: int = 1024,
        method: str | None = None,
        n_steps: int | None = None,
    ) -> jax.Array:
        """Samples valid SU(2) spin-network states.

        Args:
            key: JAX pseudo-random key.
            size: Optional leading batch shape.
            dtype: Optional dtype for returned state values.
            max_trials: Ignored because SU(2) uses constructive sampling.
            method: Per-call override: ``"metropolis"``, ``"vacuum"`` or
                legacy ``"loop_warmup"``. By default, sample a cached exact
                basis if present, otherwise use ``self.random_init``.
            n_steps: Attempted updates per output state. Defaults to
                ``self.init_warmup`` or ``max(32, 4*n_edges)``. Static under JIT.
                Used for Metropolis and legacy loop warmup only.

        Returns:
            Valid SU(2) state array with trailing dimension ``self.size``.
            The Metropolis path has a uniform stationary target and includes
            the Hastings correction for spin and intertwiner proposals. A
            finite warmup need not be uniformly distributed. Vacuum states
            remain valid outcomes, at cutoff zero they are the only outcome.
            Initialization uses the default sampler family weights and stay
            probability, with equal mixing of common-sign and independent-sign
            loop shifts. It does not inherit a separately configured sampler.
        """
        del max_trials
        shape = batch_shape(size)
        out_dtype = self.dtype if dtype is None else np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if method is not None and method not in ("metropolis", "vacuum", "loop_warmup"):
            raise ValueError(
                "method must be 'metropolis', 'vacuum', 'loop_warmup', or None."
            )
        if n_steps is not None and (
            isinstance(n_steps, (bool, np.bool_))
            or not isinstance(n_steps, (int, np.integer))
            or n_steps < 0
        ):
            raise ValueError("n_steps must be a nonnegative integer or None.")

        if (
            method is None
            and self._exact_states is not None
            and self._exact_states.shape[0] > 0
        ):
            states = jnp.asarray(self._exact_states, dtype=out_dtype)
            ids = jax.random.randint(
                key, shape or (), 0, states.shape[0], dtype=jnp.int32
            )
            return states[ids]

        method = self.random_init if method is None else method
        if method == "vacuum" or self.two_j_max == 0:
            return jnp.zeros((*shape, self.size), dtype=out_dtype)

        steps = self.init_warmup if n_steps is None else n_steps
        steps = max(32, 4 * max(self.n_edges, 1)) if steps is None else int(steps)
        if method == "metropolis":
            flat = random_metropolis_warmup(self, key, count, steps)
        else:
            flat = random_loop_warmup(self, key, count, n_steps=steps)
        flat = flat.astype(out_dtype)
        return flat[0] if not shape else flat.reshape((*shape, self.size))


__all__ = ["SU2SamplingMixin"]
