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

from plum import dispatch

import jax
import jax.numpy as jnp

from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace


@dispatch
def random_state(
    space: AbstractHilbertSpace,
    key: jax.Array,
    *,
    size: int = 1,
) -> jax.Array:
    """
    Default random state generator for unconstrained Hilbert spaces.

    This function draws configurations by sampling each site independently and uniformly from the
    discrete set of allowed local basis states. It is intended as a generic, "works everywhere"
    implementation for *unconstrained* spaces, where i.i.d. sampling does not violate any global
    constraints.

    For constrained spaces (e.g. gauge-fixed / gauge-invariant subspaces), i.i.d. local sampling
    generally produces invalid configurations. Those cores should override random-state generation
    with a constraint-aware routine (typically: sample the free degrees of freedom, then
    deterministically reimpose the constraints in topological order).

    Note on extensibility: this function is registered via `plum.dispatch`, so advanced users may
    add alternative `random_state` implementations for their own subclasses of
    `AbstractHilbertSpace` without modifying this module (plugin-style extension). However, the
    preferred pattern inside neuraLQX is to implement constraint-aware sampling directly on the
    concrete core class when the constraints are intrinsic to that space.

    :param space: The Hilbert space core defining the local basis states, size, and dtype.
    :param key: A JAX key used to generate random samples.
    :param size: The number of configurations (batch size) to generate.
    :return: A batch of random states with shape ``(size, space.size)`` and dtype
      ``space.dtype``.
    """

    qn = jnp.asarray(space.allowed_basis_states.all_states())
    ridx = jax.random.randint(key, (size, int(space.size)), 0, qn.shape[0])
    return qn[ridx].astype(space.dtype)
