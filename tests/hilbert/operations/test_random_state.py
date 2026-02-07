#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np
import pytest


def test_random_state_shape_dtype_and_membership(unconstrained_space, jax, jnp):
    from neuralqx.hilbert.operations.random import random_state

    space = unconstrained_space
    key = jax.random.PRNGKey(0)

    out = random_state(space, key, size=5)
    assert out.shape == (5, int(space.size))
    assert out.dtype == space.dtype

    allowed = set(
        map(int, np.asarray(space.allowed_basis_states.all_states()).tolist())
    )
    out_vals = set(map(int, np.asarray(out).reshape(-1).tolist()))
    assert out_vals.issubset(allowed)


def test_random_state_deterministic_for_same_key(unconstrained_space, jax):
    from neuralqx.hilbert.operations.random import random_state

    space = unconstrained_space
    key = jax.random.PRNGKey(123)

    a = random_state(space, key, size=3)
    b = random_state(space, key, size=3)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_random_state_differs_for_different_keys(unconstrained_space, jax):
    from neuralqx.hilbert.operations.random import random_state

    space = unconstrained_space
    a = random_state(space, jax.random.PRNGKey(0), size=3)
    b = random_state(space, jax.random.PRNGKey(1), size=3)
    assert not np.array_equal(np.asarray(a), np.asarray(b))
