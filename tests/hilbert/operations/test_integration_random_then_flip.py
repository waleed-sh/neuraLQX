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


def test_random_then_flip_preserves_value_domain(unconstrained_space, jax, jnp):
    from neuralqx.hilbert.u1.operations.random import random_state
    from neuralqx.hilbert.u1.operations.flip import flip_state

    space = unconstrained_space
    N = int(space.size)
    allowed = set(
        map(int, np.asarray(space.allowed_basis_states.all_states()).tolist())
    )

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)

    s = random_state(space, k1, size=4)
    s2 = flip_state(space, s, k2, number_of_edges=3, adjacency=False)

    s2_np = np.asarray(s2)
    assert s2_np.shape == (4, N)
    assert set(map(int, s2_np.reshape(-1).tolist())).issubset(allowed)
