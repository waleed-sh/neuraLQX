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


def test_dispatch_random_state_matches_core_and_is_valid(constrained_core_small, jax):

    from neuralqx.hilbert.u1._dispatch_ops import random_state as dispatch_random_state

    core = constrained_core_small
    out = dispatch_random_state(core, jax.random.PRNGKey(0), size=5)
    assert out.shape == (5, int(core.size))
    assert bool(np.asarray(core.check_states(out)).all()) is True


def test_dispatch_flip_state_preserves_invariance(constrained_core_small, jax):

    from neuralqx.hilbert.u1._dispatch_ops import flip_state as dispatch_flip_state

    core = constrained_core_small
    sigma = core.random_state(jax.random.PRNGKey(0), size=3)

    out = dispatch_flip_state(
        core,
        sigma,
        jax.random.PRNGKey(1),
        number_of_edges=2,
        adjacency=True,
        scope="single",
    )
    assert out.shape == sigma.shape
    assert bool(np.asarray(core.check_states(out)).all()) is True
