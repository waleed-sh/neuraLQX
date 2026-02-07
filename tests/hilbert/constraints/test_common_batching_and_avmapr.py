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


def test_ensure_batchdim_single_and_batched(jnp):
    from neuralqx.hilbert.constraints.utils._common import ensure_batchdim

    x1 = jnp.asarray([1, 2, 3])
    xb, was_single = ensure_batchdim(x1)
    assert xb.shape == (1, 3)
    assert was_single is True

    x2 = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    xb2, was_single2 = ensure_batchdim(x2)
    assert xb2.shape == (2, 3)
    assert was_single2 is False


def test_avmapr_splits_keys_per_outer_axis_and_recurses(jax, jnp):
    from neuralqx.hilbert.constraints.utils._common import avmapr

    B, G, N = 3, 2, 4
    x = jnp.zeros((B, G, N), dtype=jnp.int32)

    def f(leaf, key):
        r = jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.int32)
        return leaf + r

    key = jax.random.PRNGKey(0)
    y = avmapr(f, x, key)

    assert y.shape == x.shape

    y_np = np.asarray(y)
    assert np.all(y_np == y_np[..., :1])

    flat_leaf_vals = y_np.reshape((-1, N))[:, 0]
    assert len(set(map(int, flat_leaf_vals.tolist()))) > 1
