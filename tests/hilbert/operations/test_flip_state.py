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


def test_flip_state_validates_number_of_edges(unconstrained_space, jax, jnp):
    from neuralqx.hilbert.operations.flip import flip_state

    space = unconstrained_space
    sigma = jnp.zeros((int(space.size),), dtype=space.dtype)
    key = jax.random.PRNGKey(0)

    with pytest.raises(ValueError):
        flip_state(space, sigma, key, number_of_edges=0)

    with pytest.raises(ValueError):
        flip_state(space, sigma, key, number_of_edges=int(space.size) + 1)


def test_flip_state_random_assignment_branch_is_correct(
    monkeypatch, unconstrained_space, jax, jnp
):
    from neuralqx.hilbert.operations import flip as flip_mod

    space = unconstrained_space
    N = int(space.size)
    sigma = jnp.arange(N, dtype=space.dtype)
    key = jax.random.PRNGKey(0)

    def fake_choice(key, a, shape, replace):
        assert a == N
        assert shape == (2,)
        assert replace is False
        return jnp.asarray([1, 4], dtype=jnp.int32)

    def fake_randint(key, shape, minval, maxval):
        assert shape == (2,)
        assert minval == 0
        return jnp.asarray([2, 0], dtype=jnp.int32)

    monkeypatch.setattr(flip_mod.jax.random, "choice", fake_choice)
    monkeypatch.setattr(flip_mod.jax.random, "randint", fake_randint)

    out = flip_mod.flip_state(space, sigma, key, number_of_edges=2, adjacency=False)
    out_np = np.asarray(out)

    expected = np.array(sigma)
    expected[1] = 1
    expected[4] = -1

    assert np.array_equal(out_np, expected)


def test_flip_state_adjacency_uses_mod_add_when_qrange_present(
    monkeypatch, u1_like_space, jax, jnp
):
    from neuralqx.hilbert.operations import flip as flip_mod

    space = u1_like_space
    N = int(space.size)
    sigma = jnp.zeros((N,), dtype=space.dtype)
    key = jax.random.PRNGKey(0)

    def fake_choice_indices(key, a, shape, replace):
        if isinstance(a, int):
            return jnp.asarray([0, 7], dtype=jnp.int32)
        return jnp.asarray([a[1], a[0]], dtype=a.dtype)

    monkeypatch.setattr(flip_mod.jax.random, "choice", fake_choice_indices)

    calls = {"n": 0}

    def fake_mod_add(x, deltas, *, q_min, q_max, step):
        calls["n"] += 1

        L = (q_max - q_min) // step + 1
        dig = (x - q_min) // step
        ddig = deltas // step
        out_dig = (dig + ddig) % L
        return q_min + out_dig * step

    monkeypatch.setattr(flip_mod, "mod_add", fake_mod_add)

    out = flip_mod.flip_state(space, sigma, key, number_of_edges=2, adjacency=True)
    assert calls["n"] >= 1

    out_np = np.asarray(out)
    assert out_np[0] == 1
    assert out_np[7] == -1


def test_flip_state_batch_splits_keys_independently(unconstrained_space, jax, jnp):
    from neuralqx.hilbert.operations.flip import flip_state

    space = unconstrained_space
    N = int(space.size)

    sigma = jnp.zeros((2, N), dtype=space.dtype)
    key = jax.random.PRNGKey(0)

    out = flip_state(space, sigma, key, number_of_edges=3, adjacency=False)
    out_np = np.asarray(out)

    assert out_np.shape == (2, N)
    assert not np.array_equal(out_np[0], out_np[1])


def test_flip_state_scope_is_ignored(unconstrained_space, jax, jnp):
    from neuralqx.hilbert.operations.flip import flip_state

    space = unconstrained_space
    sigma = jnp.zeros((int(space.size),), dtype=space.dtype)
    key = jax.random.PRNGKey(0)

    a = flip_state(
        space, sigma, key, number_of_edges=2, adjacency=False, scope="single"
    )
    b = flip_state(space, sigma, key, number_of_edges=2, adjacency=False, scope="all")
    assert np.array_equal(np.asarray(a), np.asarray(b))
