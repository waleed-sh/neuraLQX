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


def _sample_uniform_domain(jax, jnp, values, shape, dtype):

    values = jnp.asarray(values, dtype=dtype)
    key = jax.random.PRNGKey(0)
    idx = jax.random.randint(key, shape, 0, values.shape[0])
    return values[idx]


def test_random_state_is_valid(constrained_core_small, jax, jnp):
    core = constrained_core_small
    B = 8
    sigma = core.random_state(jax.random.PRNGKey(123), size=B)

    assert sigma.shape == (B, int(core.size))
    ok = np.asarray(core.check_states(sigma))
    assert ok.shape == (B,)
    assert bool(ok.all()) is True


def test_check_states_rejects_corrupted_known_slave(constrained_core_small, jax, jnp):

    core = constrained_core_small
    sigma = core.random_state(jax.random.PRNGKey(7), size=4)
    assert bool(np.asarray(core.check_states(sigma)).all()) is True

    c0 = core.constraints_base[0]
    lhs = int(c0.lhs)

    v = core.view(sigma)
    step = int(core.q_step)

    v_bad = v.at[0, 0, lhs].set(v[0, 0, lhs] + jnp.asarray(step, dtype=v.dtype))
    bad = core.flatten(v_bad)

    ok = np.asarray(core.check_states(bad))
    assert ok.shape == (4,)
    assert bool(ok[0]) is False


def test_reimpose_projects_to_valid_and_is_idempotent(constrained_core_small, jax, jnp):
    core = constrained_core_small
    B = 6
    N = int(core.size)
    values = core.allowed_basis_states.all_states()

    sigma_raw = _sample_uniform_domain(jax, jnp, values, (B, N), dtype=jnp.int64)

    sigma_proj = core.reimpose_gauge_fixing(sigma_raw)
    ok = np.asarray(core.check_states(sigma_proj))
    assert bool(ok.all()) is True

    sigma_proj2 = core.reimpose_gauge_fixing(sigma_proj)
    assert np.array_equal(np.asarray(sigma_proj2), np.asarray(sigma_proj))


def test_reimpose_preserves_free_edges(constrained_core_small, jax, jnp):
    core = constrained_core_small
    B = 5
    N = int(core.size)
    values = core.allowed_basis_states.all_states()

    sigma_raw = _sample_uniform_domain(jax, jnp, values, (B, N), dtype=jnp.int64)
    v_raw = core.view(sigma_raw)

    sigma_proj = core.reimpose_gauge_fixing(sigma_raw)
    v_proj = core.view(sigma_proj)

    free = jnp.asarray(core.gauge_fixing.free, dtype=jnp.int32)

    assert np.array_equal(np.asarray(v_proj[:, :, free]), np.asarray(v_raw[:, :, free]))


def test_reimpose_overwrites_slaves_deterministically(constrained_core_small, jax, jnp):
    core = constrained_core_small
    sigma = core.random_state(jax.random.PRNGKey(0), size=3)
    assert bool(np.asarray(core.check_states(sigma)).all()) is True

    slave0 = int(core.gauge_fixing.slave_topo[0])
    v = core.view(sigma)

    step = int(core.q_step)
    v_corrupt = v.at[0, 0, slave0].set(
        v[0, 0, slave0] + jnp.asarray(step, dtype=v.dtype)
    )
    sigma_corrupt = core.flatten(v_corrupt)

    sigma_restored = core.reimpose_gauge_fixing(sigma_corrupt)
    assert np.array_equal(np.asarray(sigma_restored), np.asarray(sigma))


def test_view_flatten_roundtrip(constrained_core_small, jax, jnp):
    core = constrained_core_small
    sigma = core.random_state(jax.random.PRNGKey(99), size=4)

    v = core.view(sigma)
    sigma2 = core.flatten(v)
    assert np.array_equal(np.asarray(sigma2), np.asarray(sigma))


def test_jit_smoke_check_states_and_reimpose(constrained_core_small, jax, jnp):

    core = constrained_core_small
    sigma = core.random_state(jax.random.PRNGKey(5), size=3)

    f_check = jax.jit(core.check_states)
    out = f_check(sigma)
    assert out.shape == (3,)
    assert bool(np.asarray(out).all()) is True

    values = core.allowed_basis_states.all_states()
    raw = _sample_uniform_domain(jax, jnp, values, sigma.shape, dtype=sigma.dtype)

    f_reimpose = jax.jit(core.reimpose_gauge_fixing)
    proj = f_reimpose(raw)
    assert bool(np.asarray(core.check_states(proj)).all()) is True
