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


import importlib
import numpy as np
import pytest


@pytest.fixture
def jmod():
    pytest.importorskip("numba")
    pytest.importorskip("jax")
    return importlib.import_module("neuralqx.utils.jit")


@pytest.fixture
def jnp():
    pytest.importorskip("jax.numpy")
    import jax.numpy as jnp

    return jnp


@pytest.fixture
def jax():
    pytest.importorskip("jax")
    import jax

    return jax


def mod_add_ref(m, n, q_min, q_max, step=1):
    span = q_max - q_min + step
    return ((m + n - q_min) % span) + q_min


def mod_sum_ref(values, q_min, q_max, step=1):
    span = q_max - q_min + step
    s = 0
    for x in values:
        s = ((s + x - q_min) % span) + q_min
    return s


def test_mod_add_njit_scalar_wrap_int(jmod):
    out = jmod.mod_add_njit(2, 2, -2, 2, 1)

    assert out == -1

    out2 = jmod.mod_add_njit(-2, -1, -2, 2, 1)
    assert out2 == 2


def test_mod_add_njit_array_plus_scalar_matches_reference(jmod):
    q_min, q_max, step = -2, 2, 1
    m = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    out = jmod.mod_add_njit(m, 2, q_min, q_max, step)
    expected = mod_add_ref(m, 2, q_min, q_max, step)
    assert np.array_equal(out, expected)

    assert out.min() >= q_min
    assert out.max() <= q_max


def test_mod_add_njit_step_changes_span(jmod):
    q_min, q_max, step = -4, 4, 2
    out = jmod.mod_add_njit(4, 4, q_min, q_max, step)
    expected = mod_add_ref(4, 4, q_min, q_max, step)
    assert out == expected
    assert out == -2


def test_mod_add_njit_vectorized_random_regression(jmod):
    rng = np.random.default_rng(0)
    q_min, q_max, step = -3, 3, 1
    m = rng.integers(-50, 51, size=(17,), dtype=np.int64)
    n = rng.integers(-50, 51, size=(17,), dtype=np.int64)

    out = jmod.mod_add_njit(m, n, q_min, q_max, step)
    expected = mod_add_ref(m, n, q_min, q_max, step)

    assert np.array_equal(out, expected)
    assert out.min() >= q_min
    assert out.max() <= q_max


def test_mod_sum_jax_requires_cutoff_or_bounds(jmod, jnp):
    vals = jnp.array([1, 2, 3], dtype=jnp.int32)
    with pytest.raises(
        ValueError, match=r"Provide either `cutoff` or \(`q_min`, `q_max`\)"
    ):
        jmod.mod_sum_jax(vals)


def test_mod_sum_jax_cutoff_branch_correct_and_dtype_preserved(jmod, jnp):
    vals = jnp.array([1, 2, 3], dtype=jnp.int32)

    out = jmod.mod_sum_jax(vals, cutoff=2)

    expected = mod_sum_ref([1, 2, 3], -2, 2, 1)
    assert int(out) == expected
    assert out.dtype == vals.dtype
    assert -2 <= int(out) <= 2


def test_mod_sum_jax_explicit_bounds_override_cutoff(jmod, jnp):
    vals = jnp.array([1, 2, 3], dtype=jnp.int32)

    out = jmod.mod_sum_jax(vals, q_min=-1, q_max=1, cutoff=999, step=1)
    expected = mod_sum_ref([1, 2, 3], -1, 1, 1)

    assert int(out) == expected
    assert -1 <= int(out) <= 1


def test_mod_sum_jax_step_effect(jmod, jnp):
    vals = jnp.array([2, 2, 2], dtype=jnp.int32)
    out = jmod.mod_sum_jax(vals, q_min=-4, q_max=4, step=2)
    expected = mod_sum_ref([2, 2, 2], -4, 4, 2)

    assert int(out) == expected
    assert -4 <= int(out) <= 4


def test_mod_sum_jax_random_matches_reference(jmod, jnp):
    rng = np.random.default_rng(0)
    q_min, q_max, step = -3, 3, 1
    data = rng.integers(-50, 51, size=(25,), dtype=np.int64).tolist()

    vals = jnp.array(data, dtype=jnp.int32)
    out = jmod.mod_sum_jax(vals, q_min=q_min, q_max=q_max, step=step)
    expected = mod_sum_ref(data, q_min, q_max, step)

    assert int(out) == expected
    assert q_min <= int(out) <= q_max
    assert out.dtype == vals.dtype


def test_mod_sum_jax_empty_values_returns_zero_of_dtype(jmod, jnp):
    vals = jnp.array([], dtype=jnp.int32)
    out = jmod.mod_sum_jax(vals, cutoff=2)
    assert int(out) == 0
    assert out.dtype == vals.dtype


def test_mod_sum_jax_is_jittable(jmod, jnp, jax):
    vals = jnp.array([1, 2, 3, -4, 5], dtype=jnp.int32)

    f = jax.jit(lambda v: jmod.mod_sum_jax(v, cutoff=3))
    out_jit = f(vals)
    out_eager = jmod.mod_sum_jax(vals, cutoff=3)

    assert int(out_jit) == int(out_eager)
    assert out_jit.dtype == out_eager.dtype
