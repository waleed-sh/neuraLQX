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
def amod():
    pytest.importorskip("jax")
    return importlib.import_module("neuralqx.utils.misc.arithmetic")


@pytest.fixture
def jax():
    pytest.importorskip("jax")
    import jax

    return jax


@pytest.fixture
def jnp():
    pytest.importorskip("jax.numpy")
    import jax.numpy as jnp

    return jnp


def mod_add_ref(m, n, q_min, q_max, step=1):
    span = q_max - q_min + step
    return ((m + n - q_min) % span) + q_min


def mod_sum_ref(arr, q_min, q_max, step=1):
    s = 0
    for e in arr:
        s = mod_add_ref(s, e, q_min, q_max, step)
    return s


def test_mod_add_requires_bounds_or_cutoff(amod):
    with pytest.raises(
        ValueError, match=r"Must provide either `cutoff or \(`q_min`, `q_max`\)\."
    ):
        amod.mod_add(1, 2)


def test_mod_add_cutoff_fallback_domain(amod):
    out = amod.mod_add(2, 2, cutoff=2)
    assert int(out) == -1


def test_mod_add_explicit_bounds_override_cutoff(amod):
    out = amod.mod_add(2, 2, q_min=-1, q_max=1, cutoff=99)
    assert int(out) == 1


def test_mod_add_step_affects_span(amod):
    out = amod.mod_add(4, 4, q_min=-4, q_max=4, step=2)

    assert int(out) == -2


def test_mod_add_numpy_array_broadcast_matches_reference(amod):
    q_min, q_max, step = -2, 2, 1
    m = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    out = amod.mod_add(m, 2, q_min=q_min, q_max=q_max, step=step)
    expected = mod_add_ref(m, 2, q_min, q_max, step)
    assert isinstance(out, np.ndarray)

    assert out.dtype == m.dtype
    assert np.array_equal(out, expected)


def test_mod_add_integer_scalar_returns_integer_typed_jax_scalar(amod, jnp):

    out = amod.mod_add(2, 2, cutoff=2)
    assert hasattr(out, "dtype")
    assert jnp.issubdtype(out.dtype, jnp.integer)
    assert int(out) == -1


def test_mod_add_float_inputs_do_not_force_integer_cast(amod):
    out = amod.mod_add(2.0, 2.0, cutoff=2)

    assert float(out) == pytest.approx(-1.0)


def test_mod_add_random_regression_against_reference(amod):
    rng = np.random.default_rng(0)
    q_min, q_max, step = -3, 3, 1
    m = rng.integers(-100, 101, size=(50,), dtype=np.int64)
    n = rng.integers(-100, 101, size=(50,), dtype=np.int64)

    out = amod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)
    expected = mod_add_ref(m, n, q_min, q_max, step)
    assert np.array_equal(out, expected)
    assert out.min() >= q_min
    assert out.max() <= q_max


def test_mod_sum_matches_reference_with_cutoff(amod):
    arr = [1, 2, 3, -4, 5]
    out = amod.mod_sum(arr, cutoff=3)
    expected = mod_sum_ref(arr, -3, 3, 1)
    assert int(out) == expected


def test_mod_sum_matches_reference_with_bounds_and_step(amod):
    arr = [2, 2, 2]
    out = amod.mod_sum(arr, q_min=-4, q_max=4, step=2)
    expected = mod_sum_ref(arr, -4, 4, 2)
    assert int(out) == expected


def test_mod_sum_empty_returns_zero(amod):
    assert amod.mod_sum([], cutoff=2) == 0


@pytest.mark.parametrize(
    "x, expected", [(-1, -1), (-1e-12, -1), (0, 1), (1e-12, 1), (5, 1)]
)
def test_get_sgn(amod, x, expected):
    assert amod.get_sgn(x) == expected


@pytest.mark.parametrize(
    "s, expected", [("-x", -1), ("-", -1), ("x", 1), ("+x", 1), ("", 1), ("  -x", 1)]
)
def test_get_signed_value(amod, s, expected):
    assert amod.get_signed_value(s) == expected


@pytest.mark.parametrize(
    "n, expected", [(0, 1), (1, 1), (2, 2), (3, 6), (5, 120), (8, 40320)]
)
def test_factorial_small(amod, n, expected):
    assert amod.factorial(n) == expected


def test_factorial_negative_should_raise_future(amod):
    with pytest.raises(ValueError):
        amod.factorial(-1)


def test_generate_plus_minus_one_values_and_shape(amod, jax, jnp):
    key = jax.random.PRNGKey(0)

    x0 = amod.generate_plus_minus_one(key, shape=())
    assert tuple(getattr(x0, "shape", ())) == ()
    assert int(x0) in (-1, 1)

    key2 = jax.random.PRNGKey(123)
    x = amod.generate_plus_minus_one(key2, shape=(100,))
    vals = np.array(x)
    assert vals.shape == (100,)
    assert set(np.unique(vals)).issubset({-1, 1})


def test_generate_plus_minus_one_deterministic_for_same_key(amod, jax):
    key = jax.random.PRNGKey(0)
    a = amod.generate_plus_minus_one(key, shape=(20,))
    b = amod.generate_plus_minus_one(key, shape=(20,))

    assert np.array_equal(np.array(a), np.array(b))


def test_plus_key_modifies_all_but_last_tuple(amod):
    t = (1, 2, 3, "sentinel")
    out = amod.plus_key(t, 10)
    assert out == (11, 12, 13, "sentinel")


def test_minus_key_modifies_all_but_last_list(amod):
    t = [1.5, -2.0, 7.0, "keep"]
    out = amod.minus_key(t, 0.5)
    assert out == (1.0, -2.5, 6.5, "keep")
