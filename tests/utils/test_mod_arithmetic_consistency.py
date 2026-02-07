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
def jitmod():
    pytest.importorskip("numba")
    pytest.importorskip("jax")
    return importlib.import_module("neuralqx.utils.jit")


@pytest.fixture
def armod():
    pytest.importorskip("jax")
    return importlib.import_module("neuralqx.utils.misc.arithmetic")


@pytest.fixture
def jnp():
    pytest.importorskip("jax.numpy")
    import jax.numpy as jnp

    return jnp


def _as_py(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def _assert_same_value(a, b, *, atol=1e-12):
    assert _as_py(a) == pytest.approx(_as_py(b), abs=atol)


@pytest.mark.parametrize(
    "q_min,q_max,step",
    [
        (-2, 2, 1),
        (-3, 3, 1),
        (-4, 4, 2),
        (0, 5, 1),
    ],
)
@pytest.mark.parametrize(
    "m,n",
    [
        (2, 2),
        (-2, -1),
        (0, 0),
        (10, -25),
    ],
)
def test_mod_add_scalar_consistency_integer_domain(
    jitmod, armod, q_min, q_max, step, m, n
):
    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)
    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)
    _assert_same_value(out_jit, out_py)


@pytest.mark.parametrize(
    "q_min,q_max,step",
    [
        (-1.5, 2.5, 0.5),
        (-2.0, 2.0, 0.5),
        (0.0, 3.0, 0.5),
    ],
)
@pytest.mark.parametrize(
    "m,n",
    [
        (2.0, 2.0),
        (-2.0, -1.0),
        (1.25, -3.5),
        (10.0, -25.0),
    ],
)
def test_mod_add_scalar_consistency_fractional_domain_float_inputs(
    jitmod, armod, q_min, q_max, step, m, n
):
    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)
    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)
    _assert_same_value(out_jit, out_py, atol=1e-12)


def test_mod_add_numpy_array_consistency_integer_domain_and_dtype(jitmod, armod):
    q_min, q_max, step = -3, 3, 1
    m = np.array([-10, -3, -1, 0, 1, 3, 10], dtype=np.int64)
    n = np.array([5, 5, 5, 5, 5, 5, 5], dtype=np.int64)

    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)
    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)

    assert isinstance(out_jit, np.ndarray)
    assert isinstance(out_py, np.ndarray)
    assert np.array_equal(out_jit, out_py)
    assert out_jit.dtype == np.int64
    assert out_py.dtype == np.int64


def test_mod_add_numpy_array_consistency_fractional_domain_float(jitmod, armod):
    q_min, q_max, step = -1.5, 2.5, 0.5
    m = np.array([-2.0, -1.5, -1.0, 0.0, 1.5, 2.0], dtype=np.float64)
    n = 2.0

    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)
    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)

    assert np.allclose(out_jit, out_py, rtol=0, atol=1e-12)


def test_mod_add_cutoff_matches_jit_when_bounds_equivalent(jitmod, armod):
    cutoff = 4
    q_min, q_max, step = -cutoff, cutoff, 1

    m, n = 7, -3
    out_cutoff = armod.mod_add(m, n, cutoff=cutoff, step=step)
    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)

    _assert_same_value(out_cutoff, out_jit)


def test_mod_add_integer_inputs_fractional_domain_diverges_due_to_casting(
    jitmod, armod
):

    q_min, q_max, step = -1.5, 2.5, 0.5

    m, n = 2, 2

    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)

    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)

    assert _as_py(out_jit) == _as_py(out_py)


@pytest.mark.parametrize(
    "cutoff, data",
    [
        (2, [1, 2, 3]),
        (3, [1, 2, 3, -4, 5]),
        (4, [10, -9, 8, -7, 6, -5, 4]),
    ],
)
def test_mod_sum_consistency_cutoff(jitmod, armod, jnp, cutoff, data):
    vals = jnp.array(data, dtype=jnp.int32)
    out_scan = jitmod.mod_sum_jax(vals, cutoff=cutoff)
    out_loop = armod.mod_sum(data, cutoff=cutoff)

    _assert_same_value(out_scan, out_loop)
    assert out_scan.dtype == vals.dtype


@pytest.mark.parametrize(
    "q_min,q_max,step,data",
    [
        (-3, 3, 1, [1, 2, 3, -4, 5]),
        (-4, 4, 2, [2, 2, 2, 2]),
        (0, 5, 1, [6, 7, -1, 2]),
    ],
)
def test_mod_sum_consistency_explicit_bounds_integer(
    jitmod, armod, jnp, q_min, q_max, step, data
):
    vals = jnp.array(data, dtype=jnp.int32)
    out_scan = jitmod.mod_sum_jax(vals, q_min=q_min, q_max=q_max, step=step)
    out_loop = armod.mod_sum(data, q_min=q_min, q_max=q_max, step=step)

    _assert_same_value(out_scan, out_loop)


def test_mod_sum_missing_domain_raises_valueerror_both(jitmod, armod, jnp):
    vals = jnp.array([1, 2, 3], dtype=jnp.int32)

    with pytest.raises(
        ValueError, match=r"Provide either `cutoff` or \(`q_min`, `q_max`\)"
    ):
        jitmod.mod_sum_jax(vals)

    with pytest.raises(
        ValueError, match=r"Must provide either `cutoff or \(`q_min`, `q_max`\)\."
    ):
        armod.mod_sum([1, 2, 3])


@pytest.mark.parametrize(
    "q_min,q_max,step,m,n",
    [
        (-1.5, 2.5, 0.5, 2, 2),
        (-1.5, 2.5, 0.5, -2, -1),
        (-2.0, 2.0, 0.5, 7, -3),
        (0.0, 3.0, 0.5, 10, -25),
    ],
)
def test_mod_add_integer_inputs_fractional_domain_should_match_in_future(
    jitmod, armod, q_min, q_max, step, m, n
):

    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)
    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)
    _assert_same_value(out_jit, out_py, atol=1e-12)


def test_mod_add_integer_numpy_array_fractional_domain_should_match_in_future(
    jitmod, armod
):
    q_min, q_max, step = -1.5, 2.5, 0.5
    m = np.array([-2, -1, 0, 1, 2], dtype=np.int64)

    n = 2

    out_jit = jitmod.mod_add_njit(m, n, q_min, q_max, step)
    out_py = armod.mod_add(m, n, q_min=q_min, q_max=q_max, step=step)

    assert np.allclose(out_jit, out_py, rtol=0, atol=1e-12)


def test_mod_add_integer_scalar_fractional_domain_dtype_and_value_should_align_in_future(
    jitmod, armod
):
    q_min, q_max, step = -1.5, 2.5, 0.5
    out_jit = jitmod.mod_add_njit(2, 2, q_min, q_max, step)
    out_py = armod.mod_add(2, 2, q_min=q_min, q_max=q_max, step=step)

    _assert_same_value(out_jit, out_py, atol=1e-12)

    assert float(out_py) == pytest.approx(float(out_jit), abs=1e-12)
