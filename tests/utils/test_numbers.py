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
import numbers as py_numbers

import numpy as np
import pytest


@pytest.fixture
def nmod():
    pytest.importorskip("jax")
    return importlib.import_module("neuralqx.utils.numbers")


@pytest.fixture
def jnp():
    pytest.importorskip("jax.numpy")
    import jax.numpy as jnp

    return jnp


@pytest.mark.parametrize("x", [0, 1, -3, 1.25, -0.0, 2 + 3j])
def test_is_scalar_like_python_numbers(nmod, x):
    assert nmod.is_scalar_like(x) is True


def test_is_scalar_like_numpy_scalar(nmod):
    assert nmod.is_scalar_like(np.float32(1.5)) is True
    assert nmod.is_scalar_like(np.int64(7)) is True


def test_is_scalar_like_numpy_0d_array(nmod):
    x = np.array(3.0)
    assert x.ndim == 0
    assert nmod.is_scalar_like(x) is True


def test_is_scalar_like_jax_0d_array(nmod, jnp):
    x = jnp.asarray(3.0).reshape(())
    assert int(x.ndim) == 0
    assert nmod.is_scalar_like(x) is True


def test_is_scalar_like_non_scalar_arrays_false(nmod, jnp):
    assert nmod.is_scalar_like(np.array([1, 2])) is False
    assert nmod.is_scalar_like(jnp.ones((2,))) is False


def test_is_scalar_like_object_with_ndim_0_true(nmod):
    class Weird:
        ndim = 0

    assert nmod.is_scalar_like(Weird()) is True


def test_is_scalar_like_object_with_ndim_none_false(nmod):
    class Weird:
        ndim = None

    assert nmod.is_scalar_like(Weird()) is False


def test_canonical_jax_dtype_none(nmod):
    assert nmod.canonical_jax_dtype(None) is None


def test_canonical_jax_dtype_from_dtype_objects(nmod, jnp):
    assert nmod.canonical_jax_dtype(np.dtype("float32")) == jnp.dtype("float32")
    assert nmod.canonical_jax_dtype(jnp.dtype("int32")) == jnp.dtype("int32")


def test_canonical_jax_dtype_from_hasattr_dtype(nmod, jnp):
    arr = np.arange(3, dtype=np.int32)
    assert nmod.canonical_jax_dtype(arr) == jnp.dtype(np.int32)

    x = np.float64(1.0)
    assert nmod.canonical_jax_dtype(x) == jnp.dtype(np.float64)


def test_canonical_jax_dtype_from_python_number(nmod, jnp):
    x = 1.25
    assert nmod.canonical_jax_dtype(x) == jnp.asarray(x).dtype


def test_canonical_jax_dtype_unknown_returns_none(nmod):
    class NoDType:
        pass

    assert nmod.canonical_jax_dtype(NoDType()) is None


def test__is_real_dtype(nmod, jnp):
    assert nmod._is_real_dtype(jnp.dtype("float32")) is True
    assert nmod._is_real_dtype(jnp.dtype("int32")) is True
    assert nmod._is_real_dtype(jnp.dtype("complex64")) is False


@pytest.mark.parametrize("x", [1, 2.5, 3 + 0j, np.float32(1.5), np.array(7)])
def test__to_0d_jax_array_returns_0d(nmod, jnp, x):
    arr = nmod._to_0d_jax_array(x)
    assert isinstance(arr, jnp.ndarray)
    assert int(arr.ndim) == 0
    assert tuple(arr.shape) == ()


def test__to_0d_jax_array_rejects_non_scalar(nmod):
    with pytest.raises(TypeError, match=r"Expected a scalar-like value"):
        nmod._to_0d_jax_array(np.array([1, 2, 3]))


def test_promote_constant_complex_op_promotes_real_constant(nmod, jnp):
    op_dt = jnp.dtype("complex64")
    tgt_dt, c = nmod.promote_constant_for_op_dtype(op_dt, 2.0)
    assert tgt_dt == op_dt
    assert int(c.ndim) == 0
    assert c.dtype == op_dt
    assert c.item() == 2.0 + 0.0j


def test_promote_constant_complex_op_casts_complex_constant(nmod, jnp):
    op_dt = (
        jnp.dtype("complex128")
        if hasattr(jnp, "complex128")
        else jnp.dtype("complex64")
    )
    tgt_dt, c = nmod.promote_constant_for_op_dtype(op_dt, 2.0 + 3.0j)
    assert tgt_dt == op_dt
    assert c.dtype == op_dt
    assert c.item() == pytest.approx((2.0 + 3.0j))


def test_promote_constant_real_op_casts_real_constant(nmod, jnp):
    op_dt = jnp.dtype("float32")
    tgt_dt, c = nmod.promote_constant_for_op_dtype(op_dt, 1.5)
    assert tgt_dt == op_dt
    assert c.dtype == op_dt
    assert c.item() == pytest.approx(1.5)


def test_promote_constant_real_op_drops_exact_zero_imag(nmod, jnp):
    op_dt = jnp.dtype("float32")
    tgt_dt, c = nmod.promote_constant_for_op_dtype(op_dt, 2.0 + 0.0j)
    assert tgt_dt == op_dt
    assert c.dtype == op_dt
    assert c.item() == pytest.approx(2.0)


def test_promote_constant_real_op_rejects_nonzero_imag_default_tolerance(nmod, jnp):
    op_dt = jnp.dtype("float32")
    with pytest.raises(
        TypeError, match=r"Cannot add complex constant with non-zero imaginary part"
    ):
        nmod.promote_constant_for_op_dtype(op_dt, 2.0 + 1e-15j)


def test_promote_constant_real_op_accepts_tiny_imag_with_atol(nmod, jnp):
    op_dt = jnp.dtype("float32")
    tgt_dt, c = nmod.promote_constant_for_op_dtype(op_dt, 2.0 + 1e-15j, atol=1e-12)
    assert tgt_dt == op_dt
    assert c.dtype == op_dt
    assert c.item() == pytest.approx(2.0)


def test_promote_constant_requires_scalar_like(nmod, jnp):
    op_dt = jnp.dtype("float32")
    with pytest.raises(TypeError, match=r"Expected a scalar-like value"):
        nmod.promote_constant_for_op_dtype(op_dt, np.array([1.0, 2.0]))


def test_hermitian_flag_false_if_op_not_hermitian(nmod, jnp):
    assert (
        nmod.hermitian_flag_for_sum_with_scalar(False, jnp.asarray(1.0).reshape(()))
        is False
    )
    assert (
        nmod.hermitian_flag_for_sum_with_scalar(
            False, jnp.asarray(1.0 + 0.0j).reshape(())
        )
        is False
    )


def test_hermitian_flag_true_for_real_const_when_op_hermitian(nmod, jnp):
    assert (
        nmod.hermitian_flag_for_sum_with_scalar(True, jnp.asarray(3.0).reshape(()))
        is True
    )


def test_hermitian_flag_complex_const_exact_zero_imag_ok(nmod, jnp):
    c = jnp.asarray(1.0 + 0.0j).reshape(())
    assert nmod.hermitian_flag_for_sum_with_scalar(True, c) is True


def test_hermitian_flag_complex_const_nonzero_imag_fails_default(nmod, jnp):
    c = jnp.asarray(1.0 + 1e-15j).reshape(())
    assert nmod.hermitian_flag_for_sum_with_scalar(True, c) is False


def test_hermitian_flag_complex_const_tiny_imag_ok_with_atol(nmod, jnp):
    c = jnp.asarray(1.0 + 1e-15j).reshape(())
    assert nmod.hermitian_flag_for_sum_with_scalar(True, c, atol=1e-12) is True


def test_to_python_scalar_0d_ok(nmod, jnp):
    x = jnp.asarray(7.5).reshape(())
    out = nmod.to_python_scalar(x)
    assert isinstance(out, py_numbers.Number)
    assert out == pytest.approx(7.5)


def test_to_python_scalar_rejects_non_0d(nmod, jnp):
    x = jnp.asarray([1.0, 2.0])
    with pytest.raises(TypeError, match=r"to_python_scalar expects a 0-d array"):
        nmod.to_python_scalar(x)


def test_promote_then_hermitian_pipeline(nmod, jnp):
    op_dt = jnp.dtype("float32")
    _, c = nmod.promote_constant_for_op_dtype(op_dt, 2.0 + 0.0j)
    assert nmod.hermitian_flag_for_sum_with_scalar(True, c) is True
