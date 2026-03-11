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


def test_ensure_2d_states_and_restore_roundtrip_various_ranks(rng):
    from neuralqx.hilbert.u1.index.utils._array_normalise import (
        _ensure_2d_states,
        _restore_states_shape,
    )

    N = 7

    shapes = [
        (N,),
        (3, N),
        (2, 3, N),
        (1, 2, 3, N),
    ]

    for shp in shapes:
        x = rng.integers(0, 5, size=shp, dtype=np.int64)
        x2d, was_single = _ensure_2d_states(x)

        assert x2d.ndim == 2
        assert x2d.shape[1] == N
        assert was_single is (len(shp) == 1)

        x_back = _restore_states_shape(x2d, shp)
        assert x_back.shape == shp
        np.testing.assert_array_equal(x_back, x)


def test_ensure_1d_numbers_scalar_and_batches(rng):
    from neuralqx.hilbert.u1.index.utils._array_normalise import _ensure_1d_numbers

    a, was = _ensure_1d_numbers(5)
    assert a.shape == (1,)
    assert was is True
    assert int(a[0]) == 5

    b = np.array(7, dtype=np.int64)
    a, was = _ensure_1d_numbers(b)
    assert a.shape == (1,)
    assert was is True
    assert int(a[0]) == 7

    arr = rng.integers(0, 10, size=(2, 3), dtype=np.int64)
    a, was = _ensure_1d_numbers(arr)
    assert a.ndim == 1
    assert a.shape == (6,)
    assert was is False
    np.testing.assert_array_equal(a, arr.reshape(-1))


def test_as_numpy_accepts_lists_tuples_and_arrays():
    from neuralqx.hilbert.u1.index.utils._array_normalise import _as_numpy

    assert isinstance(_as_numpy([1, 2, 3]), np.ndarray)
    assert isinstance(_as_numpy((1, 2)), np.ndarray)
    x = np.array([1, 2])
    y = _as_numpy(x)
    np.testing.assert_array_equal(y, x)
