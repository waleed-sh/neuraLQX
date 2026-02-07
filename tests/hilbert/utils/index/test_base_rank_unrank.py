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


def test_choose_return_dtype_auto_boundaries():
    from neuralqx.hilbert.utils.index._base import _choose_return_dtype

    assert _choose_return_dtype(base=2, n_digits=63, requested="auto") == np.int64
    assert _choose_return_dtype(base=2, n_digits=64, requested="auto") == np.dtype(
        object
    )

    assert _choose_return_dtype(base=1, n_digits=10, requested="auto") == np.int64


def test_choose_return_dtype_requested_values_and_errors():
    from neuralqx.hilbert.utils.index._base import _choose_return_dtype

    assert _choose_return_dtype(3, 5, "int64") == np.int64
    assert _choose_return_dtype(3, 5, "i8") == np.int64
    assert _choose_return_dtype(3, 5, "object") == np.dtype(object)
    assert _choose_return_dtype(3, 5, "python") == np.dtype(object)

    with pytest.raises(ValueError):
        _choose_return_dtype(3, 5, "nonsense-dtype")


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("base", [2, 3, 5])
@pytest.mark.parametrize("N", [0, 1, 2, 5, 7])
def test_rank_unrank_roundtrip_int64_random(order, base, N):
    from neuralqx.hilbert.utils.index._base import (
        _rank_digits_base,
        _unrank_numbers_base,
    )

    rng = np.random.default_rng(1)
    B = 11
    digits = rng.integers(0, base, size=(B, N), dtype=np.int64)

    nums = _rank_digits_base(digits, base=base, order=order, out_dtype=np.int64)
    assert nums.shape == (B,)
    assert nums.dtype == np.int64

    digits2 = _unrank_numbers_base(
        nums, base=base, n_digits=N, order=order, digits_dtype=np.int64
    )
    np.testing.assert_array_equal(digits2, digits)


def test_rank_order_convention_matches_manual_formula():
    from neuralqx.hilbert.utils.index._base import _rank_digits_base

    base = 10
    digits = np.array([[1, 2, 3]], dtype=np.int64)

    nC = _rank_digits_base(digits, base=base, order="C", out_dtype=np.int64)[0]
    assert int(nC) == 123

    nF = _rank_digits_base(digits, base=base, order="F", out_dtype=np.int64)[0]
    assert int(nF) == 321


def test_object_dtype_bigint_roundtrip():
    from neuralqx.hilbert.utils.index._base import (
        _rank_digits_base,
        _unrank_numbers_base,
    )

    rng = np.random.default_rng(2)
    base = 10
    N = 30
    B = 3

    digits = rng.integers(0, base, size=(B, N), dtype=np.int64)
    nums = _rank_digits_base(digits, base=base, order="C", out_dtype=np.dtype(object))
    assert nums.dtype == object

    digits2 = _unrank_numbers_base(
        nums, base=base, n_digits=N, order="C", digits_dtype=np.int64
    )
    np.testing.assert_array_equal(digits2, digits)


def test_unrank_rejects_negative_and_overflow():
    from neuralqx.hilbert.utils.index._base import _unrank_numbers_base

    with pytest.raises(ValueError, match=">= 0"):
        _unrank_numbers_base(
            np.array([-1], dtype=np.int64), base=2, n_digits=3, order="C"
        )

    with pytest.raises(ValueError, match="out of range"):
        _unrank_numbers_base(
            np.array([8], dtype=np.int64), base=2, n_digits=3, order="C"
        )
