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


def test_infer_netket_site_order_is_consistent_with_netket(static_range_small, nk):
    from neuralqx.hilbert.utils.index._netket import _infer_netket_site_order

    order = _infer_netket_site_order(static_range_small)
    assert order in ("C", "F")

    h2 = nk.hilbert.HomogeneousHilbert(local_states=static_range_small, N=2)
    vals = np.asarray(static_range_small.all_states())
    L = int(vals.shape[0])

    s0, s1 = vals[0], vals[1]
    n01 = int(np.asarray(h2.states_to_numbers(np.asarray([[s0, s1]])))[0])
    n10 = int(np.asarray(h2.states_to_numbers(np.asarray([[s1, s0]])))[0])

    if order == "C":
        assert (n10, n01) == (L, 1)
    else:
        assert (n10, n01) == (1, L)


def test_infer_netket_site_order_truncates_giant_local_states(nk):
    from netket.utils import StaticRange
    from neuralqx.hilbert.utils.index._netket import _infer_netket_site_order

    big = StaticRange(start=0, step=1, length=2001)
    small = StaticRange(start=0, step=1, length=100)

    assert _infer_netket_site_order(big) == _infer_netket_site_order(small)


def test_local_values_to_digits_and_back_roundtrip(static_range_small):
    from neuralqx.hilbert.utils.index._netket import (
        _local_values_to_digits,
        _local_digits_to_values,
    )

    vals = np.asarray(static_range_small.all_states())
    B, N = 4, 6

    x = np.tile(vals[:N], (B, 1))
    d = _local_values_to_digits(static_range_small, x)
    x2 = _local_digits_to_values(static_range_small, d)

    np.testing.assert_array_equal(x2, x)


def test_local_values_to_digits_staticrange_affine_mapping_no_validation(
    static_range_small,
):

    from neuralqx.hilbert.utils.index._netket import _local_values_to_digits

    x = np.array([[0, 1, 2]], dtype=np.int64)
    d = _local_values_to_digits(static_range_small, x)
    np.testing.assert_array_equal(d, x)


def test_local_values_to_digits_out_of_basis_staticrange_wraps_due_to_dtype(
    static_range_small,
):

    from neuralqx.hilbert.utils.index._netket import _local_values_to_digits

    x = np.array([[0, 1, 999]], dtype=np.int64)
    d = _local_values_to_digits(static_range_small, x)

    assert d.shape == x.shape

    expected = x.astype(static_range_small.dtype)
    np.testing.assert_array_equal(d, expected)


def test_local_digits_to_values_out_of_range_does_not_raise_for_staticrange(
    static_range_small,
):

    from neuralqx.hilbert.utils.index._netket import _local_digits_to_values

    L = int(static_range_small.length)
    d = np.array([[0, L, 1]], dtype=np.int64)
    v = _local_digits_to_values(static_range_small, d)

    assert v.shape == d.shape
    assert int(v[0, 1]) == L


@pytest.mark.xfail(
    strict=False,
    reason="If we later add explicit bounds checks in _local_values_to_digits, enable this as strict.",
)
def test_local_values_to_digits_rejects_out_of_basis_future_contract(
    static_range_small,
):
    from neuralqx.hilbert.utils.index._netket import _local_values_to_digits

    x = np.array([[0, 1, 999]], dtype=np.int64)
    with pytest.raises(ValueError):
        _local_values_to_digits(static_range_small, x)


@pytest.mark.xfail(
    strict=False,
    reason="If we later add explicit bounds checks in _local_digits_to_values, enable this as strict.",
)
def test_local_digits_to_values_rejects_invalid_digits_future_contract(
    static_range_small,
):
    from neuralqx.hilbert.utils.index._netket import _local_digits_to_values

    L = int(static_range_small.length)
    d = np.array([[0, L, 1]], dtype=np.int64)
    with pytest.raises(ValueError):
        _local_digits_to_values(static_range_small, d)
