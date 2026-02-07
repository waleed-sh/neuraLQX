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


def test_gauge_fixed_states_to_numbers_is_bijection_on_reduced_parameterisation(
    gauge_fixed_space_small,
):
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import (
        states_to_numbers,
        numbers_to_states,
    )
    from neuralqx.hilbert.utils.index._constrained_index import (
        _extract_reduced_free_values,
        _reconstruct_from_reduced_free_values,
    )

    space = gauge_fixed_space_small
    L = int(space.allowed_basis_states.length)
    G = int(space.gauge_dimensions)
    F = int(len(space.gauge_fixing.free))
    D = G * F

    import itertools

    reduced = np.array(list(itertools.product(range(L), repeat=D)), dtype=np.int64)

    sigma = _reconstruct_from_reduced_free_values(space, reduced)
    sigma_np = np.asarray(sigma)

    nums = states_to_numbers(space, sigma_np, backend="auto")
    nums = np.asarray(nums).reshape((-1,))

    assert np.array_equal(nums, np.arange(L**D, dtype=nums.dtype))

    sigma2 = np.asarray(numbers_to_states(space, nums, backend="auto"))
    np.testing.assert_array_equal(sigma2, sigma_np)

    red2 = _extract_reduced_free_values(space, sigma2)
    np.testing.assert_array_equal(red2, reduced)


def test_gauge_fixed_validate_true_rejects_non_gauge_invariant_states(
    gauge_fixed_space_small, rng
):
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import states_to_numbers
    from neuralqx.hilbert.utils.index._constrained_index import (
        _reconstruct_from_reduced_free_values,
    )

    space = gauge_fixed_space_small
    L = int(space.allowed_basis_states.length)
    G = int(space.gauge_dimensions)
    F = int(len(space.gauge_fixing.free))
    D = G * F
    N = int(space.size)

    B = 3
    reduced = rng.integers(0, L, size=(B, D), dtype=np.int64)
    sigma = np.asarray(_reconstruct_from_reduced_free_values(space, reduced))
    assert sigma.shape == (B, N)

    sigma_bad = sigma.copy()
    sigma_bad[0, 1] = (sigma_bad[0, 1] + 1) % L

    _ = states_to_numbers(space, sigma_bad, backend="python", validate=False)

    with pytest.raises(ValueError, match="non gauge-invariant"):
        _ = states_to_numbers(space, sigma_bad, backend="python", validate=True)


def test_gauge_fixed_numbers_to_states_validate_range_checks(gauge_fixed_space_small):
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import numbers_to_states

    space = gauge_fixed_space_small
    L = int(space.allowed_basis_states.length)
    D = int(space.gauge_dimensions) * int(len(space.gauge_fixing.free))
    maxn = L**D

    with pytest.raises(ValueError, match="out of range"):
        _ = numbers_to_states(space, -1, backend="python", validate=True)

    with pytest.raises(ValueError, match="out of range"):
        _ = numbers_to_states(space, maxn, backend="python", validate=True)


def test_gauge_fixed_shapes_are_restored(gauge_fixed_space_small, rng):
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import (
        states_to_numbers,
        numbers_to_states,
    )
    from neuralqx.hilbert.utils.index._constrained_index import (
        _reconstruct_from_reduced_free_values,
    )

    space = gauge_fixed_space_small
    L = int(space.allowed_basis_states.length)
    N = int(space.size)

    shp = (2, 3)
    B = shp[0] * shp[1]

    D = int(space.gauge_dimensions) * int(len(space.gauge_fixing.free))
    reduced = rng.integers(0, L, size=(B, D), dtype=np.int64)

    sigma = np.asarray(_reconstruct_from_reduced_free_values(space, reduced)).reshape(
        (*shp, N)
    )

    nums = states_to_numbers(space, sigma, backend="auto")
    assert np.asarray(nums).shape == shp

    sigma2 = numbers_to_states(space, nums, backend="auto")
    assert np.asarray(sigma2).shape == (*shp, N)
    np.testing.assert_array_equal(np.asarray(sigma2), sigma)
