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


def test_space_is_gauge_fixed_detection(gauge_fixed_space_small, unconstrained_space):
    from neuralqx.hilbert.u1.index._constrained_index import _space_is_gauge_fixed

    assert _space_is_gauge_fixed(gauge_fixed_space_small) is True
    assert _space_is_gauge_fixed(unconstrained_space) is False


def test_extract_reduced_free_values_shape_and_order(gauge_fixed_space_small, rng):
    from neuralqx.hilbert.u1.index._constrained_index import (
        _extract_reduced_free_values,
    )

    space = gauge_fixed_space_small
    G, E = space.gauge_dimensions, space.tiny_size
    free = space.gauge_fixing.free
    F = len(free)

    B = 4
    states = rng.integers(
        0, space.allowed_basis_states.length, size=(B, G * E), dtype=np.int64
    )

    v = states.reshape((B, G, E))
    expected = v[:, :, free].reshape((B, G * F))

    got = _extract_reduced_free_values(space, states)
    assert got.shape == (B, G * F)
    np.testing.assert_array_equal(got, expected)


def test_reconstruct_from_reduced_free_values_roundtrip_for_consistent_states(
    gauge_fixed_space_small, rng, jnp
):
    from neuralqx.hilbert.u1.index._constrained_index import (
        _extract_reduced_free_values,
        _reconstruct_from_reduced_free_values,
    )

    space = gauge_fixed_space_small
    G, E = space.gauge_dimensions, space.tiny_size
    B = 6

    F = len(space.gauge_fixing.free)
    free_vals = rng.integers(
        0, space.allowed_basis_states.length, size=(B, G * F), dtype=np.int64
    )

    v = np.zeros((B, G, E), dtype=space.allowed_basis_states.dtype)
    v[:, :, space.gauge_fixing.free] = free_vals.reshape((B, G, F))
    sigma0 = np.asarray(
        space.reimpose_gauge_fixing(
            jnp.asarray(v.reshape((B, G * E)), dtype=space.dtype)
        )
    )

    reduced = _extract_reduced_free_values(space, sigma0)
    sigma1 = _reconstruct_from_reduced_free_values(space, reduced)
    np.testing.assert_array_equal(np.asarray(sigma1), sigma0)

    v1 = np.asarray(sigma1).reshape((B, G, E))
    np.testing.assert_array_equal(
        v1[:, :, space.gauge_fixing.free],
        v.reshape((B, G, E))[:, :, space.gauge_fixing.free],
    )
