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


def test_states_to_numbers_matches_netket_for_random_batches(
    unconstrained_space, rng, nk
):
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import states_to_numbers

    space = unconstrained_space
    L = int(space.allowed_basis_states.length)
    N = int(space.size)

    shapes = [
        (N,),
        (7, N),
        (2, 3, N),
    ]

    for shp in shapes:
        states = rng.integers(0, L, size=shp, dtype=np.int64)
        if len(shp) == 1:
            nk_num = int(np.asarray(space.hilbert.states_to_numbers(states))[()])
        else:
            st2d = states.reshape((-1, N))
            nk_num = np.asarray(space.hilbert.states_to_numbers(st2d)).reshape(shp[:-1])

        got_netket = states_to_numbers(space, states, backend="netket")
        got_python = states_to_numbers(space, states, backend="python")
        got_auto = states_to_numbers(space, states, backend="auto")

        np.testing.assert_array_equal(np.asarray(got_netket), nk_num)
        np.testing.assert_array_equal(np.asarray(got_python), nk_num)
        np.testing.assert_array_equal(np.asarray(got_auto), nk_num)


def test_numbers_to_states_matches_netket_for_scalar_and_arrays(
    unconstrained_space, rng, nk
):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import numbers_to_states

    space = unconstrained_space
    L = int(space.allowed_basis_states.length)
    N = int(space.size)

    num = int(rng.integers(0, L**N))
    st_nk = np.asarray(space.hilbert.numbers_to_states(num)).reshape((N,))
    st_map = np.asarray(numbers_to_states(space, num, backend="python"))
    np.testing.assert_array_equal(st_map, st_nk)

    nums = rng.integers(0, L**N, size=(9,), dtype=np.int64)
    st_nk = np.asarray(space.hilbert.numbers_to_states(nums)).reshape((9, N))
    st_map = np.asarray(numbers_to_states(space, nums, backend="python"))
    np.testing.assert_array_equal(st_map, st_nk)

    nums2 = rng.integers(0, L**N, size=(2, 4), dtype=np.int64)
    st_nk = np.asarray(space.hilbert.numbers_to_states(nums2.reshape(-1))).reshape(
        (2, 4, N)
    )
    st_map = np.asarray(numbers_to_states(space, nums2, backend="python"))
    np.testing.assert_array_equal(st_map, st_nk)


def test_states_numbers_roundtrip_bijection_small_exhaustive(unconstrained_space, nk):

    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import (
        states_to_numbers,
        numbers_to_states,
    )

    space = unconstrained_space
    L = int(space.allowed_basis_states.length)
    N = int(space.size)

    assert L**N <= 2000

    vals = np.asarray(space.allowed_basis_states.all_states())
    all_states = np.array(
        list(__import__("itertools").product(vals, repeat=N)), dtype=vals.dtype
    )

    nums = states_to_numbers(space, all_states, backend="python")
    nums = np.asarray(nums).reshape((-1,))
    assert nums.shape[0] == all_states.shape[0]

    assert set(map(int, nums.tolist())) == set(range(L**N))

    st2 = np.asarray(numbers_to_states(space, nums, backend="python"))
    np.testing.assert_array_equal(st2, all_states)


def test_states_to_numbers_auto_falls_back_to_python_if_netket_fails(
    unconstrained_space, rng
):
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils.index.mapping import states_to_numbers

    class BrokenHilbert:
        def states_to_numbers(self, *_a, **_k):
            raise RuntimeError("kaboom")

    space = unconstrained_space
    broken = type(space)(
        hilbert=BrokenHilbert(),
        allowed_basis_states=space.allowed_basis_states,
        size=space.size,
        dtype=space.dtype,
    )

    L = int(space.allowed_basis_states.length)
    N = int(space.size)
    states = rng.integers(0, L, size=(4, N), dtype=np.int64)

    out_auto = states_to_numbers(broken, states, backend="auto")
    with pytest.raises(Exception):
        _ = states_to_numbers(broken, states, backend="netket")

    out_py = states_to_numbers(space, states, backend="python")
    np.testing.assert_array_equal(np.asarray(out_auto), np.asarray(out_py))


def test_states_to_numbers_return_dtype_auto_switches_to_object_when_needed(
    unconstrained_space, rng, nk, jnp
):

    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")

    StaticRange = nk.utils.StaticRange
    from neuralqx.hilbert.utils.index.mapping import states_to_numbers

    local_states = StaticRange(start=0, step=1, length=2)
    N = 70
    h = nk.hilbert.HomogeneousHilbert(local_states=local_states, N=N)

    class Space:
        hilbert = h
        allowed_basis_states = local_states
        size = N
        dtype = jnp.int32

    st = rng.integers(0, 2, size=(2, N), dtype=np.int64)

    nums = states_to_numbers(Space, st, backend="python", return_dtype="auto")
    nums = np.asarray(nums)
    assert nums.dtype == object
