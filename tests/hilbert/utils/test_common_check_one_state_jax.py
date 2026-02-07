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


def _mod_reduce(vals: np.ndarray, *, q_min: int, q_max: int, step: int) -> int:
    L = (q_max - q_min) // step + 1
    ks = ((vals - q_min) // step).astype(int)
    ksum = int(np.sum(ks) % L)
    return int(q_min + ksum * step)


def _constraint_ok(state: np.ndarray, constraint, *, q_min, q_max, step) -> bool:
    inc_data, out_data = constraint
    inc_vals = np.array([sgn * state[idx] for idx, sgn in inc_data], dtype=int)
    out_vals = np.array([sgn * state[idx] for idx, sgn in out_data], dtype=int)
    return _mod_reduce(inc_vals, q_min=q_min, q_max=q_max, step=step) == _mod_reduce(
        out_vals, q_min=q_min, q_max=q_max, step=step
    )


def test_check_one_state_jax_matches_reference_and_make_hashable_integration():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import check_one_state_jax, make_hashable

    q_min, q_max, step = 0, 2, 1

    state = np.array([1, 1, 2, 1], dtype=int)

    constraints_list = [
        ([(0, +1), (1, +1)], [(2, +1)]),
        ([(3, +1)], [(1, +1)]),
    ]
    constraints = make_hashable(constraints_list)

    out = check_one_state_jax(jnp.asarray(state), constraints, q_min, q_max, step)
    assert bool(out) is True

    bad_state = np.array([1, 0, 2, 1], dtype=int)
    out2 = check_one_state_jax(jnp.asarray(bad_state), constraints, q_min, q_max, step)
    assert bool(out2) is False

    assert (
        _constraint_ok(state, constraints[0], q_min=q_min, q_max=q_max, step=step)
        is True
    )
    assert (
        _constraint_ok(bad_state, constraints[0], q_min=q_min, q_max=q_max, step=step)
        is False
    )


def test_check_one_state_jax_multiple_constraints_are_conjoined():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import check_one_state_jax

    q_min, q_max, step = 0, 1, 1
    state = jnp.asarray([0, 1, 1], dtype=jnp.int32)

    constraints = (
        (((0, +1),), ((1, +1),)),
        (((1, +1),), ((2, +1),)),
    )

    out = check_one_state_jax(state, constraints, q_min, q_max, step)
    assert bool(out) is False
