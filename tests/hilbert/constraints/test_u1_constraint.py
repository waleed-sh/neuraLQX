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


def test_u1constraint_call_batched_and_jittable(nk, jax, jnp):
    from neuralqx.hilbert.constraints.u1_constraint import U1Constraint
    from neuralqx.hilbert.utils._common import make_hashable
    from neuralqx.hilbert.constraints.utils._common import (
        preprocess_constraints_for_jax,
    )

    raw = [
        [["0", "1"], ["2"]],
    ]
    gf = preprocess_constraints_for_jax(raw)

    c = U1Constraint(gauge_fixing=gf, cutoff=1, q_min=0, q_max=2, q_step=1)

    x = jnp.asarray(
        [
            [1, 1, 2],
            [1, 0, 2],
            [2, 2, 1],
        ],
        dtype=jnp.int32,
    )
    y = c(x)
    assert y.shape == (3,)
    assert np.asarray(y).tolist() == [True, False, True]

    f = jax.jit(lambda z: c(z))
    y2 = f(x)
    assert np.asarray(y2).tolist() == [True, False, True]


def test_u1constraint_hash_and_equality_stable_against_structure_variants(nk, jax, jnp):
    from neuralqx.hilbert.constraints.u1_constraint import U1Constraint

    gf_list = [([(0, +1)], [(1, +1)])]
    gf_tuple = (tuple(gf_list[0][0]), tuple(gf_list[0][1]))

    c1 = U1Constraint(gauge_fixing=gf_list, cutoff=2, q_min=0, q_max=1, q_step=1)
    c2 = U1Constraint(
        gauge_fixing=[(list(gf_list[0][0]), list(gf_list[0][1]))],
        cutoff=2,
        q_min=0,
        q_max=1,
        q_step=1,
    )
    c3 = U1Constraint(
        gauge_fixing=[(list(gf_tuple[0]), list(gf_tuple[1]))],
        cutoff=2,
        q_min=0,
        q_max=1,
        q_step=1,
    )

    assert c1 == c2 == c3
    assert hash(c1) == hash(c2) == hash(c3)

    c_other_cutoff = U1Constraint(
        gauge_fixing=gf_list, cutoff=3, q_min=0, q_max=1, q_step=1
    )
    assert c1 != c_other_cutoff
    assert hash(c1) != hash(c_other_cutoff)


def test_u1constraint_equality_should_depend_on_qrange_future_contract(nk, jax, jnp):
    from neuralqx.hilbert.constraints.u1_constraint import U1Constraint

    gf = [([(0, +1)], [(1, +1)])]
    c1 = U1Constraint(gauge_fixing=gf, cutoff=2, q_min=0, q_max=1, q_step=1)
    c2 = U1Constraint(gauge_fixing=gf, cutoff=2, q_min=0, q_max=2, q_step=1)
    assert c1 != c2


def test_u1constraint_is_pytree_leaf(nk, jax, jnp):
    from neuralqx.hilbert.constraints.u1_constraint import U1Constraint

    gf = [([(0, +1)], [(1, +1)])]
    c = U1Constraint(gauge_fixing=gf, cutoff=2, q_min=0, q_max=1, q_step=1)

    leaves, treedef = jax.tree_util.tree_flatten(c)
    assert leaves == []
