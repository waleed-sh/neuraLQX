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

import pytest

from neuralqx.utils.errors import (
    ExpectationValueError,
    InvalidOperatorsSequenceError,
    InvalidOperatorsInSequenceError,
)


def test_expect_and_grad_sequence_matches_sum_operator(
    mcstate, ops_spin_2, helpers, jnp
):
    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2

    stats_seq, grad_seq = mcstate.expect_and_grad([sx0, sz0], mutable=False)
    stats_sum, grad_sum = mcstate.expect_and_grad(sx0 + sz0, mutable=False)

    assert jnp.allclose(mean(stats_seq), mean(stats_sum), rtol=1e-6, atol=1e-7)
    tree_allclose(grad_seq, grad_sum, rtol=1e-6, atol=1e-7)


def test_expect_and_grad_sequence_equals_sum_of_single_grads(
    mcstate, ops_spin_2, helpers, jnp, jax
):
    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2

    stats_a, g_a = mcstate.expect_and_grad(sx0, mutable=False)
    stats_b, g_b = mcstate.expect_and_grad(sz0, mutable=False)
    stats_seq, g_seq = mcstate.expect_and_grad([sx0, sz0], mutable=False)

    assert jnp.allclose(
        mean(stats_seq), mean(stats_a) + mean(stats_b), rtol=1e-6, atol=1e-7
    )

    g_ab = jax.tree_util.tree_map(lambda a, b: a + b, g_a, g_b)
    tree_allclose(g_seq, g_ab, rtol=1e-6, atol=1e-7)


def test_expect_and_grad_empty_sequence_raises(mcstate):
    with pytest.raises(ExpectationValueError):
        mcstate.expect_and_grad([], mutable=False)


def test_expect_and_grad_sequence_penaltycost_matches_scaled_parent_sum(
    mcstate, ops_spin_2, helpers, nqx, jnp, jax
):

    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2

    f = 0.5
    pc_x = nqx.operators.PenaltyCost(sx0, factor=f)
    pc_z = nqx.operators.PenaltyCost(sz0, factor=1.0)

    stats_seq, grad_seq = mcstate.expect_and_grad([pc_x, pc_z], mutable=False)

    stats_x, grad_x = mcstate.expect_and_grad([pc_x], mutable=False)
    stats_z, grad_z = mcstate.expect_and_grad([pc_z], mutable=False)

    assert jnp.allclose(
        mean(stats_seq), mean(stats_x) + mean(stats_z), rtol=1e-6, atol=1e-7
    )
    grad_ref = jax.tree_util.tree_map(lambda a, b: a + b, grad_x, grad_z)
    tree_allclose(grad_seq, grad_ref, rtol=1e-6, atol=1e-7)

    pc_x1 = nqx.operators.PenaltyCost(sx0, factor=1.0)
    stats_x1, grad_x1 = mcstate.expect_and_grad([pc_x1], mutable=False)

    assert jnp.allclose(mean(stats_x), f * mean(stats_x1), rtol=1e-6, atol=1e-7)
    grad_scaled = jax.tree_util.tree_map(lambda v: f * v, grad_x1)
    tree_allclose(grad_x, grad_scaled, rtol=1e-6, atol=1e-7)


def test_expect_and_grad_single_penaltycost_scales_like_parent(
    mcstate, ops_spin_2, helpers, nqx, jnp
):
    mean, tree_allclose = helpers
    sx0, _, _ = ops_spin_2
    f = 0.5

    pc = nqx.operators.PenaltyCost(sx0, factor=f)

    stats_pc, grad_pc = mcstate.expect_and_grad(pc, mutable=False)
    stats_ref, grad_ref = mcstate.expect_and_grad(f * sx0, mutable=False)

    assert jnp.allclose(mean(stats_pc), mean(stats_ref), rtol=1e-6, atol=1e-7)
    tree_allclose(grad_pc, grad_ref, rtol=1e-6, atol=1e-7)


def test_squared_sequence_requires_python_list(mcstate, ops_spin_2, nk):

    sx0, _, _ = ops_spin_2
    o1 = nk.operator.Squared(sx0)
    o2 = nk.operator.Squared(0.3 * sx0)

    with pytest.raises(InvalidOperatorsSequenceError):
        mcstate.expect_and_grad((o1, o2), mutable=False)


def test_squared_sequence_accepts_different_member_types(mcstate, ops_spin_2, nk):

    sx0, sz0, _ = ops_spin_2
    o1 = nk.operator.Squared(sx0)

    mcstate.expect_and_grad([o1, sz0], mutable=False)


def test_squared_expect_and_grad_smoke(mcstate, ops_spin_2, nk):
    sx0, _, _ = ops_spin_2
    o1 = nk.operator.Squared(sx0)
    mcstate.expect_and_grad(o1, mutable=False)
