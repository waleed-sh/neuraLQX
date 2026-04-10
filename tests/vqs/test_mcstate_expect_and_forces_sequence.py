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
import jax
import jax.numpy as jnp
import neuralqx as nqx

from neuralqx.utils.errors import ExpectationValueError
from neuralqx.vqs.mc.common import force_to_grad


def _tree_max_abs(a, b, jnp):
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    assert treedef_a == treedef_b
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(leaves_a, leaves_b))


def test_expect_and_forces_sequence_matches_sum_operator(mcstate, ops_spin_2, helpers):
    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2

    stats_seq, forces_seq = mcstate.expect_and_forces([sx0, sz0], mutable=False)
    stats_sum, forces_sum = mcstate.expect_and_forces(sx0 + sz0, mutable=False)

    assert jnp.allclose(mean(stats_seq), mean(stats_sum), rtol=1e-6, atol=1e-7)
    tree_allclose(forces_seq, forces_sum, rtol=1e-6, atol=1e-7)


def test_expect_and_forces_sequence_equals_sum_of_single_forces(
    mcstate, ops_spin_2, helpers
):

    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2

    stats_a, f_a = mcstate.expect_and_forces(sx0, mutable=False)
    stats_b, f_b = mcstate.expect_and_forces(sz0, mutable=False)
    stats_seq, f_seq = mcstate.expect_and_forces([sx0, sz0], mutable=False)

    assert jnp.allclose(
        mean(stats_seq), mean(stats_a) + mean(stats_b), rtol=1e-6, atol=1e-7
    )

    f_ab = jax.tree_util.tree_map(lambda a, b: a + b, f_a, f_b)
    tree_allclose(f_seq, f_ab, rtol=1e-6, atol=1e-7)


def test_expect_and_forces_empty_sequence_raises(mcstate):
    with pytest.raises(ExpectationValueError):
        mcstate.expect_and_forces([], mutable=False)


def test_expect_and_forces_sequence_penaltycost_matches_scaled_parent_sum(
    mcstate, ops_spin_2, helpers
):

    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2
    f = 0.5

    pc = nqx.operators.PenaltyCost(sx0, factor=f)

    stats_seq, forces_seq = mcstate.expect_and_forces([pc, sz0], mutable=False)
    stats_ref, forces_ref = mcstate.expect_and_forces(f * sx0 + sz0, mutable=False)

    assert jnp.allclose(mean(stats_seq), mean(stats_ref), rtol=1e-6, atol=1e-7)
    tree_allclose(forces_seq, forces_ref, rtol=1e-6, atol=1e-7)


def test_expect_and_forces_single_penaltycost_scales_like_parent(
    mcstate, ops_spin_2, helpers
):
    mean, tree_allclose = helpers
    sx0, _, _ = ops_spin_2
    f = 0.5

    pc = nqx.operators.PenaltyCost(sx0, factor=f)

    stats_pc, forces_pc = mcstate.expect_and_forces(pc, mutable=False)
    stats_ref, forces_ref = mcstate.expect_and_forces(f * sx0, mutable=False)

    assert jnp.allclose(mean(stats_pc), mean(stats_ref), rtol=1e-6, atol=1e-7)
    tree_allclose(forces_pc, forces_ref, rtol=1e-6, atol=1e-7)


def test_expect_and_forces_single_iec_matches_expect_and_grad(
    mcstate, ops_spin_2, helpers
):
    mean, _ = helpers
    _, sz0, _ = ops_spin_2
    iec = nqx.operators.InverseExpectationCost(sz0, factor=0.7, alpha=0.2)

    stats_f, forces = mcstate.expect_and_forces(iec, mutable=False)
    stats_g, grad = mcstate.expect_and_grad(iec, mutable=False)
    grad_from_forces = force_to_grad(forces, mcstate.parameters)

    assert jnp.allclose(mean(stats_f), mean(stats_g), rtol=1e-6, atol=1e-7)
    assert _tree_max_abs(grad_from_forces, grad, jnp) < 1e-3


def test_expect_and_forces_sequence_single_iec_matches_expect_and_grad(
    mcstate, ops_spin_2, helpers
):
    mean, _ = helpers
    _, sz0, _ = ops_spin_2
    iec = nqx.operators.InverseExpectationCost(sz0, factor=0.7, alpha=0.2)

    stats_f, forces = mcstate.expect_and_forces([iec], mutable=False)
    stats_g, grad = mcstate.expect_and_grad([iec], mutable=False)
    grad_from_forces = force_to_grad(forces, mcstate.parameters)

    assert jnp.allclose(mean(stats_f), mean(stats_g), rtol=1e-6, atol=1e-7)
    assert _tree_max_abs(grad_from_forces, grad, jnp) < 1e-3
