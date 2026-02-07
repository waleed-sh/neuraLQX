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
import jax.numpy as jnp
import neuralqx as nqx

from neuralqx.utils.errors import ExpectationValueError


def test_expect_sequence_matches_sum_operator(mcstate, ops_spin_2, helpers):
    mean, _ = helpers
    sx0, sz0, _ = ops_spin_2

    stats_seq = mcstate.expect([sx0, sz0])
    stats_sum = mcstate.expect(sx0 + sz0)

    assert jnp.allclose(mean(stats_seq), mean(stats_sum), rtol=1e-6, atol=1e-7)
    assert jnp.allclose(
        jnp.asarray(stats_seq.Sigma), jnp.asarray(stats_sum.Sigma), rtol=1e-6, atol=1e-7
    )


def test_expect_sequence_accepts_any_sequence_type(mcstate, ops_spin_2, helpers):
    mean, _ = helpers
    sx0, sz0, _ = ops_spin_2

    stats_list = mcstate.expect([sx0, sz0])
    stats_tuple = mcstate.expect((sx0, sz0))

    assert jnp.allclose(mean(stats_list), mean(stats_tuple), rtol=1e-6, atol=1e-7)


def test_expect_sequence_empty_raises(mcstate):
    with pytest.raises(ExpectationValueError):
        mcstate.expect([])


def test_expect_sequence_does_not_resample_when_samples_cached(
    mcstate, ops_spin_2, helpers
):

    mean, _ = helpers
    sx0, sz0, _ = ops_spin_2

    def _nope(*args, **kwargs):
        raise AssertionError(
            "MCState.sample() was called, but samples should have been reused"
        )

    mcstate.sample = _nope

    stats_seq = mcstate.expect([sx0, sz0])
    stats_sum = mcstate.expect(sx0 + sz0)

    assert jnp.allclose(mean(stats_seq), mean(stats_sum), rtol=1e-6, atol=1e-7)


def test_expect_sequence_penaltycost_matches_scaled_parent_sum(
    mcstate, ops_spin_2, helpers
):

    mean, _ = helpers
    sx0, sz0, _ = ops_spin_2
    f = 0.5

    pc = nqx.operators.PenaltyCost(sx0, factor=f)

    stats_seq = mcstate.expect([pc, sz0])
    stats_ref = mcstate.expect(f * sx0 + sz0)

    assert jnp.allclose(mean(stats_seq), mean(stats_ref), rtol=1e-6, atol=1e-7)
