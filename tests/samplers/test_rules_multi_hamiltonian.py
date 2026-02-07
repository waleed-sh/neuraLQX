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

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

from neuralqx.samplers.rules import _hamiltonian_list as mh


def test_canonical_probs_normalizes_and_handles_all_zero():
    p = mh._canonical_probs(None, K=4)
    assert p.shape == (4,)
    assert jnp.allclose(jnp.sum(p), 1.0)
    assert jnp.allclose(p, jnp.full((4,), 0.25))

    p2 = mh._canonical_probs(jnp.array([0.0, 0.0, 0.0]), K=3)
    assert jnp.allclose(p2, jnp.full((3,), 1 / 3))

    with pytest.raises(ValueError):
        _ = mh._canonical_probs(jnp.array([1.0, 2.0]), K=3)


def test_multihamiltonian_rule_dispatch_rejects_mixed_operator_types(monkeypatch):

    class JaxOp:
        pass

    class CpuOp:
        pass

    monkeypatch.setattr(
        mh, "_is_known_safe_jax", lambda op: isinstance(op, JaxOp), raising=True
    )
    monkeypatch.setattr(mh, "DiscreteOperatorT", (CpuOp,), raising=False)

    with pytest.raises(TypeError):
        mh.MultiHamiltonianRule([JaxOp(), CpuOp()])


def test_multihamiltonian_rulejax_one_op_transition_log_corr_changes_when_n_conn_changes(
    monkeypatch,
):

    class _FakeDiscreteJaxOperator:
        pass

    monkeypatch.setattr(
        mh, "DiscreteJaxOperator", _FakeDiscreteJaxOperator, raising=True
    )

    class ToyOp(_FakeDiscreteJaxOperator):
        def get_conn_padded(self, x):
            B, D = x.shape
            xp1 = x.at[:, 0].add(1)
            xp2 = x.at[:, 0].add(-1)
            xp = jnp.stack([xp1, xp2], axis=1)

            m1 = jnp.ones((B,))
            m2 = jnp.where(x[:, 0] == 0, 0.0, 1.0)
            mels = jnp.stack([m1, m2], axis=1)
            return xp, mels

        def n_conn(self, x):
            return jnp.where(x[:, 0] == 0, 1, 2)

    op = ToyOp()

    x = jnp.array(
        [
            [0, 0],
            [1, 0],
        ],
        dtype=jnp.int32,
    )
    key = jax.random.PRNGKey(0)

    x_p, logcorr = mh.MultiHamiltonianRuleJax._one_op_transition(op, key, x)

    n1 = jnp.maximum(op.n_conn(x), 1)
    n2 = jnp.maximum(op.n_conn(x_p), 1)
    expected = jnp.log(n1) - jnp.log(n2)

    assert x_p.shape == x.shape
    assert logcorr.shape == (x.shape[0],)
    assert jnp.allclose(logcorr, expected)


def test_multihamiltonian_rulejax_choose_per_chain_uses_per_chain_operator_selection(
    monkeypatch,
):

    class _FakeDiscreteJaxOperator:
        pass

    monkeypatch.setattr(
        mh, "DiscreteJaxOperator", _FakeDiscreteJaxOperator, raising=True
    )

    class Op0(_FakeDiscreteJaxOperator):
        def get_conn_padded(self, x):
            xp = x.at[:, 0].add(1)
            xp = xp[:, None, :]
            mels = jnp.ones((x.shape[0], 1))
            return xp, mels

        def n_conn(self, x):
            return jnp.ones((x.shape[0],), dtype=jnp.int32)

    class Op1(_FakeDiscreteJaxOperator):
        def get_conn_padded(self, x):
            xp = x.at[:, 0].add(10)
            xp = xp[:, None, :]
            mels = jnp.ones((x.shape[0], 1))
            return xp, mels

        def n_conn(self, x):
            return jnp.ones((x.shape[0],), dtype=jnp.int32)

    def fake_sample_categorical(key, p):
        return (key[0] & jnp.uint32(1)).astype(jnp.int32)

    monkeypatch.setattr(
        mh, "_sample_categorical", fake_sample_categorical, raising=True
    )

    rule = mh.MultiHamiltonianRuleJax(operators=[Op0(), Op1()], choose_per_chain=True)
    x = jnp.zeros((6, 2), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)

    x_p, logcorr = rule.transition(
        sampler=None, machine=None, parameters=None, state=None, key=key, x=x
    )

    deltas = x_p[:, 0] - x[:, 0]
    assert bool(jnp.any(deltas == 1)) and bool(
        jnp.any(deltas == 10)
    ), "Per-chain selection did not result in mixed operator effects"
    assert logcorr.shape == (x.shape[0],)
    assert jnp.allclose(logcorr, 0.0)


def test_multihamiltonian_rule_dispatch_prefers_jax_when_all_safe(monkeypatch):

    class _FakeDiscreteJaxOperator:
        pass

    monkeypatch.setattr(
        mh, "DiscreteJaxOperator", _FakeDiscreteJaxOperator, raising=True
    )

    class JaxOp(_FakeDiscreteJaxOperator):
        pass

    monkeypatch.setattr(
        mh,
        "_is_known_safe_jax",
        lambda op: isinstance(op, _FakeDiscreteJaxOperator),
        raising=True,
    )

    monkeypatch.setattr(mh, "DiscreteOperatorT", tuple(), raising=False)

    r = mh.MultiHamiltonianRule([JaxOp(), JaxOp()])
    assert isinstance(r, mh.MultiHamiltonianRuleJax)
