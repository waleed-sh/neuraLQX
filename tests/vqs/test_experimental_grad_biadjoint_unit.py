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

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import neuralqx as nqx


def _load_module():
    with nqx.cfg.patch("EXPERIMENTAL", True):
        return importlib.import_module(
            "neuralqx.experimental.vqs.mc.mc_state.expect_grad_biadjoint"
        )


mod = _load_module()


@dataclass
class _StatsBox:
    Mean: complex


class _Op:
    def __init__(self, name: str, adjoint=None, dtype=jnp.float32):
        self.name = name
        self._adjoint = adjoint
        self.dtype = dtype

    @property
    def adjoint(self):
        if isinstance(self._adjoint, Exception):
            raise self._adjoint
        return self._adjoint


def _tree_max_abs_diff(a, b) -> float:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(la, lb))


@pytest.fixture(autouse=True)
def _enable_experimental_grad_flag():
    with nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        yield


def test_forces_to_biadjoint_grad_real_and_complex_branches():
    # Unit physics identity:
    # real params  -> Re(F_A + conj(F_A†))
    # complex params -> F_A (Wirtinger branch)
    parameters = {
        "real": jnp.array([1.0, -2.0], dtype=jnp.float32),
        "complex": jnp.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=jnp.complex64),
    }
    F_A = {
        "real": jnp.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=jnp.complex64),
        "complex": jnp.array([5.0 + 6.0j, -7.0 + 8.0j], dtype=jnp.complex64),
    }
    F_Ad = {
        "real": jnp.array([-0.5 + 1.5j, 2.0 - 1.0j], dtype=jnp.complex64),
        "complex": jnp.array([9.0 - 3.0j, 4.0 + 2.0j], dtype=jnp.complex64),
    }

    grad = mod._forces_to_biadjoint_grad(
        F_A,
        F_Ad,
        parameters,
        project_real_for_real_params=True,
    )

    expected_real = (F_A["real"] + jnp.conj(F_Ad["real"])).real.astype(jnp.float32)
    expected_complex = F_A["complex"].astype(jnp.complex64)

    assert jnp.allclose(grad["real"], expected_real, rtol=0.0, atol=0.0)
    assert jnp.allclose(grad["complex"], expected_complex, rtol=0.0, atol=0.0)

    grad_full = mod._forces_to_biadjoint_grad(
        F_A,
        F_Ad,
        parameters,
        project_real_for_real_params=False,
    )
    expected_real_full = (F_A["real"] + jnp.conj(F_Ad["real"])).astype(jnp.complex64)

    assert jnp.allclose(grad_full["real"], expected_real_full, rtol=0.0, atol=0.0)
    assert jnp.iscomplexobj(grad_full["real"])
    assert jnp.allclose(grad_full["complex"], expected_complex, rtol=0.0, atol=0.0)


def test_expect_and_grad_biadjoint_single_disabled_flag_warns_and_falls_back():
    op_adj = _Op("A_dag")
    op = _Op("A", adjoint=op_adj)
    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    with nqx.cfg.patch("EXPERIMENTAL_GRAD", False):
        with pytest.warns(UserWarning, match="`cfg.EXPERIMENTAL_GRAD` is False"):
            out = mod.expect_and_grad_biadjoint(
                vstate, op, chunk_size=None, mutable=False
            )

    assert out is None


def test_expect_and_grad_biadjoint_single_success(monkeypatch):
    op_adj = _Op("A_dag")
    op = _Op("A", adjoint=op_adj)

    stats = _StatsBox(Mean=1.25)
    F_A = {"p": jnp.array([2.0 + 3.0j], dtype=jnp.complex64)}
    F_Ad = {"p": jnp.array([4.0 - 1.0j], dtype=jnp.complex64)}

    calls = []

    def fake_expect_and_forces(vstate, operator, chunk_size, mutable=False):
        calls.append((operator.name, bool(mutable), chunk_size))
        if operator is op:
            return stats, F_A
        if operator is op_adj:
            return _StatsBox(Mean=0.0), F_Ad
        raise AssertionError("Unexpected operator")

    monkeypatch.setattr(mod, "expect_and_forces", fake_expect_and_forces)

    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    out = mod.expect_and_grad_biadjoint(vstate, op, chunk_size=16, mutable=True)

    assert out is not None
    out_stats, out_grad = out
    assert out_stats is stats

    expected = {
        "p": jnp.array([(2.0 + 3.0j + jnp.conj(4.0 - 1.0j)).real], dtype=jnp.float32)
    }
    assert _tree_max_abs_diff(out_grad, expected) == 0.0
    assert calls == [("A", True, 16), ("A_dag", False, 16)]


def test_expect_and_grad_biadjoint_single_self_adjoint_uses_one_force_call(monkeypatch):
    class FakeSquared:
        def __init__(self, parent_operator):
            self.parent = parent_operator

    monkeypatch.setattr(mod, "Squared", FakeSquared)

    parent = _Op("P")
    op = FakeSquared(parent)
    stats = _StatsBox(Mean=2.0)
    F_A = {"p": jnp.array([2.0 + 3.0j], dtype=jnp.complex64)}
    calls = []

    def fake_expect_and_forces(vstate, operator, chunk_size, mutable=False):
        calls.append((operator, bool(mutable), chunk_size))
        return stats, F_A

    monkeypatch.setattr(mod, "expect_and_forces", fake_expect_and_forces)

    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    out = mod.expect_and_grad_biadjoint(vstate, op, chunk_size=5, mutable=True)

    assert out is not None
    out_stats, out_grad = out
    assert out_stats is stats
    expected = {"p": jnp.array([4.0], dtype=jnp.float32)}
    assert _tree_max_abs_diff(out_grad, expected) == 0.0
    assert calls == [(op, True, 5)]


def test_expect_and_grad_biadjoint_single_missing_adjoint_warns_and_falls_back():
    op = _Op("bad", adjoint=NotImplementedError("no adjoint"))
    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    with pytest.warns(UserWarning, match=r"does not implement `?\.?adjoint`?"):
        out = mod.expect_and_grad_biadjoint(vstate, op, chunk_size=None, mutable=False)

    assert out is None


def test_expect_and_grad_biadjoint_single_complex_holomorphic_parameters_supported(
    monkeypatch,
):
    op_adj = _Op("A_dag")
    op = _Op("A", adjoint=op_adj)
    stats = _StatsBox(Mean=0.75)
    F_A = {"p": jnp.array([1.0 + 2.0j], dtype=jnp.complex64)}
    F_Ad = {"p": jnp.array([9.0 - 4.0j], dtype=jnp.complex64)}
    vstate = SimpleNamespace(
        parameters={"p": jnp.array([1.0 + 0.0j], dtype=jnp.complex64)}
    )

    calls = []

    def fake_expect_and_forces(vstate, operator, chunk_size, mutable=False):
        calls.append((operator.name, bool(mutable), chunk_size))
        if operator is op:
            return stats, F_A
        if operator is op_adj:
            return _StatsBox(Mean=0.0), F_Ad
        raise AssertionError("Unexpected operator")

    monkeypatch.setattr(mod, "expect_and_forces", fake_expect_and_forces)
    monkeypatch.setattr(mod, "_is_probably_holomorphic_vstate", lambda _: True)

    out = mod.expect_and_grad_biadjoint(vstate, op, chunk_size=4, mutable=False)
    assert out is not None
    out_stats, out_grad = out

    assert out_stats is stats
    assert _tree_max_abs_diff(out_grad, F_A) == 0.0
    # Complex-holomorphic shortcut uses only F_A (no adjoint-force pass).
    assert calls == [("A", False, 4)]


def test_expect_and_grad_biadjoint_single_nonholomorphic_falls_back(monkeypatch):
    op_adj = _Op("A_dag")
    op = _Op("A", adjoint=op_adj)
    vstate = SimpleNamespace(
        parameters={"p": jnp.array([1.0 + 0.0j], dtype=jnp.complex64)}
    )

    monkeypatch.setattr(mod, "_is_probably_holomorphic_vstate", lambda _: False)

    def should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "expect_and_forces should not be called for non-holomorphic models"
        )

    monkeypatch.setattr(mod, "expect_and_forces", should_not_be_called)

    with pytest.warns(
        UserWarning, match="requires holomorphicity for complex-parameter models"
    ):
        out = mod.expect_and_grad_biadjoint(vstate, op, chunk_size=None, mutable=False)

    assert out is None


def test_expect_and_grad_biadjoint_single_complex_objective_falls_back(monkeypatch):
    op_adj = _Op("A_dag", dtype=jnp.complex64)
    op = _Op("A", adjoint=op_adj, dtype=jnp.complex64)
    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    def should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "expect_and_forces should not be called for complex dtype fallback"
        )

    monkeypatch.setattr(mod, "expect_and_forces", should_not_be_called)

    with pytest.warns(UserWarning, match="requires a real-valued objective"):
        out = mod.expect_and_grad_biadjoint(vstate, op, chunk_size=None, mutable=False)

    assert out is None


def test_expect_and_grad_biadjoint_sequence_success_with_squared_branch(monkeypatch):
    parent_adj = _Op("P_dag")
    parent = _Op("P", adjoint=parent_adj)
    op1_adj = _Op("A1_dag")
    op1 = _Op("A1", adjoint=op1_adj)

    class FakeSquared:
        def __init__(self, parent_operator):
            self.parent = parent_operator

    monkeypatch.setattr(mod, "Squared", FakeSquared)

    sq = FakeSquared(parent)
    operators = [op1, sq]

    stats = _StatsBox(Mean=0.5)
    F_A = {"p": jnp.array([1.0 + 0.2j], dtype=jnp.complex64)}
    F_Ad = {"p": jnp.array([-3.0 + 0.7j], dtype=jnp.complex64)}

    calls = []

    def fake_expect_and_forces(vstate, operator, chunk_size, mutable=False):
        calls.append((operator, bool(mutable), chunk_size))
        if len(calls) == 1:
            assert isinstance(operator, list) and len(operator) == 2
            assert operator[0] is op1
            assert operator[1] is sq
            return stats, F_A
        if len(calls) == 2:
            assert isinstance(operator, list) and len(operator) == 2
            assert operator[0] is op1_adj
            # Squared(A) is Hermitian and therefore maps to itself in the adjoint pass.
            assert operator[1] is sq
            return _StatsBox(Mean=0.0), F_Ad
        raise AssertionError("Unexpected operator list")

    monkeypatch.setattr(mod, "expect_and_forces", fake_expect_and_forces)

    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    out = mod.expect_and_grad_biadjoint_sequence(
        vstate,
        operators,
        chunk_size=8,
        mutable=True,
    )

    assert out is not None
    out_stats, out_grad = out
    assert out_stats is stats
    expected = {
        "p": jnp.array([(1.0 + 0.2j + jnp.conj(-3.0 + 0.7j)).real], dtype=jnp.float32)
    }
    assert _tree_max_abs_diff(out_grad, expected) == 0.0

    assert len(calls) == 2
    assert calls[0] == (operators, True, 8)
    assert calls[1][1:] == (False, 8)


def test_expect_and_grad_biadjoint_sequence_all_self_adjoint_uses_one_force_call(
    monkeypatch,
):
    class FakeSquared:
        def __init__(self, parent_operator):
            self.parent = parent_operator

    monkeypatch.setattr(mod, "Squared", FakeSquared)

    sq1 = FakeSquared(_Op("P1"))
    sq2 = FakeSquared(_Op("P2"))
    operators = [sq1, sq2]

    stats = _StatsBox(Mean=1.5)
    F_A = {"p": jnp.array([1.25 - 0.5j], dtype=jnp.complex64)}
    calls = []

    def fake_expect_and_forces(vstate, operator, chunk_size, mutable=False):
        calls.append((operator, bool(mutable), chunk_size))
        return stats, F_A

    monkeypatch.setattr(mod, "expect_and_forces", fake_expect_and_forces)

    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    out = mod.expect_and_grad_biadjoint_sequence(
        vstate,
        operators,
        chunk_size=11,
        mutable=True,
    )

    assert out is not None
    out_stats, out_grad = out
    assert out_stats is stats
    expected = {"p": jnp.array([2.5], dtype=jnp.float32)}
    assert _tree_max_abs_diff(out_grad, expected) == 0.0
    assert calls == [(operators, True, 11)]


def test_expect_and_grad_biadjoint_sequence_missing_adjoint_warns_and_falls_back(
    monkeypatch,
):
    good_adj = _Op("good_dag")
    good = _Op("good", adjoint=good_adj)
    bad = _Op("bad", adjoint=NotImplementedError("missing"))

    vstate = SimpleNamespace(parameters={"p": jnp.array([0.0], dtype=jnp.float32)})

    def should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "expect_and_forces should not be called when adjoint resolution fails"
        )

    monkeypatch.setattr(mod, "expect_and_forces", should_not_be_called)

    with pytest.warns(UserWarning, match=r"does not implement `?\.?adjoint`?"):
        out = mod.expect_and_grad_biadjoint_sequence(
            vstate,
            [good, bad],
            chunk_size=None,
            mutable=False,
        )

    assert out is None


def test_expect_and_grad_biadjoint_sequence_complex_holomorphic_parameters_shortcut(
    monkeypatch,
):
    op1_adj = _Op("A1_dag")
    op1 = _Op("A1", adjoint=op1_adj)
    op2_adj = _Op("A2_dag")
    op2 = _Op("A2", adjoint=op2_adj)
    operators = [op1, op2]

    stats = _StatsBox(Mean=0.25)
    F_A = {"p": jnp.array([3.0 + 4.0j], dtype=jnp.complex64)}
    vstate = SimpleNamespace(
        parameters={"p": jnp.array([1.0 + 0.0j], dtype=jnp.complex64)}
    )

    calls = []

    def fake_expect_and_forces(vstate, operator, chunk_size, mutable=False):
        calls.append((operator, bool(mutable), chunk_size))
        if len(calls) == 1:
            assert list(operator) == operators
            return stats, F_A
        raise AssertionError(
            "Adjoint sequence should not be evaluated in complex-holomorphic shortcut"
        )

    monkeypatch.setattr(mod, "expect_and_forces", fake_expect_and_forces)
    monkeypatch.setattr(mod, "_is_probably_holomorphic_vstate", lambda _: True)

    out = mod.expect_and_grad_biadjoint_sequence(
        vstate,
        operators,
        chunk_size=3,
        mutable=False,
    )

    assert out is not None
    out_stats, out_grad = out
    assert out_stats is stats
    assert _tree_max_abs_diff(out_grad, F_A) == 0.0
    assert len(calls) == 1
