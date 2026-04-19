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

import importlib
import sys
import warnings

import numpy as np
import pytest

import neuralqx.vqs.mc.mc_state.expect_forces as forces_mod
import neuralqx.vqs.mc.mc_state.expect_grad as grad_mod

from tests.operators.toy_ops import KetSigmap


class _KetSigmapWithAdjoint(KetSigmap):
    @property
    def adjoint(self):
        return self


class _KetSigmapWithAdjointRealDtype(_KetSigmapWithAdjoint):
    @property
    def dtype(self):
        return np.float64


def _drop_experimental_modules():
    names = [
        name
        for name in list(sys.modules)
        if name == "neuralqx.experimental" or name.startswith("neuralqx.experimental.")
    ]
    for name in names:
        sys.modules.pop(name, None)


def test_experimental_grad_routes_single_nonhermitian(mcstate, monkeypatch, nqx):
    op = KetSigmap(mcstate.hilbert, 0)

    called = {"count": 0}
    sentinel = (object(), object())

    def fake_try(vstate, Ô, chunk_size, *, mutable):
        called["count"] += 1
        assert Ô is op
        return sentinel

    monkeypatch.setattr(grad_mod, "_try_biadjoint_single", fake_try)

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        out = mcstate.expect_and_grad(op, mutable=False)

    assert called["count"] == 1
    assert out is sentinel


def test_experimental_grad_routes_nonhermitian_sequence(mcstate, monkeypatch, nqx):
    ops = [KetSigmap(mcstate.hilbert, 0), KetSigmap(mcstate.hilbert, 1)]

    called = {"count": 0}
    sentinel = (object(), object())

    def fake_try(vstate, Ô_list, chunk_size, *, mutable):
        called["count"] += 1
        assert list(Ô_list) == ops
        return sentinel

    monkeypatch.setattr(grad_mod, "_try_biadjoint_sequence", fake_try)

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        out = mcstate.expect_and_grad(ops, mutable=False)

    assert called["count"] == 1
    assert out is sentinel


def test_experimental_grad_routes_squared_single(
    mcstate, ops_spin_2, monkeypatch, nqx, nk
):
    sx0, _, _ = ops_spin_2
    sq = nk.operator.Squared(sx0)

    called = {"count": 0}
    sentinel = (object(), object())

    def fake_try(vstate, Ô, chunk_size, *, mutable):
        called["count"] += 1
        assert Ô is sq
        return sentinel

    monkeypatch.setattr(grad_mod, "_try_biadjoint_single", fake_try)

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        out = mcstate.expect_and_grad(sq, mutable=False)

    assert called["count"] == 1
    assert out is sentinel


def test_experimental_grad_sequence_respects_fused_forces_path(
    mcstate,
    monkeypatch,
    nqx,
):
    op0 = _KetSigmapWithAdjointRealDtype(mcstate.hilbert, 0)
    op1 = _KetSigmapWithAdjointRealDtype(mcstate.hilbert, 1)
    ops = [op0, op1]

    called = {"fused": 0}
    original = forces_mod._forces_expect_hermitian_sequence_fused

    def wrapped(*args, **kwargs):
        called["fused"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(forces_mod, "_forces_expect_hermitian_sequence_fused", wrapped)

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        with nqx.cfg.patch("FUSED_KERNELS", True):
            _, grad_fused = mcstate.expect_and_grad(ops, mutable=False)
        fused_calls = called["fused"]

        with nqx.cfg.patch("FUSED_KERNELS", False):
            _, grad_unfused = mcstate.expect_and_grad(ops, mutable=False)

    assert grad_fused is not None
    assert grad_unfused is not None
    assert fused_calls >= 2
    assert called["fused"] == fused_calls


def test_try_biadjoint_import_does_not_emit_symbolic_module_warning(nqx):
    class _NoAdjoint:
        @property
        def adjoint(self):
            raise NotImplementedError

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = grad_mod._try_biadjoint_single(
                None, _NoAdjoint(), None, mutable=False
            )

    msgs = [str(w.message) for w in caught]

    assert out is None
    assert any("does not implement `.adjoint`" in m for m in msgs)
    assert not any(
        "Module `neuralqx.experimental.operators.symbolic` is deprecated" in m
        for m in msgs
    )


def test_experimental_namespace_guard_respects_config_flags(nqx):
    _drop_experimental_modules()
    with (
        nqx.cfg.patch("EXPERIMENTAL", False),
        nqx.cfg.patch("EXPERIMENTAL_GRAD", False),
    ):
        with pytest.raises(nqx.utils.errors.DeniedExperimentalModuleImportError):
            importlib.import_module("neuralqx.experimental")

    _drop_experimental_modules()
    with nqx.cfg.patch("EXPERIMENTAL", False), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        with pytest.raises(nqx.utils.errors.DeniedExperimentalModuleImportError):
            importlib.import_module("neuralqx.experimental")

    _drop_experimental_modules()
    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", False):
        mod = importlib.import_module("neuralqx.experimental")
        assert mod.__name__ == "neuralqx.experimental"

    _drop_experimental_modules()


def test_symbolic_deprecation_emits_only_when_symbolic_is_imported(nqx):
    _drop_experimental_modules()
    with nqx.cfg.patch("EXPERIMENTAL", True):
        with warnings.catch_warnings(record=True) as caught_ops:
            warnings.simplefilter("always")
            ops_mod = importlib.import_module("neuralqx.experimental.operators")

        msgs_ops = [str(w.message) for w in caught_ops]
        assert not any(
            "Module `neuralqx.experimental.operators.symbolic` is deprecated" in m
            for m in msgs_ops
        )

        with warnings.catch_warnings(record=True) as caught_symbolic:
            warnings.simplefilter("always")
            _ = ops_mod.symbolic

        msgs_symbolic = [str(w.message) for w in caught_symbolic]
        if nqx.cfg.get("TESTING"):
            assert not any(
                "Module `neuralqx.experimental.operators.symbolic` is deprecated" in m
                for m in msgs_symbolic
            )
        else:
            assert any(
                "Module `neuralqx.experimental.operators.symbolic` is deprecated" in m
                for m in msgs_symbolic
            )

    _drop_experimental_modules()
