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


from __future__ import annotations

import pytest


def test_solvers_class_unknown_solver_raises(solvers_modules):
    tmod, _bmod, _emod, smod = solvers_modules

    with pytest.raises(ValueError) as e:
        smod.Solvers("does-not-exist")
    msg = str(e.value).lower()
    assert "unknown solver" in msg

    assert "conjugate gradient" in msg


def test_solvers_class_lowercases_input(
    solvers_modules, patch_netket_solver_symbols, monkeypatch
):

    tmod, bmod, _emod, smod = solvers_modules
    sent = patch_netket_solver_symbols

    s = smod.Solvers("ChOlEsKy")
    assert isinstance(s.solver_cfg, tmod.Cholesky)
    assert s.solver is sent.cholesky


def test_solvers_property_caches_build(monkeypatch, solvers_modules):

    tmod, _bmod, _emod, smod = solvers_modules

    calls = {"n": 0}

    def fake_build_solver(cfg):
        calls["n"] += 1
        return ("SOLVER", cfg)

    monkeypatch.setattr(smod, "build_solver", fake_build_solver, raising=True)

    s = smod.Solvers("default solver")
    a = s.solver
    b = s.solver
    assert a == b
    assert calls["n"] == 1


def test_export_info_delegates_to_export_module(monkeypatch, solvers_modules):
    tmod, _bmod, _emod, smod = solvers_modules

    def fake_export(cfg):
        return ["X"], ["Y"]

    monkeypatch.setattr(smod, "export_info", fake_export, raising=True)

    s = smod.Solvers("lu")
    fields, values = s.export_info()
    assert fields == ["X"]
    assert values == ["Y"]
