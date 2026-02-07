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

import importlib
import types
import pytest


def _import_solvers_module(mod_name: str):
    try:
        return importlib.import_module(f"neuralqx.optimizer.solvers.{mod_name}")
    except ModuleNotFoundError:
        return importlib.import_module(f"neuralqx.optimizer.{mod_name}")


@pytest.fixture(scope="session")
def solvers_modules():
    tmod = _import_solvers_module("types")
    bmod = _import_solvers_module("builds")
    emod = _import_solvers_module("export")
    smod = _import_solvers_module("solvers")
    return tmod, bmod, emod, smod


@pytest.fixture
def patch_netket_solver_symbols(monkeypatch):

    import sys

    sent = types.SimpleNamespace(
        solve=object(),
        cholesky=object(),
        svd=object(),
        pinv=object(),
        pinv_smooth=object(),
        LU=object(),
    )

    try:
        nk = importlib.import_module("netket")
        nk_solver = importlib.import_module("netket.optimizer.solver")

        monkeypatch.setattr(nk_solver, "solve", sent.solve, raising=True)
        monkeypatch.setattr(nk_solver, "cholesky", sent.cholesky, raising=True)
        monkeypatch.setattr(nk_solver, "svd", sent.svd, raising=True)
        monkeypatch.setattr(nk_solver, "pinv", sent.pinv, raising=True)
        monkeypatch.setattr(nk_solver, "pinv_smooth", sent.pinv_smooth, raising=True)
        monkeypatch.setattr(nk_solver, "LU", sent.LU, raising=True)

        return sent

    except ModuleNotFoundError:
        netket_mod = types.ModuleType("netket")
        opt_mod = types.ModuleType("netket.optimizer")
        solver_mod = types.ModuleType("netket.optimizer.solver")

        solver_mod.solve = sent.solve
        solver_mod.cholesky = sent.cholesky
        solver_mod.svd = sent.svd
        solver_mod.pinv = sent.pinv
        solver_mod.pinv_smooth = sent.pinv_smooth
        solver_mod.LU = sent.LU

        monkeypatch.setitem(sys.modules, "netket", netket_mod)
        monkeypatch.setitem(sys.modules, "netket.optimizer", opt_mod)
        monkeypatch.setitem(sys.modules, "netket.optimizer.solver", solver_mod)

        return sent
