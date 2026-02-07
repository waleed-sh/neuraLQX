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


def test_build_solver_netket_symbols_are_returned(
    solvers_modules, patch_netket_solver_symbols
):

    tmod, bmod, *_ = solvers_modules
    sent = patch_netket_solver_symbols

    assert bmod.build_solver(tmod.Cholesky()) is sent.cholesky
    assert bmod.build_solver(tmod.SingularValueDecomposition()) is sent.svd
    assert bmod.build_solver(tmod.DefaultSolver()) is sent.solve
    assert bmod.build_solver(tmod.PseudoInverse()) is sent.pinv
    assert bmod.build_solver(tmod.PseudoInverseSmooth()) is sent.pinv_smooth
    assert bmod.build_solver(tmod.LU()) is sent.LU


def test_build_solver_conjugate_gradient_returns_jax_cg(solvers_modules):

    jax = pytest.importorskip("jax")
    from jax.scipy.sparse.linalg import cg as jax_cg

    tmod, bmod, *_ = solvers_modules
    got = bmod.build_solver(tmod.ConjugateGradient())
    assert got is jax_cg
