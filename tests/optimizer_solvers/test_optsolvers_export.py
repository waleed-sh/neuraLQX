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


@pytest.mark.parametrize(
    "CfgName, expected_value",
    [
        ("ConjugateGradient", "Conjugate Gradient"),
        ("PseudoInverseSmooth", "Pseudo-Inverse Smooth Solver"),
        ("PseudoInverse", "Pseudo-Inverse Solver"),
        ("SingularValueDecomposition", "Singular Value Decomposition"),
        ("Cholesky", "Cholesky"),
        ("LU", "LU"),
        ("DefaultSolver", "Default Solver"),
    ],
)
def test_export_info_has_expected_solver_type(solvers_modules, CfgName, expected_value):
    tmod, _bmod, emod, _smod = solvers_modules

    cfg = getattr(tmod, CfgName)()
    fields, values = emod.export_info(cfg)

    assert fields == ["Preconditioner Solver"]
    assert values == [expected_value]
