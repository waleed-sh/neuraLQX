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

from dataclasses import FrozenInstanceError
import pytest


def test_solvers_registry_contains_expected_keys(solvers_modules):
    tmod, *_ = solvers_modules

    expected = {
        "conjugate gradient",
        "pseudo inverse smooth",
        "pseudo inverse",
        "singular value decomposition",
        "cholesky",
        "lu",
        "default solver",
    }
    assert expected.issubset(set(tmod.SOLVERS_REGISTRY.keys()))

    for k in expected:
        Cfg = tmod.SOLVERS_REGISTRY[k]
        cfg = Cfg()
        assert hasattr(cfg, "name")


@pytest.mark.parametrize(
    "CfgName",
    [
        "ConjugateGradient",
        "PseudoInverseSmooth",
        "PseudoInverse",
        "SingularValueDecomposition",
        "Cholesky",
        "LU",
        "DefaultSolver",
    ],
)
def test_solver_configs_are_frozen_dataclasses(solvers_modules, CfgName):
    tmod, *_ = solvers_modules
    Cfg = getattr(tmod, CfgName)
    cfg = Cfg()

    with pytest.raises(FrozenInstanceError):
        cfg.name = "mutated"
