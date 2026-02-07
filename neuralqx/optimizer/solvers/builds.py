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

from plum import dispatch

from .types import ConjugateGradient
from .types import Cholesky
from .types import SingularValueDecomposition
from .types import DefaultSolver
from .types import PseudoInverse
from .types import PseudoInverseSmooth
from .types import LU


@dispatch
def build_solver(cfg: ConjugateGradient):
    from jax.scipy.sparse.linalg import cg

    return cg


@dispatch
def build_solver(cfg: Cholesky):
    from netket.optimizer.solver import cholesky

    return cholesky


@dispatch
def build_solver(cfg: SingularValueDecomposition):
    from netket.optimizer.solver import svd

    return svd


@dispatch
def build_solver(cfg: DefaultSolver):
    from netket.optimizer.solver import solve

    return solve


@dispatch
def build_solver(cfg: PseudoInverse):
    from netket.optimizer.solver import pinv

    return pinv


@dispatch
def build_solver(cfg: PseudoInverseSmooth):
    from netket.optimizer.solver import pinv_smooth

    return pinv_smooth


@dispatch
def build_solver(cfg: LU):
    from netket.optimizer.solver import LU

    return LU
