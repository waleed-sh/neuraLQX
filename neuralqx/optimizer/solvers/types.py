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

from dataclasses import dataclass

from typing import Any
from typing import Dict


class SolverConfig:
    name: str


@dataclass(frozen=True)
class ConjugateGradient(SolverConfig):
    r"""
    Use Conjugate Gradient iteration to solve Ax = b.
    """

    name: str = "Conjugate Gradient"


@dataclass(frozen=True)
class PseudoInverseSmooth(SolverConfig):
    r"""
    NetKet Doc:

    Solve the linear system by building a pseudo-inverse from the
    eigendecomposition obtained from :func:`jax.numpy.linalg.eigh`.

    The eigenvalues :math:`\lambda_i` smaller than
    :math:`r_\textrm{cond} \lambda_\textrm{max}` are truncated (where
    :math:`\lambda_\textrm{max}` is the largest eigenvalue).

    The eigenvalues are further smoothed with another filter, originally introduced in
    `Medvidovic, Sels arXiv:2212.11289 (2022) <https://arxiv.org/abs/2212.11289>`_,
    given by the following equation

    .. math::

        \tilde\lambda_i^{-1}=\frac{\lambda_i^{-1}}{1+\big(\epsilon\frac{\lambda_\textrm{max}}{\lambda_i}\big)^6}

    """

    name: str = "Pseudo-Inverse Smooth Solver"


@dataclass(frozen=True)
class PseudoInverse(SolverConfig):
    """
    Solve the linear system using jax's implementation of the pseudo-inverse.
    """

    name: str = "Pseudo-Inverse Solver"


@dataclass(frozen=True)
class SingularValueDecomposition(SolverConfig):
    r"""
    Solve the linear system using Singular Value Decomposition.
    """

    name: str = "Singular Value Decomposition"


@dataclass(frozen=True)
class Cholesky(SolverConfig):
    """
    Solve the linear system using a Cholesky Factorisation.
    """

    name: str = "Cholesky"


@dataclass(frozen=True)
class LU(SolverConfig):
    """
    Solve the linear system using a LU Factorisation.
    """

    name: str = "LU"


@dataclass(frozen=True)
class DefaultSolver(SolverConfig):
    r"""
    Solve the linear system. Internally uses :func:`jax.numpy.solve`.
    """

    name: str = "Default Solver"


SOLVERS_REGISTRY: Dict[str, Any] = {
    "conjugate gradient": ConjugateGradient,
    "pseudo inverse smooth": PseudoInverseSmooth,
    "pseudo inverse": PseudoInverse,
    "singular value decomposition": SingularValueDecomposition,
    "cholesky": Cholesky,
    "lu": LU,
    "default solver": DefaultSolver,
}
