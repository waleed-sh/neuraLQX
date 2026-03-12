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

from . import jax as _j
from . import numba as _n


def EuclideanConstraintOperator(
    H,
    *,
    lapse: float = 1.0,
    power: float = 0.25,
    immirzi: float = 1.0,
    jax: bool = False,
):
    if jax:
        return _j.EuclideanConstraintJax(H, lapse=lapse, power=power, immirzi=immirzi)
    return _n.EuclideanConstraint(H, lapse=lapse, power=power, immirzi=immirzi)


def LorentzianConstraintOperator(H, *, immirzi: float = 1.0, jax: bool = False):
    if jax:
        return _j.LorentzianConstraintJax(H, immirzi=immirzi)
    return _n.LorentzianConstraint(H, immirzi=immirzi)


def QRCreationOperator(H, *, site: int, n: int = 1, jax: bool = False):
    if jax:
        return _j.QRCreationJax(H, site=site, n=n)
    return _n.QRCreation(H, site=site, n=n)


def QRAnnihilationOperator(H, *, site: int, n: int = 1, jax: bool = False):
    if jax:
        return _j.QRAnnihilationJax(H, site=site, n=n)
    return _n.QRAnnihilation(H, site=site, n=n)


def QRFluxOperator(
    H, *, site: int, power: float = 1.0, inverse: bool = False, jax: bool = False
):
    if jax:
        return _j.QRFluxJax(H, site=site, power=power, inverse=inverse)
    return _n.QRFlux(H, site=site, power=power, inverse=inverse)
