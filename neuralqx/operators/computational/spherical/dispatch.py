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

from . import numba as n
from . import jax as j


def SphericalExOperator(
    H,
    vertex: int,
    *,
    gamma: float = 1.0,
    jax: bool = False,
):
    if jax:
        return j.SphericalExJax(H, vertex, gamma=gamma)
    return n.SphericalEx(H, vertex, gamma=gamma)


def SphericalVolumeOperator(
    H,
    vertex: int,
    *,
    shift: float = 0.0,
    jax: bool = False,
):
    if jax:
        return j.SphericalVolumeJax(H, vertex, shift=shift)
    return n.SphericalVolume(H, vertex, shift=shift)


def SphericalVertexConstraintBojowaldSwiderskiOperator(
    H,
    vertex: int,
    *,
    include_gamma_terms: bool = True,
    delta: int = 2,  # dev: strictly an integer, see the first dev comment below
    outer_km_constant: int = 0,
    outer_kp_constant: int = 0,
    immirzi: float = 1.0,
    jax: bool = True,
    fast: bool = True,
    adjoint: bool = False,
):
    if jax:
        if adjoint:
            _cls = j.SphericalVertexConstraintBojowaldSwiderskiJaxAdjoint
        else:
            _cls = j.SphericalVertexConstraintBojowaldSwiderskiJax
        return _cls(
            H,
            vertex,
            include_gamma_terms=include_gamma_terms,
            delta=delta,
            outer_km_constant=outer_km_constant,
            outer_kp_constant=outer_kp_constant,
            immirzi=immirzi,
        )

    if fast:
        if adjoint:
            _cls = n.SphericalVertexConstraintBojowaldSwiderskiAdjointFast
        else:
            _cls = n.SphericalVertexConstraintBojowaldSwiderskiFast
        return _cls(
            H,
            vertex,
            include_gamma_terms=include_gamma_terms,
            delta=delta,
            outer_km_constant=outer_km_constant,
            outer_kp_constant=outer_kp_constant,
            immirzi=immirzi,
        )

    if adjoint:
        _cls = n.SphericalVertexConstraintBojowaldSwiderskiAdjoint
    else:
        _cls = n.SphericalVertexConstraintBojowaldSwiderski

    return _cls(
        H,
        vertex,
        include_gamma_terms=include_gamma_terms,
        delta=delta,
        outer_km_constant=outer_km_constant,
        outer_kp_constant=outer_kp_constant,
        immirzi=immirzi,
    )
