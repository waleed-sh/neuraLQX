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

from typing import Optional
from typing import Union

from . import jax as j
from . import numba as n


def ThiemannRegularisedVertexConstraintOperator(
    lqx: "lqx",
    vertex: Union[str, int],
    *,
    apply_lapse: bool = True,
    adjoint: bool = False,
    jax: bool = False,
    fast: bool = True,
):

    if jax:
        return j.ThiemannRegularisedVertexConstraintJax(
            lqx, vertex, apply_lapse=apply_lapse, adjoint=adjoint
        )
    else:
        if fast:
            return n.ThiemannRegularisedVertexConstraintFast(
                lqx, vertex, apply_lapse=apply_lapse, adjoint=adjoint
            )

        return n.ThiemannRegularisedVertexConstraint(
            lqx, vertex, apply_lapse=apply_lapse, adjoint=adjoint
        )


def AreaOperator(
    H,
    edges,
    *,
    squared: bool = False,
    jax: bool = False,
):
    if jax:
        return j.AreaOperatorJax(H, edges, squared=squared)

    return n.AreaOperator(H, edges, squared=squared)


def AreaDifferenceSquaredOperator(
    H,
    edges,
    *,
    jax: bool = False,
):
    if jax:
        return j.AreaDifferenceSquaredOperatorJax(H, edges)
    return n.AreaDifferenceSquaredOperator(H, edges)


def AreaDifferenceSquaredSurfacesOperator(
    H,
    surfaces,
    *,
    jax: bool = False,
):
    if jax:
        return j.AreaDifferenceSquaredSurfacesOperatorJax(H, surfaces)
    return n.AreaDifferenceSquaredSurfacesOperator(H, surfaces)


def GaussConstraintOperator(
    H,
    gauge_dimensions: Optional[int] = None,
    modded: bool = False,
    *,
    jax: bool = False,
):
    if jax:
        return j.GaussConstraintOperatorJax(H, gauge_dimensions, modded=modded)
    return n.GaussConstraintOperator(H, gauge_dimensions, modded=modded)


def VolumeOperator(
    H,
    vertex,
    *,
    jax: bool = False,
):
    if jax:
        return j.VolumeOperatorJax(H, vertex)
    return n.VolumeOperator(H, vertex)
