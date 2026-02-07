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

from typing import Union

from . import jax as j
from . import numba as n


def VolumeOperator(
    H,
    vertex,
    *,
    jax: bool = False,
):
    if jax:
        return j.VolumeOperatorJax(H, vertex)
    return n.VolumeOperator(H, vertex)


def SqrtVolumeOperator(
    H,
    vertex,
    *,
    jax: bool = False,
):
    if jax:
        return j.VolumeOperatorJaxSqrt(H, vertex)
    return n.VolumeOperatorSqrt(H, vertex)


def SquaredVolumeOperator(
    H,
    vertex,
    *,
    jax: bool = False,
):
    if jax:
        return j.VolumeOperatorJaxSquared(H, vertex)
    return n.VolumeOperatorSquared(H, vertex)


def ThiemannRegularisedVertexConstraintOperator(
    lqx: "lqx",
    vertex: Union[str, int],
    *,
    apply_lapse: bool = True,
    adjoint: bool = False,
    jax: bool = False,
):

    if jax:
        return j.ThiemannRegularisedVertexConstraint3dJax(
            lqx, vertex, apply_lapse=apply_lapse, adjoint=adjoint
        )
    else:
        return n.ThiemannRegularisedVertexConstraint3d(
            lqx, vertex, apply_lapse=apply_lapse, adjoint=adjoint
        )
