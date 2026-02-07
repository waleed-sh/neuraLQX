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

from typing import List
from typing import Sequence

from . import numba as n
from . import jax as j


def ChargeColoringOperator(
    H,
    site: int,
    charge: int,
    *,
    jax: bool = False,
):
    if jax:
        return j.ChargeColoringJax(H, site, charge)
    return n.ChargeColoring(H, site, charge)


def ColoringOperator(
    H,
    site: int,
    charge_vector: Sequence[int],
    *,
    jax: bool = False,
):
    if jax:
        return j.ColoringJax(H, site, charge_vector)
    return n.Coloring(H, site, charge_vector)


def U1HolonomyOperator(
    H,
    edge_index: int,
    *,
    adjoint: bool = False,
    jax: bool = False,
):
    if jax:
        return j.U1HolonomyJax(H, edge_index, adjoint=adjoint)
    return n.U1Holonomy(H, edge_index, adjoint=adjoint)


def NDU1HolonomyOperator(
    H,
    edge_index: int,
    *,
    adjoint: bool = False,
    jax: bool = False,
):
    if jax:
        return j.NDU1HolonomyJax(H, edge_index, adjoint=adjoint)
    return n.NDU1Holonomy(H, edge_index, adjoint=adjoint)


def IdentityOperator(
    H,
    constant: float = 1.0,
    *,
    jax: bool = False,
):
    if jax:
        return j.IdentityJax(H, constant)
    return n.Identity(H, constant)


def NDMinimalLoopHolonomyOperator(
    H,
    minimal_loop: List,
    *,
    adjoint: int = 0,
    jax: bool = False,
):
    if jax:
        return j.NDMinimalLoopHolonomyJax(H, minimal_loop, adjoint=adjoint)
    return n.NDMinimalLoopHolonomy(H, minimal_loop, adjoint=adjoint)


def MinimalLoopHolonomyOperator(
    H,
    minimal_loop: List,
    *,
    copy: int = 0,
    adjoint: bool = False,
    jax: bool = False,
):
    if jax:
        return j.MinimalLoopHolonomyJax(H, minimal_loop, copy=copy, adjoint=adjoint)
    return n.MinimalLoopHolonomy(H, minimal_loop, copy=copy, adjoint=adjoint)


def OneMinusHalfHolonomyOperator(
    H,
    minimal_loop: List,
    *,
    copy: int = 0,
    jax: bool = False,
):
    if jax:
        return j.OneMinusHalfHolonomyJax(H, minimal_loop, copy=copy)
    return n.OneMinusHalfHolonomy(H, minimal_loop, copy=copy)


def NumberOperator(
    H,
    edge: int,
    *,
    jax: bool = False,
):
    if jax:
        return j.NumberJax(H, edge)
    return n.Number(H, edge)
