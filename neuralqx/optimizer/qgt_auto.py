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


"""Automatic QGT selection policies."""

from __future__ import annotations

from typing import Any

from neuralqx.jax.differentiation import real_coordinate_size
from neuralqx.jax.tree import tree_size

from .qgt import qgt_onthefly
from .qgt_jacobian import qgt_jacobian_dense
from .solver import auto as auto_solver
from .solver import cholesky
from .solver import pinv
from .solver import pinv_smooth
from .solver import solve
from .solver._api import solver_base
from .solver._api import solver_config


def qgt_auto(
    vstate: Any,
    *,
    solver: Any | None = None,
    solver_kwargs: dict[str, Any] | None = None,
    dense_threshold: int | None = None,
    **kwargs: Any,
):
    """Select a QGT representation for the requested solver policy.

    Dense direct solves use :class:`QGTJacobianDense`; iterative solves use the
    matrix-free :class:`QGTOnTheFly`. When the solver itself is ``solver.auto``,
    the same dense threshold controls both decisions so the QGT representation
    and linear solver stay aligned.
    """
    if prefer_dense_qgt(
        vstate,
        solver=solver,
        solver_kwargs=solver_kwargs,
        dense_threshold=dense_threshold,
        qgt_kwargs=kwargs,
    ):
        return qgt_jacobian_dense(vstate, **kwargs)
    return qgt_onthefly(vstate, **kwargs)


def prefer_dense_qgt(
    vstate: Any,
    *,
    solver: Any | None,
    solver_kwargs: dict[str, Any] | None = None,
    dense_threshold: int | None = None,
    qgt_kwargs: dict[str, Any] | None = None,
) -> bool:
    """Return whether SR should build a dense-Jacobian QGT."""
    solver_base_ = solver_base(solver)
    merged_kwargs = solver_config(solver) | dict(solver_kwargs or {})
    if dense_threshold is None:
        dense_threshold = merged_kwargs.get("dense_threshold")
    if dense_threshold is None:
        dense_threshold = 2048

    if solver_base_ is auto_solver:
        return _qgt_coordinate_size(vstate, qgt_kwargs or {}) <= int(dense_threshold)
    return solver_base_ in {solve, cholesky, pinv, pinv_smooth}


def _qgt_coordinate_size(vstate: Any, qgt_kwargs: dict[str, Any]) -> int:
    mode = qgt_kwargs.get("mode")
    holomorphic = qgt_kwargs.get("holomorphic")
    if mode == "holomorphic" or holomorphic is True:
        return tree_size(vstate.parameters)
    return real_coordinate_size(vstate.parameters)


__all__ = ["prefer_dense_qgt", "qgt_auto"]
