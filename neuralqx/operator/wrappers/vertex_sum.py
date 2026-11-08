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


"""Linear sums that retain the vertex assignment of each contribution."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neuralqx.utils.struct import field

from .sum import SumOperator


class VertexSum(SumOperator):
    r"""A linear constraint $\sum_v c_v H_v$ with explicit vertex labels.

    Connectivity is identical to `SumOperator`. The labels also preserve the
    local decomposition needed to form a quadratic constraint. Each parent is
    one complete contribution at its vertex, even if that parent is itself a
    sum of several operators. Its internal summands must stay together when
    taking a square.

    LQX models combine contributions carrying the same label before squaring.
    Labels must be unique within this wrapper and agree between Euclidean and
    Lorentzian components. Use actual graph vertex objects for prebuilt models.

    Attributes:
        operators (Iterable[ComputationalOperator]): One linear contribution per vertex.
        coefficients (Any): Scalar multiplier or one coefficient per contribution.
        vertices (tuple[Any, ...]): Hashable vertex labels in the same order as the operators.
    """

    vertices: tuple[Any, ...] = field(static=True, kw_only=True, converter=tuple)
    """Distinct vertex labels identifying the local contributions."""

    def __post_init__(self) -> None:
        """Validate the linear sum and its one-to-one vertex assignment."""
        super().__post_init__()
        if len(self.vertices) != len(self.operators):
            raise ValueError("Supply one vertex label per operator.")
        if None in self.vertices:
            raise ValueError("None is reserved for a constraint without vertex labels.")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("Vertex labels must be unique within a VertexSum.")

    @property
    def adjoint(self) -> VertexSum:
        r"""Conjugate transpose each local contribution while retaining its vertex."""
        return VertexSum(
            tuple(operator.adjoint for operator in self.operators),
            coefficients=jnp.conj(self.coefficients),
            vertices=self.vertices,
        )


__all__ = ["VertexSum"]
