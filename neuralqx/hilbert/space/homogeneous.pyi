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

from typing import Any

import jax
import numpy as np

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.layout import GraphHilbertLayout
from neuralqx.hilbert.layout import HilbertStateView
from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.local import VectorRange

from .discrete import DiscreteHilbertSpace

class HomogeneousDiscreteHilbert(DiscreteHilbertSpace):
    graph: AbstractGraph
    local_space: AbstractLocalSpace | VectorRange
    edges: bool
    vertices: bool
    constraint: AbstractDiscreteConstraint
    layout: GraphHilbertLayout
    def __init__(
        self,
        graph: AbstractGraph,
        local_space: AbstractLocalSpace | VectorRange | Any,
        edges: bool = True,
        vertices: bool = False,
        constraint: AbstractDiscreteConstraint | None = None,
    ) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def local_sizes(self) -> tuple[int, ...]: ...
    @property
    def local_size(self) -> int: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def local_spaces(self) -> tuple[AbstractLocalSpace, ...]: ...
    def with_constraint(
        self,
        constraint: AbstractDiscreteConstraint,
    ) -> HomogeneousDiscreteHilbert: ...
    def states_to_local_indices(self, states: Any) -> jax.Array: ...
    def local_indices_to_states(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array: ...
    def local_states_valid(self, states: Any) -> jax.Array: ...
    def view(self, states: Any) -> HilbertStateView: ...
    def flatten(
        self, view: HilbertStateView | None = None, **blocks: Any
    ) -> jax.Array: ...
    def edge_values(self, states: Any, edge: int | Any) -> jax.Array: ...
    def vertex_values(self, states: Any, vertex: int | Any) -> jax.Array: ...
