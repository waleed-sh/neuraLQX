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

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.local import VectorRange
from neuralqx.utils.struct import Struct

from .dof import SiteDof

class HilbertStateView(Struct):
    edges: Any | None
    vertices: Any | None
    def __init__(
        self,
        edges: Any | None = None,
        vertices: Any | None = None,
    ) -> None: ...

class GraphHilbertLayout(Struct):
    graph: AbstractGraph
    edge_space: AbstractLocalSpace | VectorRange | None
    vertex_space: AbstractLocalSpace | VectorRange | None
    edge_spaces: tuple[AbstractLocalSpace | VectorRange | None, ...]
    vertex_spaces: tuple[AbstractLocalSpace | VectorRange | None, ...]
    def __init__(
        self,
        graph: AbstractGraph,
        edge_space: AbstractLocalSpace | VectorRange | Any | None = None,
        vertex_space: AbstractLocalSpace | VectorRange | Any | None = None,
        edge_spaces: (
            tuple[AbstractLocalSpace | VectorRange | Any | None, ...] | None
        ) = None,
        vertex_spaces: (
            tuple[AbstractLocalSpace | VectorRange | Any | None, ...] | None
        ) = None,
    ) -> None: ...
    @property
    def edge_widths(self) -> tuple[int, ...]: ...
    @property
    def vertex_widths(self) -> tuple[int, ...]: ...
    @property
    def edge_width(self) -> int: ...
    @property
    def vertex_width(self) -> int: ...
    @property
    def edge_offsets(self) -> tuple[int, ...]: ...
    @property
    def vertex_offsets(self) -> tuple[int, ...]: ...
    @property
    def edge_size(self) -> int: ...
    @property
    def vertex_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dofs(self) -> tuple[SiteDof, ...]: ...
    def view(self, states: Any) -> HilbertStateView: ...
    def flatten(
        self,
        view: HilbertStateView | None = None,
        *,
        edges: Any | None = None,
        vertices: Any | None = None,
    ) -> jax.Array: ...
    def edge_values(self, states: Any, edge: int | Any) -> jax.Array: ...
    def vertex_values(self, states: Any, vertex: int | Any) -> jax.Array: ...
    def edge_site(self, edge: int | Any, component: int = 0) -> int: ...
    def vertex_site(self, vertex: int | Any, component: int = 0) -> int: ...
    def site_to_edge(self, site: int) -> tuple[int, int]: ...
    def site_to_vertex(self, site: int) -> tuple[int, int]: ...

GraphDofLayout = GraphHilbertLayout
