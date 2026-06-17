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

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from neuralqx.utils.struct import Struct

from .abstract import AbstractLocalSpace

class VectorRange(Struct):
    local_space: AbstractLocalSpace
    components: int
    def __init__(
        self,
        local_space: AbstractLocalSpace | Any,
        components: int,
    ) -> None: ...
    @property
    def local_size(self) -> int: ...

class HeterogeneousLocalSpace(Struct):
    edge: AbstractLocalSpace | VectorRange | None
    vertex: AbstractLocalSpace | VectorRange | None
    edge_spaces: tuple[tuple[Any, AbstractLocalSpace | VectorRange | None], ...]
    vertex_spaces: tuple[tuple[Any, AbstractLocalSpace | VectorRange | None], ...]
    def __init__(
        self,
        edge: AbstractLocalSpace | VectorRange | Any | None = None,
        vertex: AbstractLocalSpace | VectorRange | Any | None = None,
        edge_spaces: Mapping[Any, Any] | Sequence[tuple[Any, Any]] | None = None,
        vertex_spaces: Mapping[Any, Any] | Sequence[tuple[Any, Any]] | None = None,
    ) -> None: ...

def coerce_dof_space(value: Any) -> AbstractLocalSpace | VectorRange | None: ...
def dof_width(value: AbstractLocalSpace | VectorRange | None) -> int: ...
def scalar_space(value: AbstractLocalSpace | VectorRange) -> AbstractLocalSpace: ...
