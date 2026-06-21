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

from collections.abc import Sequence

from neuralqx.utils.struct import Struct

from .relation import GaugeRelation

class GaugeFixingTopology(Struct):
    n_edges: int
    free: tuple[int, ...]
    slave_topo: tuple[int, ...]
    slave_fixers: tuple[tuple[tuple[int, int], ...], ...]
    def __init__(
        self,
        n_edges: int,
        free: Sequence[int],
        slave_topo: Sequence[int],
        slave_fixers: Sequence[Sequence[tuple[int, int]]],
    ) -> None: ...
    @property
    def n_free(self) -> int: ...

def build_topology(
    relations: Sequence[GaugeRelation],
    *,
    n_edges: int,
) -> GaugeFixingTopology: ...
