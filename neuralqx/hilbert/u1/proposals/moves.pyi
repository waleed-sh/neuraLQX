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

from neuralqx.utils.struct import Struct

class AbstractU1Move(Struct): ...

class FreeEdgeFlipSingleGauge(AbstractU1Move):
    n_edges: int
    adjacency: bool
    def __init__(self, n_edges: int = 1, adjacency: bool = False) -> None: ...

class FreeEdgeFlipAllGauge(AbstractU1Move):
    n_edges: int
    adjacency: bool
    def __init__(self, n_edges: int = 1, adjacency: bool = False) -> None: ...

class PlaquetteFlipSingleGauge(AbstractU1Move):
    def __init__(self) -> None: ...

class PlaquetteFlipAllGauge(AbstractU1Move):
    def __init__(self) -> None: ...
