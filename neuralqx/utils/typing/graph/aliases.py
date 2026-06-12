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

from collections.abc import Hashable
from collections.abc import Iterable
from typing import TypeAlias

from .dressing import EdgeDressing
from .edge import Edge
from .loop import Loop
from .vertex import Vertex

Coordinate: TypeAlias = int | float
"""A single Cartesian coordinate value accepted by neuraLQX graph inputs."""

CoordinateVertex: TypeAlias = tuple[float, float, float]
"""Canonical coordinate triple used by graph-family helper constructors."""

VertexLike: TypeAlias = Vertex | tuple[Coordinate, Coordinate, Coordinate]
"""Input accepted anywhere a graph vertex is expected."""

UnkeyedEdge: TypeAlias = tuple[VertexLike, VertexLike]
"""Input edge ``(start, end)`` without an explicit parallel-edge key."""

KeyedEdge: TypeAlias = tuple[VertexLike, VertexLike, int]
"""Input edge ``(start, end, key)`` with an explicit parallel-edge key."""

EdgeLike: TypeAlias = Edge | UnkeyedEdge | KeyedEdge
"""Input accepted anywhere a graph edge is expected."""

FamilyEdge: TypeAlias = tuple[CoordinateVertex, CoordinateVertex]
"""Coordinate edge used by graph-family helper constructors."""

GraphEdges: TypeAlias = list[Edge]
"""Canonical mutable list of graph edges."""

GraphEdgesLike: TypeAlias = Iterable[EdgeLike]
"""Input accepted when constructing graph edge collections."""

LoopLike: TypeAlias = Loop | Iterable[EdgeLike]
"""Input accepted anywhere a graph loop is expected."""

MinimalLoop: TypeAlias = Loop
"""A minimal loop represented by a canonical ``Loop`` object."""

MinimalLoopLike: TypeAlias = LoopLike
"""Input accepted when constructing minimal loops."""

DressedEdge: TypeAlias = tuple[Edge, EdgeDressing]
"""An edge together with its dressing data."""

DressedMinimalLoop: TypeAlias = Loop
"""A minimal ``Loop`` whose dressing metadata is set."""

RawEdge: TypeAlias = tuple[Hashable, Hashable, int]
"""Generic keyed edge used internally by graph algorithms."""

__all__ = [
    "Coordinate",
    "CoordinateVertex",
    "DressedEdge",
    "DressedMinimalLoop",
    "EdgeLike",
    "FamilyEdge",
    "GraphEdges",
    "GraphEdgesLike",
    "KeyedEdge",
    "LoopLike",
    "MinimalLoop",
    "MinimalLoopLike",
    "RawEdge",
    "UnkeyedEdge",
    "VertexLike",
]
