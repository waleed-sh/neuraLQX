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


"""Graph construction and graph-family APIs for neuraLQX.

The graph package provides the directed keyed multigraph representation used by
Hilbert spaces, loop operators, graph-derived sign tables, and model facades.
The main entry point is :class:`Graph`, which accepts oriented coordinate
edges, canonicalizes vertices and parallel-edge keys, and precomputes derived
objects such as shortest loops and the line-graph dual.

The package also exports named graph families used throughout examples and
prebuilt models, including square grids, ladders, circular ladders, hypercubes,
hexagonal and triangular lattices, the complete graph K5, and the half-ladder
graph used by the spherical Bojowald-Swiderski model.
"""

from .abstract_graph import AbstractGraph
from .circular_ladder import CircularLadderGraph
from .dodecahedral import DodecahedralGraph
from .graph import Graph
from .grid_2d import Grid2D
from .half_ladder_graph import HalfLadderGraph
from .hexagonal_lattice import HexagonalLatticeGraph
from .hypercube import HypercubeGraph
from .k5_graph import K5Graph
from .ladder_graph import LadderGraph
from .single_vertex import SingleVertexGraph
from .triangular_lattice import TriangularLatticeGraph

__all__ = [
    "AbstractGraph",
    "CircularLadderGraph",
    "DodecahedralGraph",
    "Graph",
    "Grid2D",
    "HalfLadderGraph",
    "HexagonalLatticeGraph",
    "HypercubeGraph",
    "K5Graph",
    "LadderGraph",
    "SingleVertexGraph",
    "TriangularLatticeGraph",
]
