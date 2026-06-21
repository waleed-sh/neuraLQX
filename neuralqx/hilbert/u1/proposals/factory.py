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


"""Factories for U(1)-specific proposal moves.

This module preserves a compact legacy ``flip_state`` interface by translating
scope strings into typed U(1) proposal move descriptors.
"""

from __future__ import annotations

from .moves import AbstractU1Move
from .moves import FreeEdgeFlipAllGauge
from .moves import FreeEdgeFlipSingleGauge


def free_edge_move(
    number_of_edges: int, *, adjacency: bool, scope: str
) -> AbstractU1Move:
    """Builds a free-edge proposal from legacy convenience arguments.

    Args:
        number_of_edges: Number of free base edges to update.
        adjacency: Whether to use adjacent modular updates.
        scope: Scope string selecting pooled or per-gauge updates.

    Returns:
        U(1)-specific free-edge move descriptor.
    """
    normalized = str(scope).strip().lower()
    if normalized in ("all", "agd", "all_gd", "per_gauge", "each"):
        return FreeEdgeFlipAllGauge(number_of_edges, adjacency=adjacency)
    return FreeEdgeFlipSingleGauge(number_of_edges, adjacency=adjacency)
