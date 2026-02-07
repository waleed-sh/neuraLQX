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

r"""
Randomisation utilities for non planar graph embeddings.

This module provides helper functionality to generate a random embedding for a graph
whose vertices are already expressed as coordinate tuples. The intended use case is to
assign fresh coordinates to an existing non planar edge list while preserving the
graph connectivity and edge labels.

A non planar edge is assumed to have the form

.. math::

   e = \bigl((x_1,y_1,z_1),(x_2,y_2,z_2),k\bigr)

where the first two entries identify the endpoints via their current coordinates and the
third entry is an arbitrary edge label or key :math:`k`.

The randomisation replaces every vertex coordinate :math:`(x,y,z)` by an independent
sample

.. math::

   (X,Y,Z) \sim \mathcal N(\mu,\sigma^2)\times\mathcal N(\mu,\sigma^2)\times\mathcal N(\mu,\sigma^2)

and returns both the rewritten edge list and the mapping from original vertices to the
new coordinates.
"""

from typing import List
from typing import Optional

import numpy as np


def randomize_graph_coordinates(
    edges: List,
    mean: float = 0.0,
    std: float = 5.0,
    seed: Optional[int] = None,
):
    r"""
    Randomise the coordinates of every vertex in a non planar edge list.

    The input is a list of edges of the form

    .. math::

       \bigl((x_1,y_1,z_1),(x_2,y_2,z_2),k\bigr)

    Vertices are identified by their coordinate tuples in the input. The function collects
    the set of unique vertex tuples and assigns each a new coordinate triple sampled from a
    Gaussian distribution with mean `mean` and standard deviation `std`, independently for
    each Cartesian component.

    The output preserves edge labels exactly, and rewrites only the endpoint coordinates.

    :param edges: List of non planar edges. Each entry must be a triple
      ``(v1, v2, label)`` where ``v1`` and ``v2`` are coordinate tuples and ``label`` is an
      edge label or key that is preserved.
    :param mean: Mean :math:`\mu` of the Gaussian used for each coordinate component.
    :param std: Standard deviation :math:`\sigma` of the Gaussian used for each component.
    :param seed: Optional integer seed for NumPy's RNG for reproducible embeddings.
    :return: Tuple ``(edges_randomised, coords_map)`` where ``edges_randomised`` is the
      rewritten edge list with new coordinates and ``coords_map`` maps each original vertex
      tuple to its new coordinate tuple.
    """

    if seed is not None:
        np.random.seed(seed)

    # collect all unique vertices
    vertices = {v for e in edges for v in e[:2]}

    # assign each vertex a new random coordinate
    coords_map = {
        v: tuple(float(x) for x in np.random.normal(mean, std, size=3))
        for v in vertices
    }

    # rebuild edges with new coordinates
    edges_randomized = [
        (coords_map[v1], coords_map[v2], label) for (v1, v2, label) in edges
    ]

    return edges_randomized, coords_map
