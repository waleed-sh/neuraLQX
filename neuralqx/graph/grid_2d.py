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

import networkx as nx

from .core.utils.build import relabel_edges_to_nonplanar
from .core.utils.build import relabel_nx_edges
from .core.utils.checks import validate_and_insert_keys
from .graph import Graph


class Grid2D(Graph):
    r"""
    Two-dimensional rectangular grid graph :math:`P_m \,\square\, P_n`, optionally periodic.

    For ``periodic=False``, the graph is the Cartesian product of two path graphs, giving the
    usual :math:`m\times n` rectangular lattice. For ``periodic=True``, periodic boundary
    conditions are applied in both directions, yielding a torus graph :math:`C_m \,\square\, C_n`.

    When non-periodic:

    .. math::

        |V| = mn,\qquad |E| = m(n-1) + (m-1)n.

    When periodic:

    .. math::

        |V| = mn,\qquad |E| = 2mn.

    The NetworkX generator uses 2D lattice coordinates internally; these are relabelled to
    consecutive integers before constructing :class:`~neuralqx.graph.graph.Graph`. If
    ``non_planar=True``, vertices are relabelled to 3-tuples and may be given a random spatial
    embedding.

    :param m: Number of rows.
    :param n: Number of columns.
    :param periodic: If True, impose periodic boundary conditions in both directions.
    :param plot: If True, produce a visualisation via the base graph machinery.
    :param non_planar: If True, relabel vertices to 3-tuples to trigger the non-planar pipeline.
    :param random_embedding: If True, assign a random 3D embedding to vertices (non-planar only).
    :param random_embedding_mean: Mean of the Gaussian used for the random embedding.
    :param random_embedding_std: Standard deviation of the Gaussian used for the random embedding.
    :param random_embedding_seed: Seed controlling the random embedding.
    """

    def __init__(
        self,
        m: int,
        n: int,
        periodic: bool = False,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int = 123,
    ):

        # create a temporary networkx 2d grid graph
        _tmp_nx = nx.generators.grid_2d_graph(m, n, periodic=periodic)

        # relabel all edges
        edges = relabel_nx_edges(_tmp_nx)

        # if non-planar, relabel to out (x, y, z) labeling
        if non_planar:
            edges = validate_and_insert_keys(edges)
            edges, _ = relabel_edges_to_nonplanar(edges)

        self._m = m
        self._n = n
        self._periodic = periodic

        super().__init__(
            edges,
            plot,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
        )

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def periodic(self):
        return self._periodic

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"n_edges={self.n_edges}, "
            f"n_vertices={self.n_vertices}, "
            f"n_minimal_loops={self.n_minimal_loops}, "
            f"is_planar={self.is_planar}, "
            f"m={self.m}, "
            f"n={self.n}, "
            f"periodic={self.periodic}"
            f")"
        )
