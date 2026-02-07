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


class DodecahedralGraph(Graph):
    r"""
    Dodecahedral graph.

    This is the 1-skeleton of the regular dodecahedron. It is a 3-regular graph with

    .. math::

        |V| = 20,\qquad |E| = 30.

    The graph is constructed from :func:`networkx.generators.dodecahedral_graph`, then relabelled
    to consecutive integer vertex labels before being passed to :class:`~neuralqx.graph.graph.Graph`.
    If ``non_planar=True``, vertices are relabelled to fixed-length 3-tuples and may be given a
    random spatial embedding.

    :param plot: If True, produce a visualisation via the base graph machinery.
    :param non_planar: If True, relabel vertices to 3-tuples to trigger the non-planar pipeline.
    :param random_embedding: If True, assign a random 3D embedding to vertices (non-planar only).
    :param random_embedding_mean: Mean of the Gaussian used for the random embedding.
    :param random_embedding_std: Standard deviation of the Gaussian used for the random embedding.
    :param random_embedding_seed: Seed controlling the random embedding.
    """

    def __init__(
        self,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int = 123,
    ):

        # create a temporary networkx dodecahedral graph
        _tmp_nx = nx.generators.dodecahedral_graph()

        # relabel all edges
        edges = relabel_nx_edges(_tmp_nx)

        # if non-planar, relabel to out (x, y, z) labeling
        if non_planar:
            edges = validate_and_insert_keys(edges)
            edges, _ = relabel_edges_to_nonplanar(edges)

        super().__init__(
            edges,
            plot,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
        )
