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


class CircularLadderGraph(Graph):
    r"""
    Circular ladder graph :math:`C_N \,\square\, K_2`.

    This graph consists of two concentric :math:`N`-cycles ("rails") together with :math:`N` rungs
    joining corresponding vertices on the two rails. Equivalently, its vertex set can be written as

    .. math::

        V = \{0,1,\dots,N-1\} \times \{0,1\},

    with edges
    :math:`(i,a)\sim(i\pm1 \!\!\!\pmod N, a)` (cycle edges) and :math:`(i,0)\sim(i,1)` (rungs).

    Basic counts (for :math:`N\ge 3`):

    .. math::

        |V| = 2N,\qquad |E| = 3N.

    The underlying NetworkX graph is relabelled to consecutive integer vertex labels before being
    passed to :class:`~neuralqx.graph.graph.Graph`. If ``non_planar=True``, vertices are instead
    relabelled to fixed-length 3-tuples and can optionally be given a random spatial embedding.

    :param N: Ladder length (number of rungs / cycle length).
    :param plot: If True, produce a visualisation via the base graph machinery.
    :param non_planar: If True, relabel vertices to 3-tuples to trigger the non-planar pipeline.
    :param random_embedding: If True, assign a random 3D embedding to vertices (non-planar only).
    :param random_embedding_mean: Mean of the Gaussian used for the random embedding.
    :param random_embedding_std: Standard deviation of the Gaussian used for the random embedding.
    :param random_embedding_seed: Seed controlling the random embedding.
    """

    def __init__(
        self,
        N: int,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int = 123,
    ):

        # create a temporary networkx circular ladder graph
        _tmp_nx = nx.generators.circular_ladder_graph(N)

        # relabel all edges
        edges = relabel_nx_edges(_tmp_nx)

        # if non-planar, relabel to out (x, y, z) labeling
        if non_planar:
            edges = validate_and_insert_keys(edges)
            edges, _ = relabel_edges_to_nonplanar(edges)

        self._ladder_length = N

        super().__init__(
            edges,
            plot,
            random_embedding=random_embedding,
            random_embedding_mean=random_embedding_mean,
            random_embedding_std=random_embedding_std,
            random_embedding_seed=random_embedding_seed,
        )

    @property
    def ladder_length(self):
        return self._ladder_length

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"n_edges={self.n_edges}, "
            f"n_vertices={self.n_vertices}, "
            f"ladder_length={self.ladder_length}, "
            f"n_minimal_loops={self.n_minimal_loops}, "
            f"is_planar={self.is_planar}"
            f")"
        )
