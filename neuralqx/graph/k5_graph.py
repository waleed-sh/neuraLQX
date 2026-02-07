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


from .graph import Graph


class K5Graph(Graph):
    r"""
    Complete graph on five vertices :math:`K_5`.

    This graph has vertex set :math:`\{0,1,2,3,4\}` and an edge between every distinct pair, hence

    .. math::

        |V| = 5,\qquad |E| = \binom{5}{2} = 10.

    In neuraLQX this class provides a small, fixed test graph. When ``non_planar=True``, a
    pre-defined 3-tuple vertex labelling is used so that the non-planar pipeline (including the
    non-planar sign computation) can be exercised without relying on a planar embedding.

    :param plot: If True, produce a visualisation via the base graph machinery.
    :param non_planar: If True, use the pre-defined 3-tuple vertex labels for the non-planar path.
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

        # predefined edges
        if non_planar:
            k5_edges = [
                ((0, 0, 0), (0, 0, 1), 0),
                ((0, 0, 0), (0, 1, 0), 0),
                ((0, 0, 1), (1, 0, 0), 0),
                ((0, 1, 0), (0, 1, 1), 0),
                ((0, 1, 1), (1, 0, 0), 0),
                ((0, 0, 1), (0, 1, 0), 0),
                ((0, 0, 1), (0, 1, 1), 0),
                ((0, 1, 0), (1, 0, 0), 0),
                ((0, 0, 0), (0, 1, 1), 0),
                ((0, 0, 0), (1, 0, 0), 0),
            ]
        else:
            k5_edges = [
                (0, 1, 0),
                (0, 2, 0),
                (1, 3, 0),
                (2, 4, 0),
                (4, 3, 0),
                (1, 2, 0),
                (1, 4, 0),
                (2, 3, 0),
                (0, 4, 0),
                (0, 3, 0),
            ]

        super().__init__(
            k5_edges,
            plot=plot,
            random_embedding=random_embedding,
            random_embedding_std=random_embedding_std,
            random_embedding_mean=random_embedding_mean,
            random_embedding_seed=random_embedding_seed,
        )
