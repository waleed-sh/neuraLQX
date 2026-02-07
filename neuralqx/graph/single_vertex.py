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

from typing import Optional

from .graph import Graph
from .core.utils.build import relabel_edges_to_nonplanar
from .core.utils.checks import validate_and_insert_keys
from ..utils.errors import OrientationValenceMismatchError


class SingleVertexGraph(Graph):
    r"""
    Star graph with one central vertex and prescribed edge orientations.

    This graph has a distinguished central vertex labelled ``0`` and ``valence`` external vertices
    labelled ``1,2,\dots,valence``. For each external vertex :math:`i`, there is exactly one edge
    connecting :math:`0` and :math:`i`. The orientation of each edge is controlled by an integer
    direction :math:`d_i\in\{+1,-1\}`:

    - :math:`d_i=+1` gives an edge :math:`0\to i`,
    - :math:`d_i=-1` gives an edge :math:`i\to 0`.

    If ``orientation`` is not provided, all edges are oriented outwards (all :math:`d_i=+1`).

    If ``non_planar=True``, vertices are relabelled to fixed-length 3-tuples and may be given a
    random spatial embedding.

    :param valence: Number of edges incident at the central vertex.
    :param orientation: Optional list of length ``valence`` with entries in ``{+1, -1}``.
    :param plot: If True, produce a visualisation via the base graph machinery.
    :param non_planar: If True, relabel vertices to 3-tuples to trigger the non-planar pipeline.
    :param random_embedding: If True, assign a random 3D embedding to vertices (non-planar only).
    :param random_embedding_mean: Mean of the Gaussian used for the random embedding.
    :param random_embedding_std: Standard deviation of the Gaussian used for the random embedding.
    :param random_embedding_seed: Seed controlling the random embedding.
    :raises OrientationValenceMismatchError: If ``orientation`` is provided with the wrong length.
    """

    def __init__(
        self,
        valence: int,
        orientation: Optional[list] = None,
        *,
        plot: bool = False,
        non_planar: bool = False,
        random_embedding: bool = False,
        random_embedding_mean: float = 0.0,
        random_embedding_std: float = 5.0,
        random_embedding_seed: int = 123,
    ):

        if orientation is None:
            orientation = [1 for _ in range(valence + 1)]
        else:
            if len(orientation) != valence:
                raise OrientationValenceMismatchError(valence, len(orientation))

            orientation = [1] + orientation

        edges = [[0, i][::direction] for i, direction in enumerate(orientation)]

        edges.pop(edges.index([0, 0]))

        self._vertex_valence = valence
        self._orientation = orientation

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

    @property
    def orientation(self):
        return self._orientation

    @property
    def vertex_valence(
        self,
    ) -> int:
        return self._vertex_valence

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"n_edges={self.n_edges}, "
            f"n_vertices={self.n_vertices}, "
            f"is_planar={self.is_planar}, "
            f"vertex_valence={self.vertex_valence}, "
            f"orientation={self.orientation}"
            f")"
        )
