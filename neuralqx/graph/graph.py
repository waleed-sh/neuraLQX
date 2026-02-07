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
Concrete generic graph implementation.

This module defines :class:`~neuralqx.graph.graph.Graph`, the default concrete implementation of the
:class:`~neuralqx.graph.core.AbstractGraph` interface.

Design
The :class:`~neuralqx.graph.graph.Graph` class is intentionally thin. It delegates almost all
combinatorial and geometric work to the internal :class:`~neuralqx.graph._graph_handler.GraphHandler`
instance stored as :attr:`~neuralqx.graph.core.AbstractGraph.handler`.

In particular, the handler is responsible for

- building primal and dual graphs
- maintaining the bijection between keyed edges and integer indices
- extracting minimal cycles and dressing them with creation or annihilation metadata
- computing local sign factors used by volume type constructions

As a result, this class provides a stable public API while keeping implementation details contained
in the handler.
"""

from .core import AbstractGraph


class Graph(AbstractGraph):
    r"""
    A concrete directed multigraph wrapper used throughout neuraLQX.

    The :class:`Graph` class provides a user facing graph object that is compatible with neuraLQX
    Hilbert spaces, operators, and drivers. It exposes a small set of convenience methods and properties
    that forward to the underlying :class:`~neuralqx.graph._graph_handler.GraphHandler`.

    - Minimal loops
    Let :math:`G` be the underlying undirected graph obtained from the oriented edge set by forgetting
    direction and keys. A minimal loop in neuraLQX refers to a cycle in a minimum cycle basis of :math:`G`
    together with a consistent oriented, keyed edge representation.

    - Dual graph
    The dual graph used here is the line graph :math:`L(G)`. Its vertices correspond to (keyed) primal
    edges and its edges encode incidence in the primal graph.
    """

    def minimal_loops(self):
        r"""
        Return the minimal loops of the primal graph.

        In neuraLQX, a minimal loop is represented as an ordered list of keyed oriented edges

        .. math::

           \ell = \bigl[(u_0, u_1, k_0), (u_1, u_2, k_1), \dots, (u_{m-1}, u_0, k_{m-1})\bigr]

        such that successive edges share endpoints and the sequence closes.

        - How loops are chosen
        The underlying handler constructs an undirected simple graph by ignoring keys and orientations and
        then extracts a minimum cycle basis. In addition, it explicitly accounts for two edge cycles that
        arise from parallel edges in a multigraph.

        - Stability
        Each loop is canonically rotated so that the representation is stable under cyclic permutations.
        This improves reproducibility in logs and doctests.

        :return: A list of minimal loops. Each loop is a list of keyed oriented edges ``(u, v, key)``.
        """

        return self.handler.get_smallest_loops()

    @property
    def n_minimal_loops(self):
        r"""
        Return the number of minimal loops in the primal graph.

        This is a convenience property equivalent to ``len(self.minimal_loops())``. The value is computed
        from the handler data and therefore reflects the current graph structure.

        :return: The number of minimal loops.
        """

        return len(self.handler.get_smallest_loops())

    def minimal_dual_loops(self, edge_rep: bool = False):
        r"""
        Return the minimal loops of the dual graph.

        The dual graph is the line graph :math:`L(G)`. A dual loop can be represented either

        - as a list of keyed primal edges that form a closed walk in :math:`L(G)` when interpreted as dual
          vertices, or
        - as a list of dual vertex labels obtained via the mapping :math:`\mu` from keyed primal edges to
          integers

        .. math::

           \mu : E(G) \to \{0, 1, \dots, |E(G)|-1\}

        where :math:`E(G)` denotes the keyed oriented edge set.

        :param edge_rep: If True, return loops in keyed edge representation ``(u, v, key)``. If False,
          return loops in the dual integer vertex representation.
        :return: Minimal loops of the dual graph in the requested representation.
        :raises NotImplementedError: If the underlying handler does not provide dual loop extraction for
          the current model.
        """

        return self.handler.get_smallest_dual_loops(edge_rep)

    def dressed_minimal_loops(self):
        r"""
        Return dressed minimal loops of the primal graph.

        A dressed loop augments each edge in a minimal loop with metadata describing its operator type with
        respect to the loop orientation. Conceptually, this corresponds to choosing an orientation of the
        loop and assigning, for each traversed edge, whether the associated edge operator acts as a
        creation or annihilation along that traversal.

        :return: The list of dressed minimal loops stored by the handler.
        """

        return self.handler.minimal_cycles

    def dressed_minimal_dual_loops(self):
        r"""
        Return dressed minimal loops of the dual graph.

        This is the dual analogue of :meth:`~neuralqx.graph.graph.Graph.dressed_minimal_loops`. Each loop is
        represented in the handler chosen dual convention and each segment carries dressing metadata.

        Depending on the model, the dressing may encode how dual edge traversals map back to primal edge
        operators, for example when implementing dual plaquette moves or dual holonomy constructions.

        :return: The list of dressed minimal dual loops stored by the handler.
        """

        return self.handler.dual_minimal_cycles

    @property
    def nk_graph(self):
        r"""
        Return the NetKet graph of the primal graph.

        This is the NetKet representation used by samplers, Hilbert spaces, and operators that expect
        :class:`netket.graph.Graph`.

        :return: The primal NetKet graph.
        """

        return self.handler.nk_graph

    @property
    def dual_nk_graph(self):
        r"""
        Return the NetKet graph of the dual graph.

        The dual is constructed as the line graph of the primal NetworkX multigraph and then converted to a
        NetKet graph. Dual vertices correspond to keyed primal edges.

        :return: The dual NetKet graph.
        """

        return self.handler.dual_nk_graph

    @property
    def nx_graph(self):
        r"""
        Return the NetworkX multigraph of the primal graph.

        The NetworkX graph is used for connectivity queries, extracting cycle bases, and for attaching edge
        attributes used in loop dressing and sign computations.

        :return: The primal NetworkX multigraph.
        """

        return self.handler.nx_graph

    @property
    def dual_nx_graph(self):
        r"""
        Return the NetworkX graph of the dual graph.

        The dual NetworkX graph is the line graph of the primal multigraph. It is primarily used for
        diagnostics and for algorithms that naturally live on the dual, such as dual loop enumeration.

        :return: The dual NetworkX graph.
        """

        return self.handler.dual_nx_graph

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"n_edges={self.n_edges}, "
            f"n_vertices={self.n_vertices}, "
            f"n_minimal_loops={self.n_minimal_loops}, "
            f"is_planar={self.is_planar}"
            f")"
        )
