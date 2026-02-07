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

"""
Shared utilities for constrained and unconstrained Hilbert-space implementations.

This module collects small helper functions used across U(1) Hilbert-space cores and constraints,
including:

- PRNG-key aware mapping over batched arrays (:func:`avmapr`),
- normalization of single states to batched states (:func:`ensure_batchdim`),
- preprocessing of gauge-fixing constraints into integer/sign structures for fast kernels
  (:func:`preprocess_constraint`, :func:`preprocess_constraints_for_jax`),
- automatic generation of a gauge-fixing constraint array from an oriented edge list
  (:func:`generate_constraint_array`),
- human-readable formatting and optional non-planar coordinate remapping for constraints
  (:func:`pretty_format_constraints`, :func:`map_edge_string`).

All constraint preprocessing functions assume the *dual / edge-index* representation used by the
Hilbert cores (i.e., edges are referenced by integer indices or string forms of those indices).
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from collections import defaultdict
from collections import deque

import jax.random
import numpy as np

from neuralqx.debug import trace


def avmapr(f, x, key):  # pylint: disable=C0116
    """
    Apply a PRNG-keyed function to a state or a batch of states, recursively over leading axes.

    This helper is useful for "randomized per-state updates" where a function needs a JAX PRNGKey and
    should be applied independently to each element of a batch (and also to nested batches).

    Behaviour:
    - If ``x.ndim == 1``: call ``f(x, key)`` and return the result.
    - Otherwise: split ``key`` into ``x.shape[0]`` subkeys and ``vmap`` recursively over axis 0.

    This recursion makes it work for shapes like ``(B, N)`` as well as nested shapes such as
    ``(B, G, E)`` where you want independent randomness per element of the outermost batch.

    :param f: Callable with signature ``f(x_leaf, key_leaf)`` returning an updated array.
    :param x: Input array. If 1D, treated as a single state; otherwise treated as batched on axis 0.
    :param key: JAX PRNGKey used (and split) to provide independent randomness per batch element.
    :return: Output of applying ``f`` to ``x`` with appropriate key splitting and vectorization.
    :raises ValueError: If ``key`` cannot be split into the required number of subkeys.
    """

    if x.ndim == 1:
        return f(x, key)

    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(lambda y, k: avmapr(f, y, k))(x, keys)


def ensure_batchdim(x):
    """
    Ensure that an array has an explicit leading batch dimension.

    This utility normalizes state arrays to the common shape ``(B, N)``:

    - If ``x.ndim == 2``: return ``(x, False)`` (already batched).
    - Otherwise: return ``(x[None, :], True)`` (a single state promoted to batch size 1).

    The boolean flag is intended to support round-tripping: callers can later remove the added batch
    dimension if ``was_single`` is True.

    :param x: State array of shape ``(N,)`` or batched states of shape ``(B, N)``.
    :return: Tuple ``(xb, was_single)`` where ``xb`` has shape ``(B, N)`` and ``was_single`` indicates
        whether a batch dimension was added.
    :raises ValueError: If ``x`` is not 1D or 2D and cannot be interpreted as (single|batched) states.
    """

    return (x, False) if x.ndim == 2 else (x[None, :], True)


def preprocess_constraint(ce):
    """
    Preprocess a string-encoded gauge-fixing constraint list into integer/sign data.

    This function converts constraints of the form::

        [
            [["1"], ["2", "-0"]],
            [["3"], ["N", "-2", "5"]],
            ...
        ]

    into a list of ``(inc_data, out_data)`` pairs where:
    - ``inc_data`` is ``[(idx, sign), ...]`` for incoming edges,
    - ``out_data`` is ``[(idx, sign), ...]`` for outgoing edges,
    with ``idx`` as an integer edge index and ``sign`` in ``{+1, -1}``.

    Sign parsing:
    - A token like ``"-7"`` contributes ``(7, -1)``.
    - A token like ``"2"`` contributes ``(2, +1)``.
    - If the first outgoing token is exactly ``"N"``, it acts as a *global minus* for that constraint:
      all outgoing signs are multiplied by ``-1`` and the ``"N"`` token is consumed/removed.

    This representation is intended to be friendly to compiled kernels (e.g., Numba) and avoids any
    string parsing during hot loops.

    :param ce: List of constraint pairs ``[[inc_tokens], [out_tokens]]`` where tokens are strings like
        ``"2"``, ``"-0"``, or ``"N"``.
    :return: List of processed constraints ``[(inc_data, out_data), ...]`` with integer indices and
        explicit ±1 signs.
    :raises ValueError: If any non-``"N"`` token cannot be converted to an integer index.
    """

    processed = []
    for pair in ce:
        inc_strings = pair[0]
        out_strings = pair[1]

        # parse incoming edges
        inc_data = []
        for s in inc_strings:
            sgn = -1 if s.startswith("-") else 1
            idx_str = s.lstrip("-")
            idx = int(idx_str)
            inc_data.append((idx, sgn))

        # parse outgoing edges, but first handle 'N' if it’s a global minus
        local_sgn = 1
        out_data = []

        if len(out_strings) > 0 and out_strings[0] == "N":
            local_sgn = -1
            # remove the 'N' token entirely
            out_strings = out_strings[1:]

        for s in out_strings:
            sgn = -1 if s.startswith("-") else 1
            idx_str = s.lstrip("-")
            idx = int(idx_str)

            # multiply sgn by the local_sgn to reflect the global minus
            out_data.append((idx, local_sgn * sgn))

        processed.append((inc_data, out_data))

    return processed


def preprocess_constraints_for_jax(ce):
    """
    Preprocess a string-encoded gauge-fixing constraint list into a JAX-friendly Python structure.

    This function performs the same sign/index conversion as :func:`preprocess_constraint`, but returns
    a structure designed to be consumed by JAX code that iterates in Python over a fixed-size list:

    - The returned value is a plain Python ``list`` of pairs ``(inc_data, out_data)``.
    - Each of ``inc_data`` and ``out_data`` is a Python ``list`` of ``(idx, sign)`` tuples.

    The outgoing global-minus token ``"N"`` is supported:
    - If the first outgoing token is ``"N"``, all outgoing signs are multiplied by ``-1`` and ``"N"``
      is removed from the outgoing list.

    This representation is typically fed into a JAX-jitted constraint checker that loops over the list
    and builds small JAX arrays inside the loop.

    :param ce: List of constraint pairs ``[[inc_tokens], [out_tokens]]`` where tokens are strings like
        ``"2"``, ``"-0"``, or ``"N"``.
    :return: List of processed constraints suitable for JAX-friendly loops:
        ``[(inc_data, out_data), ...]`` with ``inc_data/out_data`` as lists of ``(idx, ±1)``.
    :raises ValueError: If any non-``"N"`` token cannot be converted to an integer index.
    """

    processed = []
    for pair in ce:
        inc_strings = pair[0]
        out_strings = pair[1]

        inc_data = []
        for s in inc_strings:
            sgn = -1 if s.startswith("-") else 1
            idx = int(s.lstrip("-"))
            inc_data.append((idx, sgn))

        local_sgn = 1
        if len(out_strings) > 0 and out_strings[0] == "N":
            local_sgn = -1
            out_strings = out_strings[1:]

        out_data = []
        for s in out_strings:
            sgn = -1 if s.startswith("-") else 1
            idx = int(s.lstrip("-"))
            out_data.append((idx, local_sgn * sgn))

        processed.append((inc_data, out_data))

    return processed


@trace(tag="AUTO_CONSTRAINT")
def generate_constraint_array(edges: List[Tuple]) -> List[List[List[str]]]:
    """
    Generate a gauge-fixing constraint array from an oriented multigraph edge list.

    Given an oriented, keyed edge list describing a connected undirected graph, this function builds a
    spanning tree and expresses each tree edge as an integer-signed sum of chord edges (non-tree edges).
    The resulting constraints can be used as a gauge fixing that enforces vertex-wise conservation
    (Kirchhoff / Gauß law) when interpreted as modular equalities in the U(1) Hilbert space.

    Output format:
    - Returns a list of constraints, each of the form ``[[lhs_edge_str], rhs_edge_strs]`` where:
      - ``lhs_edge_str`` is a single string representing the fixed (tree) edge,
      - ``rhs_edge_strs`` is a list of signed strings representing chord edges contributing to the RHS.

    Each edge is rendered as ``"(u, v, key)"`` using the provided edge tuples.

    Algorithm overview:
    1) Collect and index vertices, build the incidence matrix ``B`` (rows: vertices, cols: edges).
    2) Drop one row to obtain full rank (connected graphs have rank ``n-1``).
    3) Build a spanning tree via BFS to select ``n-1`` tree edges.
    4) Partition edges into tree edges ``T`` and chord edges ``C``.
    5) Solve the linear system to express ``f_T`` in terms of ``f_C`` and round coefficients to integers.
    6) Emit one constraint per tree edge.

    :param edges: List of oriented keyed edges ``(u, v, key)``. The graph is treated as undirected for
        connectivity/spanning-tree construction, but orientation is used in the incidence matrix.
    :return: Gauge-fixing constraint array as ``list[list[list[str]]]`` suitable for downstream parsing
        and pretty-printing.
    :raises TypeError: If vertex labels are not sortable/comparable (required by the current
        implementation’s use of ``sorted(...)``).
    :raises ValueError: If the underlying graph is disconnected or the spanning tree cannot be built
        (may lead to an invalid solve).
    :raises numpy.linalg.LinAlgError: If the tree incidence submatrix is singular (typically indicates
        the graph is not connected or the constructed tree is invalid).
    """

    #
    #
    #   collect and index vertices

    # extract all unique vertex labels from the edge list
    verts = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})

    # map each vertex label to a numeric index (for matrix representation)
    v2i = {v: i for i, v in enumerate(verts)}

    # number of vertices (n) and number of edges (m)
    n, m = len(verts), len(edges)

    #
    #
    #   build incidence matrix

    # initialise incidence matrix B with zeros
    # B has shape (n vertices) x (m edges)
    B = np.zeros((n, m), dtype=int)

    # fill matrix according to orientation: +1 for source vertex, -1 for target vertex
    for j, (u, v, k) in enumerate(edges):
        B[v2i[u], j] = 1
        B[v2i[v], j] = -1

    #
    #
    #   reduce matrix to full-rank form

    # remove one row from B to obtain full-rank (n-1) x m matrix Br
    # this avoids singularity (since B has rank n-1 for a connected graph)
    Br = np.delete(B, n - 1, axis=0)

    #
    #
    #   build spanning tree structures

    # initialise adjacency list (for BFS traversal)
    adj = defaultdict(list)

    # map unordered vertex pairs -> list of edge indices (for multigraph support)
    pair_to_edges = defaultdict(list)

    # populate adjacency and pair-to-edge mappings
    for j, (u, v, k) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
        pair_to_edges[(min(u, v), max(u, v))].append(j)

    #
    #
    #   construct spanning tree using BFS

    # set of visited vertices
    seen = set()

    # start BFS from the first vertex in sorted order
    start = verts[0]
    seen.add(start)

    # BFS queue initialised with the start vertex
    q = deque([start])

    # List to store indices of tree edges (spanning tree)
    tree_edges_idx = []

    # set to track unordered pairs already used as tree edges
    used_pairs = set()

    # BFS loop, continue until all vertices are connected
    while q and len(tree_edges_idx) < max(0, n - 1):

        # pop next vertex to explore
        x = q.popleft()

        # explore all adjacent vertices
        for y in adj[x]:

            # found a new vertex
            if y not in seen:
                seen.add(y)
                q.append(y)
                key = (min(x, y), max(x, y))

                if key in used_pairs:
                    # skip if already used
                    continue
                used_pairs.add(key)

                # select the first encountered edge index between x and y
                tree_edges_idx.append(pair_to_edges[key][0])

    #
    #
    #   partition edges into tree and chords

    # keep exactly n-1 tree edges
    tree_edges_idx = tree_edges_idx[: max(0, n - 1)]

    # chord edges = all edges not in the spanning tree
    chord_edges_idx = [j for j in range(m) if j not in tree_edges_idx]

    #
    #
    #   solve linear system for tree edges

    # split reduced incidence matrix Br into:
    #  - Br_T: columns corresponding to tree edges
    #  - Br_C: columns corresponding to chord edges

    # (n-1) x (n-1)
    Br_T = Br[:, tree_edges_idx]

    # (n-1) x (m-(n-1))
    Br_C = Br[:, chord_edges_idx]

    # solve Br_T * f_T + Br_C * f_C = 0  ->  f_T = - (Br_T)^(-1) * Br_C * f_C
    A = -np.linalg.solve(Br_T.astype(float), Br_C.astype(float))

    # round numerical results to nearest integers (should be in {-1, 0, 1})
    A_int = np.rint(A).astype(int)

    #
    #
    #   construct human-readable constraints

    def edge_str(e):
        """Convert edge tuple into readable string form."""
        return f"({e[0]}, {e[1]}, {e[2]})"

    # initialise list of constraints to be returned
    constraints = []

    # iterate over tree edges (each becomes one constraint)
    for r, t_idx in enumerate(tree_edges_idx):

        # the fixed tree edge (to be expressed as a sum of chords)
        fixed_edge = edges[t_idx]

        # list of dependencies (chord edges with ± signs)
        deps = []

        # loop over chord coefficients in the solved matrix A_int
        for c, coef in enumerate(A_int[r, :]):

            if coef == 0:
                # skip if no contribution
                continue

            # corresponding chord edge
            e = edges[chord_edges_idx[c]]

            # convert to string
            s = edge_str(e)

            # add with proper sign or multiplier
            if coef == 1:
                deps.append(s)
            elif coef == -1:
                deps.append("-" + s)
            else:
                deps.append(f"{coef}*" + s)

        # append constraint: [fixed_edge, list_of_signed_chords]
        constraints.append([[edge_str(fixed_edge)], deps])

    return constraints


def map_edge_string(edge_str, coords_to_int):
    """
    Rewrite vertex-coordinate tuples inside an edge string using a provided coordinate mapping.

    This helper supports pretty-printing constraints for non-planar embeddings where vertices may be
    represented by coordinate triples. It scans an ``edge_str`` for parenthesized tuples, attempts to
    parse them as comma-separated floats, and replaces any tuple present in ``coords_to_int`` by its
    mapped vertex identifier.

    The function is intentionally conservative:
    - If a matched tuple cannot be parsed as floats, it is left unchanged.
    - Only tuples that appear as keys in ``coords_to_int`` are replaced.

    :param edge_str: Edge string potentially containing coordinate tuples, e.g.
        ``"((1.1, 2.2, 3.3), (4.4, 5.5, 6.6), 0)"``.
    :param coords_to_int: Mapping from coordinate tuples to vertex identifiers (e.g. ints or names).
    :return: A rewritten string where recognized coordinate tuples have been replaced by their mapped
        identifiers.
    :raises None: This function does not intentionally raise exceptions; unparsable tuples are skipped.
    """

    import re

    tuples = re.findall(r"\(([^()]+)\)", edge_str)
    replaced = edge_str

    for t in tuples:
        try:
            coords = tuple(map(float, t.split(",")))
            rounded = tuple(x for x in coords)
            if rounded in coords_to_int:
                vertex_id = coords_to_int[rounded]
                replaced = replaced.replace(f"({t})", str(vertex_id))
        except ValueError:
            continue

    return replaced


def pretty_format_constraints(
    constraints: list[list[list[str]]],
    coordinates_map: Optional[Dict] = None,
) -> str:
    """
    Format gauge-fixing constraints into a human-readable multi-line string.

    Each constraint is expected to have the form::

        [
            [fixed_edge_as_string],
            [signed_edge_strings...]
        ]

    This function formats each constraint as a short description:

    - The LHS edge is shown as the "fixed" edge.
    - The RHS is printed as a signed expression, one line per constraint.
    - If ``coordinates_map`` is provided, coordinate tuples in edge strings are rewritten using
      :func:`map_edge_string` to recover original vertex identifiers.

    Robustness:
    - If the constraint list is empty, a short message is returned.
    - Malformed or incomplete constraint entries are reported inline in the output.

    :param constraints: List of constraints in the string-array format produced by
        :func:`generate_constraint_array` (or equivalent).
    :param coordinates_map: Optional mapping from vertex identifiers to coordinate tuples. If provided,
        a reverse map is built and used to replace coordinate tuples in edge strings with vertex ids.
    :return: A formatted string suitable for logging or displaying autogenerated gauge fixings.
    :raises TypeError: If ``coordinates_map`` contains unhashable coordinate values when building the
        reverse lookup map.
    """

    # return message for empty or None input
    if not constraints:
        return "No constraints found."

    reverse_map = None
    if coordinates_map is not None:
        reverse_map = {tuple(x for x in v): k for k, v in coordinates_map.items()}

    lines = []

    for constraint in constraints:
        # validate constraint structure
        if not isinstance(constraint, list) or len(constraint) != 2:
            lines.append(f"Malformed constraint: {constraint}")
            continue

        lhs_list, rhs_list = constraint

        # skip incomplete entries
        if not lhs_list or not rhs_list:
            lines.append(f"Incomplete constraint: {constraint}")
            continue

        # extract the left-hand side edge string (fixed edge)
        lhs = lhs_list[0]

        # build the right-hand side expression
        rhs_pretty_terms = []
        for term in rhs_list:

            # handle leading signs and coefficients
            term_str = term.strip()

            if reverse_map:
                term_str = map_edge_string(term_str, reverse_map)

            if term_str.startswith("-"):
                rhs_pretty_terms.append(f"- {term_str[1:].strip()}")
            elif term_str.startswith("+"):
                rhs_pretty_terms.append(f"+ {term_str[1:].strip()}")
            else:
                rhs_pretty_terms.append(f"+ {term_str}")

        # join terms and clean up the leading "+"
        rhs_str = " ".join(rhs_pretty_terms).lstrip("+").strip()

        if reverse_map:
            lhs = map_edge_string(lhs, reverse_map)

        # construct a readable sentence
        lines.append(f"Edge {lhs} is fixed by:\n{rhs_str}\n")

    return "\n".join(lines)
