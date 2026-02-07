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
Gauge-fixed (constrained) U(1) Hilbert-space core and gauge-fixing utilities.

This module implements a gauge-invariant (but gauge-*fixed*) Hilbert-space backend for U(1) edge
degrees of freedom on a graph. The central class is :class:`ConstrainedHilbertU1Core`, which:

- interprets a user-provided gauge-fixing array (or auto-generates one from the graph),
- identifies *free* (independent) and *slave* (dependent) edge variables,
- provides a deterministic projector :meth:`ConstrainedHilbertU1Core.reimpose_gauge_fixing` that
  overwrites slave values to satisfy all gauge-fixing relations,
- constructs a NetKet :class:`netket.hilbert.HomogeneousHilbert` with a :class:`U1Constraint`
  enforcing the gauge fixing across all gauge copies,
- exposes JAX-friendly move proposals via typed :class:`~neuralqx.hilbert.operations.moves.Move`
  objects (free-edge flips and plaquette flips), with dispatch implemented using
  :func:`functools.singledispatchmethod`.

The helper functions in this module focus on parsing and normalising edge tokens, mapping them to
base edge indices, converting a gauge-fixing array into internal constraint objects, and producing a
topological ordering of dependencies so slave edges can be recomputed in a valid order.
"""

from __future__ import annotations

import logging

from collections import defaultdict
from collections import deque

from dataclasses import dataclass
from functools import singledispatchmethod

from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

import humanize
import jax
import jax.numpy as jnp
import netket as nk

from neuralqx.graph.core import AbstractGraph
from neuralqx.utils.errors import AutoConstraintGaugeFixingConflictError
from neuralqx.utils.errors import CyclicGaugeFixingError
from neuralqx.utils.errors import IncorrectGaugeFixingArrayError
from neuralqx.utils.errors import InvalidCutoffError
from neuralqx.utils.errors import InvalidFreeEdgeSelectionError
from neuralqx.utils.errors import InvalidIndexError

from neuralqx.utils.parsing import str_edge_parser
from neuralqx.utils.misc.arithmetic import mod_add
from ..abstract_hilbert_core import AbstractHilbertSpace
from ..constraints import U1Constraint

from ..constraints.utils import generate_constraint_array
from ..constraints.utils import pretty_format_constraints

from neuralqx.hilbert.operations.moves import Move
from neuralqx.hilbert.operations.moves import FreeEdgeFlipSingleGauge
from neuralqx.hilbert.operations.moves import FreeEdgeFlipAllGauge
from neuralqx.hilbert.operations.moves import PlaquetteFlipSingleGauge
from neuralqx.hilbert.operations.moves import PlaquetteFlipAllGauge

from neuralqx.debug import errors_only, event

EdgeToken = Union[int, str, Tuple, List]


@dataclass(frozen=True, slots=True)
class _Constraint:
    """
    Internal representation of one gauge-fixing relation on base edge indices.

    A constraint encodes a single deterministic relation of the form::

        lhs = sum_i (sign_i * rhs_i)   (mod q-range/step),

    where all indices refer to *base-copy* edge indices in the range ``[0, E)`.

    :param lhs: Base edge index of the slave variable (left-hand side).
    :param rhs: Tuple of ``(index, sign)`` pairs describing dependencies, where ``sign`` is ``+1`` or
        ``-1``.
    :return: None.
    :raises None: This dataclass does not intentionally raise exceptions during normal construction.
    """

    lhs: int
    rhs: Tuple[Tuple[int, int], ...]  # ((idx, sign), ...)


@dataclass(frozen=True, slots=True)
class GaugeFixingTopo:
    """
    Topological gauge-fixing description on base edge indices ``0...E-1``.

    This object stores the *structure* of a gauge fixing after parsing and dependency analysis:

    - ``free``: indices of free (independent) edges; these parameterise the constrained space.
    - ``slave_topo``: slave edges ordered so that each slave's dependencies have been handled earlier
      in the sequence.
    - ``slave_fixers``: RHS dependency lists aligned with ``slave_topo``; the ``i``-th entry contains
      the fixers needed to compute ``slave_topo[i]``.

    This ordering is crucial for deterministic reconstruction: when free edges change, slave edges must
    be recomputed in an order that respects the dependency DAG.

    :param free: Sorted tuple of free base-edge indices.
    :param slave_topo: Tuple of slave indices in a valid topological order.
    :param slave_fixers: Tuple of RHS fixers aligned with ``slave_topo``; each element is a tuple of
        ``(idx, sign)`` pairs.
    :return: None.
    :raises None: This dataclass does not intentionally raise exceptions during normal construction.
    """

    free: Tuple[int, ...]
    slave_topo: Tuple[int, ...]
    slave_fixers: Tuple[Tuple[Tuple[int, int], ...], ...]  # aligned with slave_topo

    @property
    def n_free(self) -> int:
        """
        Number of free (independent) base edges.

        :return: The length of :attr:`free`.
        :raises None: This property does not intentionally raise exceptions.
        """

        return len(self.free)


def _is_intlike(x: object) -> bool:
    """
    Return True if the input behaves like an integer *index* for gauge-fixing parsing.

    This helper is intentionally strict:

    - Accepts Python :class:`int`.
    - Rejects :class:`bool` (since ``bool`` is a subclass of ``int`` in Python).
    - Rejects non-integer structured edge tokens (tuples/lists/strings).

    It is used to distinguish direct index references (already in base-index space) from edge descriptors
    that must be normalised and mapped through the graph.

    :param x: Candidate object to test.
    :return: ``True`` if ``x`` is an ``int`` and not a ``bool``, otherwise ``False``.
    :raises None: This function does not intentionally raise exceptions.
    """

    return isinstance(x, (int,)) and not isinstance(x, bool)


def _maybe_int_from_string(s: str) -> Optional[int]:
    """
    Attempt to parse an integer from a string.

    Accepted forms include:

    - ``"123"``
    - ``"+123"``
    - ``"-123"``

    Whitespace around the token is ignored. Any other content (including empty/whitespace-only strings
    or mixed alphanumeric strings) is rejected.

    This is used to allow gauge-fixing tokens like ``"17"`` or ``"-3"`` to be treated as direct indices
    rather than structured edge descriptors.

    :param s: Input string token.
    :return: Parsed integer value if ``s`` represents an integer, otherwise ``None``.
    :raises None: This function does not intentionally raise exceptions.
    """

    ss = s.strip()
    if ss.startswith(("+", "-")):
        core = ss[1:]
        if core.isdigit():
            return int(ss)
        return None
    if ss.isdigit():
        return int(ss)
    return None


def _canonical_edge_for_mapping(g: AbstractGraph, edge: EdgeToken) -> Union[int, Tuple]:
    """
    Canonicalise an edge token into the key-space expected by ``graph.edge_to_index``.

    Supported input forms:

    - Integer-like tokens: treated as base indices and returned as ``int``.
    - Strings:
      - If the string parses as an integer (e.g. ``"12"``), it is returned as an ``int``.
      - Otherwise it is parsed as an edge descriptor via :func:`neuralqx.utils.parsing.str_edge_parser`.
    - Tuple/list edge descriptors such as ``(u, v)`` or ``(u, v, k)`` for multi-edges.

    Planar graphs
        The edge descriptor is returned as a tuple with endpoints unchanged.

    Non-planar graphs
        Endpoints may be coordinate-like objects. Any endpoint that is not int-like is mapped through
        ``graph.nonplanar_vertex_to_index``. Additional fields (e.g. a multi-edge key) are preserved.

    This function does not call ``edge_to_index`` itself; it only normalises the token into a canonical
    key.

    :param g: Graph that defines planar/non-planar conventions and provides vertex mapping utilities.
    :param edge: Edge token (int, string, tuple, or list).
    :return: Canonical edge key: either an ``int`` (already an index) or a tuple suitable for
        ``g.edge_to_index``.
    :raises IncorrectGaugeFixingArrayError: If the token cannot be interpreted as an edge descriptor
        (invalid type or too few fields).
    :raises TypeError: If a non-planar endpoint cannot be mapped to an integer index and later cannot
        be converted to ``int``.
    """

    if _is_intlike(edge):
        return int(cast(int, edge))

    if isinstance(edge, str):
        maybe = _maybe_int_from_string(edge)
        if maybe is not None:
            return maybe
        edge = str_edge_parser(edge)

    if not isinstance(edge, (tuple, list)):
        raise IncorrectGaugeFixingArrayError(
            f"Invalid edge token type: {type(edge)} (value={edge!r})"
        )

    t = tuple(edge)
    if len(t) < 2:
        raise IncorrectGaugeFixingArrayError(
            f"Edge must have at least 2 entries: {t!r}"
        )

    if g.is_planar:
        return t

    u, v = t[0], t[1]
    if not _is_intlike(u):
        u = g.nonplanar_vertex_to_index(u)
    if not _is_intlike(v):
        v = g.nonplanar_vertex_to_index(v)

    if len(t) == 2:
        return (int(u), int(v))

    return (int(u), int(v), *t[2:])


def _edge_to_index_safe(g: AbstractGraph, edge: EdgeToken) -> int:
    """
    Resolve an arbitrary edge token into a base edge index in ``[0, E)``.

    Resolution rules:

    - If the token is int-like, it is returned unchanged as an integer index.
    - Otherwise, the token is canonicalised via :func:`_canonical_edge_for_mapping` and then mapped to a
      base edge index via ``graph.edge_to_index``.

    Graph-indexing failures (e.g. unknown edges) are translated into
    :class:`IncorrectGaugeFixingArrayError` with additional context, since gauge-fixing errors should be
    diagnosed at the constraint level rather than as low-level graph mapping failures.

    :param g: Graph providing ``edge_to_index``.
    :param edge: Edge token in any supported format (int, string, tuple, list).
    :return: Base edge index as an integer.
    :raises IncorrectGaugeFixingArrayError: If the edge cannot be resolved in the graph mapping or the
        token is malformed.
    """

    key = _canonical_edge_for_mapping(g, edge)
    if _is_intlike(key):
        return int(cast(int, key))

    try:
        return int(g.edge_to_index(cast(Tuple, key)))
    except InvalidIndexError as e:
        raise IncorrectGaugeFixingArrayError(
            "Gauge fixing references an edge that cannot be resolved in the graph mapping.\n"
            f"  raw_token={edge!r}\n"
            f"  canonical_key={key!r}\n"
            f"  original_error={e}"
        ) from e


def _parse_gauge_fixing_to_constraints(
    g: AbstractGraph,
    gauge_fixing: List[List[List[Union[int, str]]]],
) -> Tuple[_Constraint, ...]:
    """
    Parse the user gauge-fixing array into internal base-index constraints.

    Input format
        ``gauge_fixing`` is a list of gauge conditions. Each condition has the shape::

            [lhs_list, rhs_list]

        where:

        - ``lhs_list`` must be a list of length 1 describing the *slave* edge.
        - ``rhs_list`` is a list of tokens describing dependencies (the *fixers*).

    Sign conventions on the RHS
        The RHS parsing supports two mechanisms that can flip signs:

        - A literal token ``"N"`` switches on subtract-mode for subsequent RHS entries.
        - A leading ``"-"`` on a token flips the sign for that token.

        If both apply, the sign flips twice (net ``+1``).

    Output representation
        Each parsed constraint is stored as a :class:`_Constraint`:

        - ``lhs``: base edge index of the slave variable.
        - ``rhs``: tuple of ``(index, sign)`` dependency pairs, where ``sign`` is ``+1`` or ``-1``.

    :param g: Graph used to map edge tokens to base indices.
    :param gauge_fixing: Gauge fixing array in the user-facing format.
    :return: Tuple of :class:`_Constraint` objects in base-index space.
    :raises IncorrectGaugeFixingArrayError: If the gauge-fixing array is malformed, has empty RHS,
        repeats a LHS edge, or references edges that cannot be resolved in the graph.
    """

    if gauge_fixing is None:
        raise IncorrectGaugeFixingArrayError("Gauge fixing is None.")

    constraints: List[_Constraint] = []
    for cond in gauge_fixing:
        if not isinstance(cond, list) or len(cond) != 2:
            raise IncorrectGaugeFixingArrayError(
                "Each gauge condition must be a list: [lhs_list, rhs_list]."
            )
        lhs_list, rhs_list = cond[0], cond[1]
        if not isinstance(lhs_list, list) or len(lhs_list) != 1:
            raise IncorrectGaugeFixingArrayError(
                "Each gauge condition must have exactly one incoming edge on the LHS."
            )
        lhs_idx = _edge_to_index_safe(g, lhs_list[0])

        subtract_mode = False
        rhs_pairs: List[Tuple[int, int]] = []
        for token in rhs_list:
            if isinstance(token, str) and token.strip() == "N":
                subtract_mode = True
                continue

            s = str(token).strip()
            explicit_minus = s.startswith("-")
            if explicit_minus:
                s = s[1:].strip()

            ridx = _edge_to_index_safe(g, s)

            flips = int(subtract_mode) + int(explicit_minus)
            sign = -1 if (flips % 2 == 1) else +1
            rhs_pairs.append((ridx, sign))

        if not rhs_pairs:
            raise IncorrectGaugeFixingArrayError(
                f"Gauge condition for lhs index {lhs_idx} has an empty RHS."
            )

        constraints.append(_Constraint(lhs=lhs_idx, rhs=tuple(rhs_pairs)))

    lhs_all = [c.lhs for c in constraints]
    if len(lhs_all) != len(set(lhs_all)):
        raise IncorrectGaugeFixingArrayError("Duplicate LHS edges in gauge fixing.")

    return tuple(constraints)


def _topological_ordering(constraints: Sequence[_Constraint]) -> GaugeFixingTopo:
    """
    Compute a dependency-aware ordering of gauge-fixing constraints and identify free edges.

    Each constraint is interpreted as directed dependencies:

    - Each RHS edge points to its LHS edge (RHS → LHS).
    - Free edges are those with in-degree 0 in the resulting dependency graph.
    - Slave edges must be updated in a topological order so that all dependencies of a slave are already
      up-to-date when the slave is recomputed.

    The result is returned as :class:`GaugeFixingTopo` containing:

    - ``free``: sorted tuple of all in-degree-0 nodes,
    - ``slave_topo``: slave edges in a valid topological order,
    - ``slave_fixers``: RHS fixers aligned with ``slave_topo``.

    :param constraints: Sequence of parsed :class:`_Constraint` relations in base-index space.
    :return: A :class:`GaugeFixingTopo` instance describing free/slave structure and update order.
    :raises CyclicGaugeFixingError: If the dependency graph contains a cycle (no valid topological
        ordering exists).
    """

    deps = defaultdict(list)
    indeg = defaultdict(int)
    nodes = set()

    slave_fixers_map: dict[int, Tuple[Tuple[int, int], ...]] = {}
    for c in constraints:
        nodes.add(c.lhs)
        slave_fixers_map[c.lhs] = c.rhs
        indeg.setdefault(c.lhs, 0)
        for ridx, _sgn in c.rhs:
            nodes.add(ridx)
            indeg.setdefault(ridx, 0)
            deps[ridx].append(c.lhs)
            indeg[c.lhs] += 1

    free = tuple(sorted([n for n in nodes if indeg[n] == 0]))

    q = deque([n for n in nodes if indeg[n] == 0])
    topo: list[int] = []
    while q:
        n = q.popleft()
        topo.append(n)
        for child in deps.get(n, []):
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)

    if len(topo) != len(nodes):
        raise CyclicGaugeFixingError

    slave_topo = [n for n in topo if n in slave_fixers_map]
    slave_fixers = [slave_fixers_map[n] for n in slave_topo]

    return GaugeFixingTopo(
        free=tuple(free),
        slave_topo=tuple(slave_topo),
        slave_fixers=tuple(slave_fixers),
    )


class ConstrainedHilbertU1Core(AbstractHilbertSpace):
    """
    Gauge-invariant (gauge-fixed) :math:`U(1)` Hilbert-space core.

    This core implements a *constrained* Hilbert space of :math:`U(1)` edge degrees of freedom by
    enforcing a user-specified (or auto-generated) gauge fixing. In practice, the gauge fixing is a
    collection of deterministic relations that identify a subset of edges as:

    Free edges
        Independent degrees of freedom that parameterise the constrained space. Free edges are the only
        variables that are directly updated by move proposals in this core.

    Slave edges
        Dependent degrees of freedom that are recomputed deterministically from free edges (and possibly
        other slaves) via the gauge-fixing relations. Slave edges are never sampled independently.

    Configuration layout
        The full state ``sigma`` is stored as a flat vector of length ``N = G * E`` where:

        - ``E = tiny_size`` is the number of base edge sites per gauge copy,
        - ``G = gauge_dimensions`` is the number of gauge copies.

        Internally, the core frequently reshapes to ``(B, G, E)`` for batch operations.

    Gauge-fixing pipeline
        During construction, the core:

        1. Validates that ``cutoff`` and ``step`` are integer-valued for this constrained implementation.
        2. Resolves the gauge-fixing specification:
           - if ``auto_constraint=True``, it generates a gauge fixing from the graph,
           - otherwise it parses the provided ``constraint`` array.
        3. Converts the gauge fixing to internal :class:`_Constraint` objects on base indices.
        4. Computes a topological ordering of dependencies via :func:`_topological_ordering` to obtain
           ``free`` and a valid slave update order.
        5. Lifts the base-copy constraints across all gauge copies and constructs a NetKet
           :class:`netket.hilbert.HomogeneousHilbert` with a :class:`U1Constraint` enforcing those
           relations.
        6. Precomputes plaquette (minimal-loop) data in a JAX-safe padded form to support JIT-compiled
           plaquette moves.

    Deterministic projection
        The method :meth:`reimpose_gauge_fixing` overwrites slave edges (in topological order) to satisfy
        all constraints. This allows the core to:

        - generate gauge-consistent random states by sampling arbitrary values and then projecting,
        - apply updates to free edges and then re-project to restore the constraints.

    Move proposals
        Proposals are expressed via typed :class:`~neuralqx.hilbert.operations.moves.Move` objects and
        applied through :meth:`propose`. Supported move families include:

        - Free-edge flips (single-gauge selection space or per-gauge-copy updates),
        - Plaquette flips (single-gauge or all-gauge), implemented as ±step updates around minimal loops.

    Dimension reporting
        If the underlying NetKet Hilbert is indexable, the exact dimension is taken from
        ``hilbert.n_states``. Otherwise, an estimate based on the number of free edges is used, and a
        scientific-notation string is cached for readable printing.

    :param graph: Graph defining edge degrees of freedom and index mappings.
    :param cutoff: Integer cutoff controlling the local basis range for U(1) labels.
    :param step: Integer step size between allowed basis labels.
    :param gauge_dimensions: Number of gauge copies in the flattened layout.
    :param constraint: User gauge-fixing array, or ``None`` when using ``auto_constraint=True``.
    :param auto_constraint: If ``True``, auto-generate gauge fixing from the graph.
    :param positive_qn: If ``True``, restrict local labels to a non-negative (or shifted) range.
    :param qn_start: Optional starting label when ``positive_qn=True``.
    :return: None.
    :raises InvalidCutoffError: If ``cutoff`` or ``step`` is provided as a float in this constrained
        implementation.
    :raises AutoConstraintGaugeFixingConflictError: If both ``auto_constraint=True`` and an explicit
        ``constraint`` array are provided.
    :raises IncorrectGaugeFixingArrayError: If the gauge-fixing array is malformed or references
        unmappable edges.
    :raises CyclicGaugeFixingError: If the gauge-fixing dependency graph contains a cycle.
    """

    __slots__ = (
        "_gf",
        "_constraints",
        "_constraints_full",
        "_print_dims",
        # free edges (cached as JAX arrays)
        "_free_base",
        "_free_all",
        # plaquettes (JAX-safe padded arrays)
        "_plaquettes_py",
        "_plaquettes_idx",
        "_plaquettes_mask",
        "_n_plaquettes",
        "_plaquettes_Lmax",
    )

    def __init__(
        self,
        graph: AbstractGraph,
        cutoff: int,
        *,
        step: Union[int, float] = 1,
        gauge_dimensions: int = 1,
        constraint: Optional[List[List[List[Union[int, str]]]]] = None,
        auto_constraint: Optional[bool] = False,
        positive_qn: bool = False,
        qn_start: Optional[int] = None,
    ):
        """
        Construct a gauge-invariant (gauge-fixed) U(1) Hilbert-space core.

        This initialiser performs:

        1. Base-class initialisation to construct the allowed local basis and range metadata.
        2. Gauge-fixing resolution:
           - conflict checking between ``constraint`` and ``auto_constraint``,
           - optional auto-generation of a constraint array from the graph,
           - parsing into internal base-index constraints,
           - topological dependency analysis to identify free/slave structure.
        3. JAX-friendly caching of free-edge indices:
           - ``_free_base`` for indices in the base copy,
           - ``_free_all`` for flattened indices across all gauge copies.
        4. Lifting of base constraints across all gauge copies, and construction of a NetKet
           :class:`netket.hilbert.HomogeneousHilbert` with a :class:`U1Constraint`.
        5. Precomputation of plaquette (minimal-loop) data in a padded representation suitable for JIT.
        6. Dimension estimation and installation of legacy hooks expected by older call sites.

        :param graph: Graph defining edge DoFs and edge-index mappings.
        :param cutoff: Integer cutoff for local U(1) basis labels.
        :param step: Step size between allowed basis labels (must be integer-valued here).
        :param gauge_dimensions: Number of gauge copies stored in the flattened configuration.
        :param constraint: Gauge-fixing array in the user-facing format. Required unless
            ``auto_constraint=True``.
        :param auto_constraint: If ``True``, generate a gauge fixing from the graph.
        :param positive_qn: If ``True``, restrict the local labels to a non-negative (or shifted) range.
        :param qn_start: Optional starting label used when ``positive_qn=True``.
        :return: None.
        :raises InvalidCutoffError: If ``cutoff`` or ``step`` is not integer-valued for this constrained
            implementation.
        :raises AutoConstraintGaugeFixingConflictError: If both ``auto_constraint=True`` and ``constraint``
            is not ``None``.
        :raises IncorrectGaugeFixingArrayError: If parsing the gauge-fixing array fails or references
            unknown edges.
        :raises CyclicGaugeFixingError: If gauge-fixing dependencies contain a cycle.
        """

        if isinstance(cutoff, float) or isinstance(step, float):
            raise InvalidCutoffError(type(cutoff), type(step))

        super().__init__(
            graph=graph,
            cutoff=cutoff,
            step=step,
            gauge_dimensions=gauge_dimensions,
            positive_qn=positive_qn,
            qn_start=qn_start,
        )

        if auto_constraint and constraint is not None:
            raise AutoConstraintGaugeFixingConflictError()

        if auto_constraint:
            edges = graph.edges if graph.is_planar else graph.nonplanar_edges
            constraint = generate_constraint_array(edges)

            from neuralqx.utils.io.printing import NQXPrinter

            NQXPrinter().print(
                "Autogenerated the following gauge fixing:\n"
                + pretty_format_constraints(
                    constraint,
                    graph.nonplanar_vertex_mapping if not graph.is_planar else None,
                )
            )

        self._constraints = _parse_gauge_fixing_to_constraints(
            graph, cast(list, constraint)
        )
        self._gf = _topological_ordering(self._constraints)

        E = int(self.tiny_size)
        G = int(self.gauge_dimensions)

        # cache free edges as JAX arrays for fast and jit-friendly move proposals
        self._free_base = jnp.asarray(self._gf.free, dtype=jnp.int32)  # (F,)
        offsets = (jnp.arange(G, dtype=jnp.int32) * jnp.int32(E))[:, None]  # (G,1)
        self._free_all = (self._free_base[None, :] + offsets).reshape(-1)  # (G*F,)

        # lift constraints for all gauge copies
        full: list[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]] = []
        for gd in range(G):
            off = gd * E
            for c in self._constraints:
                inc = ((c.lhs + off, +1),)
                out = tuple((ridx + off, sgn) for (ridx, sgn) in c.rhs)
                full.append((inc, out))
        self._constraints_full = tuple(full)

        # NetKet Hilbert (flat)
        N = E * G
        self._hilbert = nk.hilbert.HomogeneousHilbert(
            local_states=self.allowed_basis_states,
            N=N,
            constraint=U1Constraint(
                self._constraints_full,
                self.cutoff,
                self.q_min,
                self.q_max,
                self.q_step,
            ),
        )

        # plaquettes
        (
            self._plaquettes_py,
            self._plaquettes_idx,
            self._plaquettes_mask,
            self._n_plaquettes,
            self._plaquettes_Lmax,
        ) = self._build_plaquettes_base_padded()

        # dimension estimate
        if not self.hilbert.is_indexable:
            self._dimensions = int(self.local_size) ** int(G * self._gf.n_free)
        else:
            self._dimensions = int(self.hilbert.n_states)

        self._print_dims = humanize.number.scientific(self._dimensions, precision=4)

        # legacy hooks for NK callers
        setattr(self._hilbert, "flip_gi_state_sgd", self.flip_state)
        setattr(self._hilbert, "flip_gi_state_agd", self.flip_state_all_gd)
        setattr(self._hilbert, "plaquette_flip", self.plaquette_flip)
        self._hilbert.random_state = self.random_state

        event(
            msg="INITIALIZING HILBERT CORE",
            tag="HILBERT:U1:CONSTRAINED:INIT",
            level=logging.INFO,
        )

    @property
    def hilbert(self) -> nk.hilbert.AbstractHilbert:
        """
        Return the underlying NetKet Hilbert space.

        The returned object is a flat :class:`netket.hilbert.HomogeneousHilbert` of size
        ``N = tiny_size * gauge_dimensions`` equipped with a :class:`U1Constraint` that enforces the lifted
        gauge-fixing relations.

        :return: NetKet Hilbert object backing this constrained core.
        :raises None: This property does not intentionally raise exceptions.
        """

        return self._hilbert

    @property
    def dimensions_pretty(self) -> str:
        """
        Return a human-friendly scientific-notation string for the Hilbert-space dimension.

        This is intended for logging and representations, since the exact dimension can be extremely large.

        :return: Dimension formatted in scientific notation.
        :raises None: This property does not intentionally raise exceptions.
        """

        return self._print_dims

    @property
    def gauge_fixing(self) -> GaugeFixingTopo:
        """
        Return the parsed gauge-fixing topology.

        The topology identifies free edges and provides a topological order for updating slave edges, along
        with the RHS fixer lists required to recompute each slave.

        :return: :class:`GaugeFixingTopo` describing free/slave structure and dependency order.
        :raises None: This property does not intentionally raise exceptions.
        """

        return self._gf

    @property
    def constraints_base(self) -> Tuple[_Constraint, ...]:
        """
        Return gauge-fixing constraints in base-edge indexing.

        Each returned constraint is a :class:`_Constraint` with:

        - ``lhs``: base index of the slave edge,
        - ``rhs``: tuple of ``(base_index, sign)`` dependency pairs.

        :return: Tuple of :class:`_Constraint` objects in base-index space.
        :raises None: This property does not intentionally raise exceptions.
        """

        return self._constraints

    def _view(self, sigma: jax.Array) -> jax.Array:
        """
        Reshape a batch of flattened states into a `(B, G, E)` view.

        This is an internal helper used to apply per-gauge-copy logic while keeping the external state
        representation flat.

        :param sigma: Batched flattened states of shape ``(B, N)`` with ``N = G * E``.
        :return: Reshaped view of shape ``(B, G, E)``.
        :raises ValueError: If ``sigma`` cannot be reshaped consistently into the expected ``(B, G, E)``
            dimensions.
        """

        B = sigma.shape[0]
        return sigma.reshape((B, int(self.gauge_dimensions), int(self.tiny_size)))

    def _flatten(self, v: jax.Array) -> jax.Array:
        """
        Flatten a `(B, G, E)` view back into `(B, N)`.

        This is the inverse of :meth:`_view` for batched arrays.

        :param v: Batched state view of shape ``(B, G, E)``.
        :return: Flattened states of shape ``(B, N)`` with ``N = G * E``.
        :raises ValueError: If ``v`` cannot be reshaped consistently into the expected flattened shape.
        """

        B = v.shape[0]
        return v.reshape((B, int(self.gauge_dimensions) * int(self.tiny_size)))

    def _mod_sum_last(self, vals: jax.Array) -> jax.Array:
        """
        Compute a modular sum along the last axis using :func:`neuralqx.utils.misc.arithmetic.mod_add`.

        This helper reduces an array ``vals[..., i]`` along the last axis using modular addition compatible
        with the configured local range and step. It is implemented using :func:`jax.lax.fori_loop` to be
        JIT-friendly.

        :param vals: Array of shape ``(..., K)`` to be summed along the last axis.
        :return: Array of shape ``(...)`` containing the modular sums.
        :raises None: This method does not intentionally raise exceptions.
        """

        K = int(vals.shape[-1])
        init = jnp.zeros(vals.shape[:-1], dtype=vals.dtype)

        def body(i, acc):
            return mod_add(
                acc, vals[..., i], q_min=self.q_min, q_max=self.q_max, step=self.q_step
            )

        return jax.lax.fori_loop(0, K, body, init)

    @errors_only(tag="REIMPOSING_GAUGE")
    def reimpose_gauge_fixing(self, sigma: jax.Array) -> jax.Array:
        """
        Deterministically reimpose gauge fixing by recomputing slave edges.

        This method projects an arbitrary state (or batch of states) onto the gauge-fixed subspace by
        overwriting *only* slave edges. Free edges are left unchanged.

        Algorithm:

        1. Reshape the input into ``(B, G, E)``.
        2. Iterate over slave edges in the stored topological order.
        3. For each slave edge, compute a signed modular sum of its fixer edges, and set the slave value to
           that result across all gauge copies.
        4. Flatten back to ``(B, N)`` and restore the input shape convention.

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :return: Gauge-consistent state(s) with the same shape as the input.
        :raises None: This method does not intentionally raise exceptions.
        """

        single = sigma.ndim == 1
        if single:
            sigma = sigma[None, :]

        v = self._view(sigma)

        for slave, fixers in zip(self._gf.slave_topo, self._gf.slave_fixers):
            dep_idx = jnp.asarray([d for (d, _s) in fixers], dtype=jnp.int32)
            signs = jnp.asarray([s for (_d, s) in fixers], dtype=v.dtype)

            contrib = v[:, :, dep_idx] * signs[None, None, :]
            val = self._mod_sum_last(contrib)
            v = v.at[:, :, int(slave)].set(val.astype(v.dtype))

        out = self._flatten(v)
        return out[0] if single else out

    @errors_only(tag="CHECK_STATES")
    def check_states(self, sigma: jax.Array) -> jax.Array:
        """
        Check whether state(s) satisfy all base gauge-fixing constraints across all gauge copies.

        For each base constraint and each gauge copy, this method verifies that the stored value on the LHS
        matches the modular signed sum of the RHS fixers.

        This is primarily intended for correctness/debugging. Constraint enforcement during sampling is
        handled by NetKet via the attached :class:`U1Constraint`, while constructive enforcement is provided
        by :meth:`reimpose_gauge_fixing`.

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :return: Boolean (for a single input state) or boolean array of shape ``(B,)`` indicating whether
            each state satisfies all constraints.
        :raises None: This method does not intentionally raise exceptions.
        """

        single = sigma.ndim == 1
        if single:
            sigma = sigma[None, :]

        v = self._view(sigma)
        ok = jnp.ones((sigma.shape[0],), dtype=bool)

        for c in self._constraints:
            lhs = int(c.lhs)
            dep_idx = jnp.asarray([d for (d, _s) in c.rhs], dtype=jnp.int32)
            signs = jnp.asarray([s for (_d, s) in c.rhs], dtype=v.dtype)

            rhs = self._mod_sum_last(v[:, :, dep_idx] * signs[None, None, :])
            match = jnp.all(v[:, :, lhs] == rhs, axis=1)
            ok = jnp.logical_and(ok, match)

        return ok[0] if single else ok

    def is_gauge_invariant(self, sigma: jax.Array) -> jax.Array:
        """
        Alias for :meth:`check_states`.

        This name is provided for API clarity: in this core, "gauge invariant" means "satisfies the stored
        gauge fixing relations".

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :return: Boolean (single state) or boolean array of shape ``(B,)`` indicating gauge invariance.
        :raises None: This method does not intentionally raise exceptions.
        """

        return self.check_states(sigma)

    def random_state(self, key: jax.Array, size: int = 1, dtype=None) -> jax.Array:
        """
        Generate random gauge-consistent states.

        Strategy:

        1. Sample values for all sites from the allowed local basis set.
        2. Apply :meth:`reimpose_gauge_fixing` to overwrite slave edges deterministically.

        The resulting states satisfy the gauge fixing exactly by construction.

        :param key: JAX PRNG key.
        :param size: Number of states to generate (batch size).
        :param dtype: Optional dtype override for the sampled states. If ``None``, the local basis dtype is
            used.
        :return: Batch of gauge-consistent states of shape ``(size, N)``.
        :raises None: This method does not intentionally raise exceptions.
        """

        qn = jnp.asarray(self.allowed_basis_states.all_states())
        if dtype is None:
            dtype = qn.dtype

        B = int(size)
        G = int(self.gauge_dimensions)
        E = int(self.tiny_size)

        k_idx = jax.random.randint(key, (B, G, E), 0, qn.shape[0])
        v = qn[k_idx].astype(dtype)
        sigma = v.reshape((B, G * E))
        return self.reimpose_gauge_fixing(sigma)

    def propose(self, sigma: jax.Array, key: jax.Array, move: Move) -> jax.Array:
        """
        Propose new state(s) by applying a typed move.

        This is the stable entry point for move proposals. Dispatch is performed on the concrete *type* of
        the provided :class:`~neuralqx.hilbert.operations.moves.Move` instance via
        :func:`functools.singledispatchmethod`.

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :param move: Move object describing the proposal.
        :return: Proposed state(s) with the same shape as the input.
        :raises NotImplementedError: If the move type is not supported by this core.
        :raises InvalidFreeEdgeSelectionError: If a free-edge flip requests more edges than available.
        :raises NotImplementedError: If a plaquette move is requested but the graph provides no plaquettes.
        """

        return self._propose(move, sigma, key)

    @singledispatchmethod
    def _propose(self, move: Move, sigma: jax.Array, key: jax.Array) -> jax.Array:
        """
        Internal dispatch target for move proposals.

        Concrete move implementations are registered on this method using
        :func:`functools.singledispatchmethod`. The first argument is the move instance so dispatch occurs on
        its concrete class.

        :param move: Concrete move instance (determines dispatch).
        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :return: Proposed state(s) with the same shape as the input.
        :raises NotImplementedError: If the move type has no registered implementation.
        """

        raise NotImplementedError(f"Unsupported move type: {type(move).__name__}")

    @_propose.register
    def _(
        self, move: FreeEdgeFlipSingleGauge, sigma: jax.Array, key: jax.Array
    ) -> jax.Array:
        """
        Flip free edges selected from the union of all gauge copies, then reimpose gauge fixing.

        Selection space
            Free sites are drawn from the flattened index set ``_free_all`` (shape ``(G*F,)``), which
            contains the free-edge indices for every gauge copy in the flat representation.

        Update rule
            - If ``move.adjacency`` is True, each selected site is updated by ``±q_step`` using modular
              addition.
            - Otherwise, each selected site is set to a uniformly random allowed basis value.

        After updating free edges, :meth:`reimpose_gauge_fixing` is applied to recompute all slave edges and
        restore constraint satisfaction.

        :param move: Move instance specifying how many free edges to flip and whether to use adjacency
            updates.
        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :return: Gauge-consistent proposed state(s) with the same shape as the input.
        :raises InvalidFreeEdgeSelectionError: If the requested number of edges exceeds the number of
            available free sites across all gauge copies.
        """

        n_edges = int(move.n_edges)
        if n_edges <= 0:
            return sigma

        single = sigma.ndim == 1
        if single:
            sigma = sigma[None, :]

        B = int(sigma.shape[0])

        if n_edges > int(self._free_all.shape[0]):
            raise InvalidFreeEdgeSelectionError(n_edges, int(self._free_all.shape[0]))

        qn = jnp.asarray(self.allowed_basis_states.all_states())
        step = jnp.asarray(self.q_step, dtype=sigma.dtype)

        def _flip_one(s: jax.Array, k: jax.Array) -> jax.Array:
            k_idx, k_val = jax.random.split(k, 2)
            idxs = jax.random.choice(
                k_idx, self._free_all, shape=(n_edges,), replace=False
            )

            if move.adjacency:
                deltas = jax.random.choice(
                    k_val,
                    jnp.array([-step, step], dtype=sigma.dtype),
                    shape=(n_edges,),
                    replace=True,
                )
                new_vals = mod_add(
                    s[idxs],
                    deltas,
                    q_min=self.q_min,
                    q_max=self.q_max,
                    step=self.q_step,
                ).astype(s.dtype)
            else:
                ridx = jax.random.randint(k_val, (n_edges,), 0, qn.shape[0])
                new_vals = qn[ridx].astype(s.dtype)

            return s.at[idxs].set(new_vals)

        keys = jax.random.split(key, B)
        flipped = jax.vmap(_flip_one)(sigma, keys)
        out = self.reimpose_gauge_fixing(flipped)
        return out[0] if single else out

    @_propose.register
    def _(
        self, move: FreeEdgeFlipAllGauge, sigma: jax.Array, key: jax.Array
    ) -> jax.Array:
        """
        Flip free edges inside each gauge copy, then reimpose gauge fixing.

        Selection space
            Free edges are drawn from ``_free_base`` (shape ``(F,)``), which contains free edge indices in
            the base copy. The input is reshaped to ``(B, G, E)``, and the same selection/update rule is
            applied independently to each `(state, gauge_copy)` slice.

        Update rule per slice
            - If ``move.adjacency`` is True, apply ``±q_step`` using modular addition.
            - Otherwise, assign uniformly random allowed basis values.

        After flipping in all copies, :meth:`reimpose_gauge_fixing` recomputes slave edges and restores
        constraint satisfaction.

        :param move: Move instance specifying how many free edges to flip and whether to use adjacency
            updates.
        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :return: Gauge-consistent proposed state(s) with the same shape as the input.
        :raises InvalidFreeEdgeSelectionError: If the requested number of edges exceeds the number of base
            free edges available in a single gauge copy.
        """

        n_edges = int(move.n_edges)
        if n_edges <= 0:
            return sigma

        single = sigma.ndim == 1
        if single:
            sigma = sigma[None, :]

        B = int(sigma.shape[0])
        G = int(self.gauge_dimensions)

        if n_edges > int(self._free_base.shape[0]):
            raise InvalidFreeEdgeSelectionError(n_edges, int(self._free_base.shape[0]))

        qn = jnp.asarray(self.allowed_basis_states.all_states())
        step = jnp.asarray(self.q_step, dtype=sigma.dtype)

        v = self._view(sigma)

        def _flip_one_state(s: jax.Array, k: jax.Array) -> jax.Array:
            k_idx, k_val = jax.random.split(k, 2)
            idxs = jax.random.choice(
                k_idx, self._free_base, shape=(n_edges,), replace=False
            )

            if move.adjacency:
                deltas = jax.random.choice(
                    k_val,
                    jnp.array([-step, step], dtype=s.dtype),
                    shape=(n_edges,),
                    replace=True,
                )
                new_vals = mod_add(
                    s[idxs],
                    deltas,
                    q_min=self.q_min,
                    q_max=self.q_max,
                    step=self.q_step,
                ).astype(s.dtype)
            else:
                ridx = jax.random.randint(k_val, (n_edges,), 0, qn.shape[0])
                new_vals = qn[ridx].astype(s.dtype)

            return s.at[idxs].set(new_vals)

        # make a key per (B,G) slice

        # (-1) = key width (usually 2)
        keys = jax.random.split(key, B * G).reshape((B * G, -1))
        vflat = v.reshape((B * G, v.shape[-1]))
        vflat2 = jax.vmap(_flip_one_state)(vflat, keys)
        v2 = vflat2.reshape(v.shape)

        out = self.reimpose_gauge_fixing(self._flatten(v2))
        return out[0] if single else out

    def _build_plaquettes_base_padded(
        self,
    ) -> Tuple[Tuple[Tuple[int, ...], ...], jax.Array, jax.Array, int, int]:
        """
        Precompute plaquette (minimal-loop) data in both Python and JAX-safe padded forms.

        This routine extracts minimal loops from the graph (if available) and converts each loop into a
        tuple of base edge indices. It then builds a padded representation suitable for JIT-safe selection
        and updates:

        - ``loops_py``: tuple-of-tuples of base indices (introspection/debug).
        - ``idx``: int32 array of shape ``(P, Lmax)`` with edge indices padded by ``-1``.
        - ``msk``: boolean array of shape ``(P, Lmax)`` marking valid entries in ``idx``.
        - ``P``: number of plaquettes.
        - ``Lmax``: maximum plaquette length.

        :param None: This method takes no arguments beyond ``self``.
        :return: Tuple ``(loops_py, idx, msk, P, Lmax)`` as described above.
        :raises IncorrectGaugeFixingArrayError: If a loop edge token cannot be resolved to a base index.
        """

        if not hasattr(self.graph, "minimal_loops"):
            empty_idx = jnp.zeros((0, 0), dtype=jnp.int32)
            empty_msk = jnp.zeros((0, 0), dtype=jnp.bool_)
            return tuple(), empty_idx, empty_msk, 0, 0

        loops_py: list[Tuple[int, ...]] = []
        for loop in self.graph.minimal_loops():
            idxs = tuple(_edge_to_index_safe(self.graph, e) for e in loop)
            if len(idxs) > 0:
                loops_py.append(idxs)

        if not loops_py:
            empty_idx = jnp.zeros((0, 0), dtype=jnp.int32)
            empty_msk = jnp.zeros((0, 0), dtype=jnp.bool_)
            return tuple(), empty_idx, empty_msk, 0, 0

        P = len(loops_py)
        Lmax = max(len(l) for l in loops_py)

        idx = -jnp.ones((P, Lmax), dtype=jnp.int32)
        msk = jnp.zeros((P, Lmax), dtype=jnp.bool_)

        for i, loop in enumerate(loops_py):
            ll = len(loop)
            idx = idx.at[i, :ll].set(jnp.asarray(loop, dtype=jnp.int32))
            msk = msk.at[i, :ll].set(True)

        return tuple(loops_py), idx, msk, P, Lmax

    def _apply_random_plaquette_on_state(
        self, s: jax.Array, key: jax.Array
    ) -> jax.Array:
        """
        Apply a random plaquette update to a single gauge-copy state vector.

        A plaquette update consists of:

        1. Selecting one plaquette uniformly at random.
        2. Selecting an increment ``inc`` uniformly from ``{+q_step, -q_step}``.
        3. Applying modular addition of ``inc`` to every edge in the selected plaquette.

        The implementation uses only JAX-compatible operations (dynamic indexing, masks, and a `fori_loop`)
        so it can be JIT compiled.

        :param s: State vector for a single gauge copy, shape ``(E,)``.
        :param key: JAX PRNG key.
        :return: Updated state vector of shape ``(E,)``. If no plaquettes exist, returns ``s`` unchanged.
        :raises None: This method does not intentionally raise exceptions.
        """

        if self._n_plaquettes <= 0:
            return s

        k1, k2 = jax.random.split(key, 2)
        loop_id = jax.random.randint(k1, (), 0, self._n_plaquettes, dtype=jnp.int32)

        loop = jax.lax.dynamic_index_in_dim(
            self._plaquettes_idx, loop_id, axis=0, keepdims=False
        )
        mask = jax.lax.dynamic_index_in_dim(
            self._plaquettes_mask, loop_id, axis=0, keepdims=False
        )

        step = jnp.asarray(self.q_step, dtype=s.dtype)
        inc = jax.random.choice(k2, jnp.array([-step, step], dtype=s.dtype), shape=())

        def body(i, ss):
            def do_update(x):
                e = loop[i]
                newv = mod_add(
                    x[e], inc, q_min=self.q_min, q_max=self.q_max, step=self.q_step
                ).astype(x.dtype)
                return x.at[e].set(newv)

            return jax.lax.cond(mask[i], do_update, lambda x: x, ss)

        return jax.lax.fori_loop(0, self._plaquettes_Lmax, body, s)

    def plaquette_flip(self, sigma: jax.Array, key: jax.Array) -> jax.Array:
        """
        Legacy convenience wrapper for plaquette moves.

        This method exists for backward compatibility with older APIs that exposed a direct
        ``plaquette_flip(sigma, key)`` entry point. Internally it routes to :meth:`propose` using a typed
        plaquette move.

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :return: Proposed gauge-consistent state(s) after a plaquette update.
        :raises NotImplementedError: If no plaquettes are available for the current graph.
        """

        return self.propose(sigma, key, PlaquetteFlipAllGauge())

    @_propose.register
    def _(
        self, move: PlaquetteFlipSingleGauge, sigma: jax.Array, key: jax.Array
    ) -> jax.Array:
        """
        Apply a plaquette update in exactly one randomly chosen gauge copy, then reimpose gauge fixing.

        For each state in the batch:

        1. Choose a gauge copy index uniformly at random.
        2. Apply one random plaquette update to that copy only.
        3. Recompute slave edges via :meth:`reimpose_gauge_fixing`.

        :param move: Plaquette move instance selecting the single-gauge behaviour.
        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :return: Gauge-consistent proposed state(s) with the same shape as the input.
        :raises NotImplementedError: If the graph exposes no plaquettes (no minimal loops available).
        """

        if self._n_plaquettes <= 0:
            raise NotImplementedError(
                "No plaquettes available: graph.minimal_loops() is empty/missing."
            )

        single = sigma.ndim == 1
        if single:
            sigma = sigma[None, :]

        B = int(sigma.shape[0])
        G = int(self.gauge_dimensions)
        E = int(self.tiny_size)

        keys = jax.random.split(key, B)

        def _one(s: jax.Array, k: jax.Array) -> jax.Array:
            k1, k2 = jax.random.split(k, 2)
            gd = jax.random.randint(k1, (), 0, G, dtype=jnp.int32)

            v = s.reshape((G, E))
            vg = jax.lax.dynamic_index_in_dim(v, gd, axis=0, keepdims=False)
            vg2 = self._apply_random_plaquette_on_state(vg, k2)
            v2 = v.at[gd].set(vg2)
            return v2.reshape((G * E,))

        flipped = jax.vmap(_one)(sigma, keys)
        out = self.reimpose_gauge_fixing(flipped)
        return out[0] if single else out

    @_propose.register
    def _(
        self, move: PlaquetteFlipAllGauge, sigma: jax.Array, key: jax.Array
    ) -> jax.Array:
        """
        Apply a plaquette update independently in every gauge copy, then reimpose gauge fixing.

        For each `(state, gauge_copy)` slice:

        1. Apply one random plaquette update with an independent PRNG subkey.
        2. After all copies are updated, recompute slave edges via :meth:`reimpose_gauge_fixing`.

        :param move: Plaquette move instance selecting the all-gauge behaviour.
        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :return: Gauge-consistent proposed state(s) with the same shape as the input.
        :raises NotImplementedError: If the graph exposes no plaquettes (no minimal loops available).
        """

        if self._n_plaquettes <= 0:
            raise NotImplementedError(
                "No plaquettes available: graph.minimal_loops() is empty/missing."
            )

        single = sigma.ndim == 1
        if single:
            sigma = sigma[None, :]

        B = int(sigma.shape[0])
        G = int(self.gauge_dimensions)

        v = self._view(sigma)

        # key per (B,G)
        keys = jax.random.split(key, B * G).reshape((B * G, -1))
        vflat = v.reshape((B * G, v.shape[-1]))

        vflat2 = jax.vmap(self._apply_random_plaquette_on_state)(vflat, keys)
        v2 = vflat2.reshape(v.shape)

        out = self.reimpose_gauge_fixing(self._flatten(v2))
        return out[0] if single else out

    def flip_state(
        self,
        sigma: jax.Array,
        key: jax.Array,
        number_of_edges: int = 1,
        *,
        adjacency: bool = False,
        scope: str = "single",
    ) -> jax.Array:
        """
        Backwards-compatible flip API using string-based scope.

        This wrapper translates older ``scope`` strings into typed move objects and then delegates to
        :meth:`propose`.

        Supported mappings:

        - Single-gauge selection space (union of all gauge copies):
          ``scope`` in ``{"single", "sgd", "one", "union"}`` → ``FreeEdgeFlipSingleGauge``.
        - Per-gauge-copy updates:
          ``scope`` in ``{"all", "agd", "all_gd", "per_gauge", "each"}`` → ``FreeEdgeFlipAllGauge``.

        Unrecognised scope strings default to the single-gauge behaviour.

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :param number_of_edges: Number of free edges to flip per proposal (interpreted by the chosen move).
        :param adjacency: If True, use ±step modular updates; otherwise assign random allowed values.
        :param scope: Scope selector determining single-vs-all gauge-copy behaviour.
        :return: Gauge-consistent proposed state(s) with the same shape as the input.
        :raises InvalidFreeEdgeSelectionError: If the requested number of edges exceeds available free
            edges for the selected scope.
        """

        s = str(scope).lower().strip()

        if s in ("single", "sgd", "one", "union"):
            mv = FreeEdgeFlipSingleGauge(
                n_edges=int(number_of_edges), adjacency=bool(adjacency)
            )
            return self.propose(sigma, key, mv)

        if s in ("all", "agd", "all_gd", "per_gauge", "each"):
            mv = FreeEdgeFlipAllGauge(
                n_edges=int(number_of_edges), adjacency=bool(adjacency)
            )
            return self.propose(sigma, key, mv)

        mv = FreeEdgeFlipSingleGauge(
            n_edges=int(number_of_edges), adjacency=bool(adjacency)
        )
        return self.propose(sigma, key, mv)

    def flip_state_all_gd(
        self,
        sigma: jax.Array,
        key: jax.Array,
        number_of_edges: int = 1,
        *,
        adjacency: bool = False,
        scope: str = "all",
    ) -> jax.Array:
        """
        Backwards-compatible convenience wrapper for flipping in all gauge copies.

        This wrapper always constructs a ``FreeEdgeFlipAllGauge`` move (independent flips per gauge copy)
        and delegates to :meth:`propose`. The ``scope`` argument is accepted for compatibility but does not
        change behaviour.

        :param sigma: State of shape ``(N,)`` or batch of shape ``(B, N)``.
        :param key: JAX PRNG key.
        :param number_of_edges: Number of free edges to flip inside each gauge copy.
        :param adjacency: If True, use ±step modular updates; otherwise assign random allowed values.
        :param scope: Accepted for API compatibility; ignored.
        :return: Gauge-consistent proposed state(s) with the same shape as the input.
        :raises InvalidFreeEdgeSelectionError: If the requested number of edges exceeds available free
            edges in a single gauge copy.
        """

        mv = FreeEdgeFlipAllGauge(
            n_edges=int(number_of_edges), adjacency=bool(adjacency)
        )
        return self.propose(sigma, key, mv)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"dimensions={self._print_dims}, "
            f"cutoff={self.cutoff}, "
            f"dofs={self.allowed_basis_states}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"n_free={self._gf.n_free}, "
            f"indexable={self.hilbert.is_indexable}"
            f")"
        )
