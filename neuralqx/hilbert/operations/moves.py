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

from __future__ import annotations

from dataclasses import dataclass


class Move:
    """
    Marker base class for Hilbert-space moves.

    A move is a lightweight, immutable description of a proposal to modify a state configuration
    during Monte Carlo sampling (or any other exploration of the Hilbert space). In neuraLQX, moves
    are intentionally separated from the Hilbert core logic:

    - The move object encodes what kind of modification is requested (e.g. flip a free edge, apply
      a plaquette update), along with any hyperparameters (e.g. number of edges, adjacency policy).
    - The Hilbert core (via `propose(...)` or `flip_state(...)`) encodes how to apply that
      modification while respecting structure and constraints (e.g. gauge fixing, topological
      reimposition, modular arithmetic).

    This separation lets you:
    - pass moves around as plain data (easy to store in configs, serialise, log),
    - implement multiple proposal families cleanly,
    - extend the system by adding new `Move` subclasses and registering handlers.

    Moves should be side-effect free and contain no references to mutable state. They serve as
    dispatch keys for multi-dispatch (e.g. `plum.dispatch` or `singledispatch`) in proposal-
    generation code.

    :note: This class is a marker/namespace type, it does not define behavior on its own.
    """


@dataclass(frozen=True, slots=True)
class FreeEdgeFlipSingleGauge(Move):
    """
    Flip `n_edges` free edges across the union of all gauge copies.

    This move requests a proposal that:
    - selects `n_edges` free degrees of freedom from the union of free-edge indices across all
      gauge copies (all gauge dimensions are pooled together),
    - modifies those selected degrees of freedom,
    - and (for constrained spaces) reimposes gauge fixing/constraints afterwards so the resulting
      configuration remains gauge-invariant.

    Typical semantics (as expected by the constrained U(1) core):
    - "Union" means the candidate indices are `free_edges + offset(gauge_copy)` for every gauge
      copy, and the move chooses among them globally.
    - This move is useful when you want a single proposal to touch a small number of independent
      DoFs without committing to flipping one per gauge copy.

    Adjacency policy:
    - `adjacency=False` (default): assign a fresh allowed quantum number (uniformly drawn from the
      allowed local basis set).
    - `adjacency=True`: perform a local ±step update (modularly wrapped if the space defines
      `(q_min, q_max, q_step)`).

    :param n_edges: Number of free edges to flip in total across all gauge copies. Must be
        compatible with the number of available free indices in the pooled set.
    :param adjacency: If `True`, propose ±step changes; if `False`, propose random allowed values.
    """

    n_edges: int = 1
    """The number of edges to be flipped in the proposal move."""

    adjacency: bool = False
    """If True, +/- step, otherwise the flip assigns any random allowed DoF."""


@dataclass(frozen=True, slots=True)
class FreeEdgeFlipAllGauge(Move):
    """
    Flip `n_edges` free edges independently inside each gauge copy.

    This move requests a proposal that:
    - treats each gauge copy (gauge dimension slice) independently,
    - selects `n_edges` free indices within that slice,
    - flips those indices within that slice,
    - and reimposes gauge fixing / constraints afterwards (for constrained spaces).

    Typical semantics (as expected by the constrained U(1) core):
    - If there are `G` gauge copies, this move flips up to `G * n_edges` degrees of freedom per
      configuration (because each copy performs its own selection).
    - This move is useful when you want each gauge level to explore independently and symmetrically,
      e.g. when gauge copies represent independent components of a product Hilbert space.

    Adjacency policy:
    - `adjacency=False` (default): assign fresh allowed quantum numbers.
    - `adjacency=True`: perform ±step updates with modular wrapping if supported.

    :param n_edges: Number of free edges to flip per gauge copy. Must satisfy
        `n_edges <= (# free edges in one copy)`.
    :param adjacency: If `True`, propose ±step changes, if `False`, propose random allowed values.
    """

    n_edges: int = 1
    """The number of edges to be flipped in the proposal move."""

    adjacency: bool = False
    """If True, +/- step, otherwise the flip assigns any random allowed DoF."""


@dataclass(frozen=True, slots=True)
class PlaquetteFlipSingleGauge(Move):
    """
    Apply +/- step along one randomly chosen plaquette in one gauge copy.

    A plaquette is a minimal loop (cycle) in the underlying graph (obtained from
    `graph.minimal_loops()` in the core). This move requests a proposal that:
    - selects one gauge copy (one gauge dimension slice),
    - selects one plaquette (minimal loop) within that copy,
    - adds either `+step` or `-step` to all edges along that loop (typically using modular addition
      so values remain within the allowed range),
    - and then reimposes gauge fixing/constraints (for constrained spaces) to restore gauge
      invariance.

    This type of update can improve ergodicity in lattice/gauge-like models by moving along loop
    degrees of freedom rather than single-edge changes.

    :note: This move carries no parameters yet. If needed later, it can be extended with
        fields such as a loop-selection policy, a distribution over plaquettes, or a
        custom step size.
    """

    pass


@dataclass(frozen=True, slots=True)
class PlaquetteFlipAllGauge(Move):
    """
    Apply +/- step along one randomly chosen plaquette per gauge copy.

    This move is the multi-copy analogue of `PlaquetteFlipSingleGauge`. It requests a
    proposal that:
    - for each gauge copy independently:
      - selects one plaquette (minimal loop),
      - applies either `+step` or `-step` uniformly along that loop,
    - then reimposes gauge fixing / constraints (for constrained spaces).

    Compared to the single-gauge version, this can introduce larger composite updates (per proposal)
    and may mix faster when gauge copies are intended to evolve in parallel.

    :note: This move carries no parameters yet. Future extensions could add correlated
        loop choices across copies or different per-copy selection policies.
    """

    pass
