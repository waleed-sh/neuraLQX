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


"""Gauge-invariant homogeneous U(1)^N Hilbert space.

This module defines the constructive gauge-fixed U(1) Hilbert space. Gauge
relations identify slave edges whose values are recomputed from free edges,
allowing random sampling and proposal updates to stay on the gauge-fixing slice.
"""

from __future__ import annotations

from typing import Any

import jax

from neuralqx.hilbert.u1.constraints import GaugeFixingSpec
from neuralqx.hilbert.u1.constraints import GaugeFixingTopology
from neuralqx.hilbert.u1.constraints import GaugeRelation
from neuralqx.hilbert.u1.constraints import build_topology
from neuralqx.hilbert.u1.constraints import pretty_gauge_fixing
from neuralqx.hilbert.u1.random import reimpose_gauge_fixing_jit
from neuralqx.hilbert.u1.utils.gauge_setup import constrained_dimension
from neuralqx.hilbert.u1.utils.gauge_setup import constraint_from_topology
from neuralqx.hilbert.u1.utils.gauge_setup import relations_from_inputs
from neuralqx.hilbert.u1.utils.gauge_setup import validate_integer_constraint_range
from neuralqx.hilbert.u1.utils import freeze_gauge_fixing
from neuralqx.hilbert.u1.utils import scientific_int
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.hilbert.utils import restore_trailing
from neuralqx.utils.struct import field

from .unconstrained import U1Hilbert


class U1GaugeInvariantHilbert(U1Hilbert):
    """Constructive gauge-fixed homogeneous U(1)^N edge Hilbert space.

    ``U1GaugeInvariantHilbert`` represents the physical subspace obtained by
    reconstructing selected slave edges from independently sampled free edges.
    Each relation expresses one base edge as a signed modular sum of other base
    edges, and the same relation is applied independently to every U(1) gauge
    component. The resulting flat state layout is identical to
    :class:`U1Hilbert`, but not every locally valid charge assignment is
    accepted.

    Gauge fixing can be supplied explicitly through ``gauge_fixing`` or
    generated from the graph when ``auto_constraint`` is true. During
    construction the relations are frozen, topologically ordered, converted
    into a discrete constraint, and used to compute the exact constrained
    dimension. Sampling draws free-edge values and then reconstructs slave
    values, while proposal moves are projected back onto the gauge-fixing slice
    after updates.

    This class is the public entry point for U(1) gauge-invariant simulations.
    It preserves the inherited U(1) local range, graph views, index conversion,
    free-edge proposal API, and plaquette proposal API, but the valid state set
    is defined by the resolved constructive gauge relations.

    Args:
        gauge_fixing: Optional user-facing gauge-fixing specification.
        auto_constraint: Whether to generate spanning-tree gauge fixing
            automatically from the graph.

    Attributes:
        gauge_fixing: Frozen user-facing gauge-fixing specification.
        auto_constraint: Whether spanning-tree gauge fixing is generated.
        relations: Resolved constructive gauge relations.
        topology: Dependency topology for free and slave edges.
    """

    gauge_fixing: GaugeFixingSpec | None = field(static=True, default=None)
    """Frozen user-facing gauge-fixing specification."""

    auto_constraint: bool = field(static=True, default=False)
    """Whether spanning-tree gauge fixing is generated."""

    relations: tuple[GaugeRelation, ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
    )
    """Resolved constructive gauge relations."""

    topology: GaugeFixingTopology = field(static=True, init=False, default=None)
    """Dependency topology for free and slave edges."""

    def __post_init__(self) -> None:
        """Resolves gauge fixing and rebuilds constrained metadata."""
        validate_integer_constraint_range(self.cutoff, self.step)
        object.__setattr__(self, "gauge_fixing", freeze_gauge_fixing(self.gauge_fixing))
        super().__post_init__()
        relations = relations_from_inputs(self)
        topology = build_topology(relations, n_edges=self.tiny_size)
        constraint = constraint_from_topology(self, relations, topology)
        dimension = constrained_dimension(self, topology)

        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(self, "_dimension_pretty", scientific_int(dimension))
        constraint.validate_hilbert(self)

    def __hash__(self) -> int:
        """Returns a structural hash for the gauge-invariant space."""
        return hash(
            (
                type(self),
                self.graph,
                self.cutoff,
                self.step,
                self.gauge_dimensions,
                self.positive_qn,
                self.qn_start,
                self.relations,
            )
        )

    @property
    def constraints_base(self) -> tuple[GaugeRelation, ...]:
        """Resolved constructive gauge relations."""
        return self.relations

    @property
    def gauge_fixing_topology(self) -> GaugeFixingTopology:
        """Dependency topology for free and slave edges."""
        return self.topology

    @property
    def _free_base_edges(self) -> tuple[int, ...]:
        """Base graph edges that remain independent after gauge fixing."""
        return self.topology.free

    def reimpose_gauge_fixing(self, states: Any) -> jax.Array:
        """Recomputes slave edges from free edges.

        Args:
            states: State array whose trailing dimension equals :attr:`size`.

        Returns:
            State array with gauge-fixing relations restored. Free edge values
            are preserved, and slave edge values are overwritten according to
            the dependency order stored in :attr:`topology`.
        """
        arr, single, batch = ensure_2d_trailing(states, self.size)
        out = reimpose_gauge_fixing_jit(self, arr)
        return restore_trailing(out, single, batch)

    def check_states(self, states: Any) -> jax.Array:
        """Checks whether U(1) states satisfy the resolved gauge fixing.

        Args:
            states: State array whose trailing dimension equals :attr:`size`.

        Returns:
            Boolean mask over the leading batch axes. A value is true only when
            all constructive relations hold in every gauge component.
        """
        return self.constraint(states)

    def is_gauge_invariant(self, states: Any) -> jax.Array:
        """Physics-language alias for :meth:`check_states`.

        Args:
            states: Candidate U(1) state array in the flat edge layout.

        Returns:
            Boolean mask indicating which states lie on the constructive
            gauge-fixing slice represented by this Hilbert space.
        """
        return self.check_states(states)

    def random_state(
        self,
        key: jax.Array,
        size: int | tuple[int, ...] | None = None,
        *,
        dtype: Any | None = None,
        max_trials: int = 1024,
    ) -> jax.Array:
        """Samples random gauge-fixed U(1) states.

        Args:
            key: JAX pseudo-random key.
            size: Optional leading batch shape.
            dtype: Optional dtype for state values.
            max_trials: Ignored because constructive projection is used.

        Returns:
            Random states with gauge fixing imposed.
        """
        del max_trials
        return self.reimpose_gauge_fixing(
            super().random_state(key, size=size, dtype=dtype)
        )

    def _project_after_update(self, states: jax.Array) -> jax.Array:
        """Projects proposal updates back onto the gauge-fixing slice."""
        return self.reimpose_gauge_fixing(states)

    def gauge_fixing_pretty(self) -> str:
        """Formats the resolved gauge-fixing relations for display.

        Returns:
            Multi-line text describing each slave edge and the signed source
            edges used to reconstruct it, using graph edge labels when the
            graph can provide them.
        """
        return pretty_gauge_fixing(self.graph, self.relations)

    def __repr__(self) -> str:
        """Returns a compact representation of constrained U(1) metadata.

        Returns:
            String containing the constrained dimension summary, local charge
            range parameters, graph edge count, gauge-component count, number
            of free base edges, number of gauge relations, and indexability
            status.
        """
        return (
            f"U1GaugeInvariantHilbert("
            f"dimensions={self.dimensions_pretty}, "
            f"cutoff={self.cutoff}, "
            f"step={self.step}, "
            f"local_size={self.local_size}, "
            f"n_edges={self.tiny_size}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"n_free={self.topology.n_free}, "
            f"n_relations={len(self.relations)}, "
            f"indexable={self.is_indexable})"
        )


ConstrainedU1Hilbert = U1GaugeInvariantHilbert
