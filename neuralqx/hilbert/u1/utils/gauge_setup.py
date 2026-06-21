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


"""Construction helpers for constrained U(1) spaces.

The functions in this module turn user-provided or automatically generated
gauge-fixing declarations into constraint objects, dependency topology, and an
exact constructive dimension.
"""

from __future__ import annotations

from neuralqx.hilbert.u1.constraints import GaugeFixingTopology
from neuralqx.hilbert.u1.constraints import GaugeRelation
from neuralqx.hilbert.u1.constraints import U1GaugeConstraint
from neuralqx.hilbert.u1.constraints import auto_gauge_fixing
from neuralqx.hilbert.u1.constraints import parse_gauge_fixing


def validate_integer_constraint_range(cutoff: int | float, step: int | float) -> None:
    """Validates that constrained U(1) arithmetic uses exact integer ranges.

    Args:
        cutoff: Charge cutoff supplied to the U(1) Hilbert space.
        step: Charge spacing supplied to the U(1) Hilbert space.

    Raises:
        TypeError: If either value is floating point.
    """
    if isinstance(cutoff, float) or isinstance(step, float):
        raise TypeError(
            "Constrained U(1) Hilbert spaces require integer cutoff and step."
        )


def relations_from_inputs(space: object) -> tuple[GaugeRelation, ...]:
    """Resolves explicit or automatic gauge-fixing relations.

    Args:
        space: U(1) gauge-invariant Hilbert space under construction.

    Returns:
        Tuple of constructive gauge relations.

    Raises:
        ValueError: If explicit and automatic gauge fixing are requested
            together.
    """
    if space.auto_constraint and space.gauge_fixing is not None:
        raise ValueError("Pass either gauge_fixing or auto_constraint=True, not both.")
    if space.auto_constraint:
        return auto_gauge_fixing(space.graph)
    return parse_gauge_fixing(space.graph, space.gauge_fixing)


def constraint_from_topology(
    space: object,
    relations: tuple[GaugeRelation, ...],
    topology: GaugeFixingTopology,
) -> U1GaugeConstraint:
    """Builds the discrete constraint object from resolved topology.

    Args:
        space: U(1) gauge-invariant Hilbert space under construction.
        relations: Resolved constructive gauge relations.
        topology: Dependency topology derived from ``relations``.

    Returns:
        Constraint object that checks all constructive relations.
    """
    return U1GaugeConstraint(
        relations=relations,
        gauge_dimensions=space.gauge_dimensions,
        n_edges=space.tiny_size,
        q_min=space.q_min,
        step=space.q_step,
        local_size=space.local_size,
        n_free=topology.n_free,
    )


def constrained_dimension(space: object, topology: GaugeFixingTopology) -> int:
    """Computes the exact dimension of the constructive gauge-fixed subspace.

    Args:
        space: U(1) gauge-invariant Hilbert space under construction.
        topology: Dependency topology for free and slave edges.

    Returns:
        Number of independent free-edge states across all gauge dimensions.
    """
    return int(space.local_size) ** int(space.gauge_dimensions * topology.n_free)
