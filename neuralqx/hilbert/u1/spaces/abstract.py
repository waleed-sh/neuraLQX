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


"""Shared construction for homogeneous U(1)^N Hilbert spaces.

The abstract U(1) implementation builds the common edge-only graph layout,
local charge range, plaquette metadata, and product dimension used by both
unconstrained and gauge-invariant U(1) Hilbert spaces.
"""

from __future__ import annotations

import abc
from typing import Any

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.constraint import IdentityConstraint
from neuralqx.hilbert.layout import GraphHilbertLayout
from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.local import LocalRange
from neuralqx.hilbert.space import DiscreteHilbertSpace
from neuralqx.hilbert.u1.properties import U1PropertiesMixin
from neuralqx.hilbert.u1.proposals.mixin import U1ProposalMixin
from neuralqx.hilbert.u1.sampling import U1SamplingMixin
from neuralqx.hilbert.u1.state import U1StateMixin
from neuralqx.hilbert.u1.utils.space_builders import edge_space_for_gauge_dimension
from neuralqx.hilbert.u1.utils.space_builders import plaquette_metadata
from neuralqx.hilbert.u1.utils import scientific_int
from neuralqx.hilbert.u1.utils import u1_local_range
from neuralqx.hilbert.u1.utils import validate_gauge_dimensions
from neuralqx.utils.struct import field


class AbstractU1Hilbert(
    U1PropertiesMixin,
    U1StateMixin,
    U1SamplingMixin,
    U1ProposalMixin,
    DiscreteHilbertSpace,
):
    """Base implementation for homogeneous U(1)^N edge Hilbert spaces.

    Args:
        graph: Graph whose edges carry U(1) degrees of freedom.
        cutoff: Charge cutoff used to construct the local range.
        step: Charge spacing between adjacent local values.
        gauge_dimensions: Number of independent U(1) components per edge.
        positive_qn: Whether to use a non-negative charge range.
        qn_start: Optional explicit start value for the charge range.

    Attributes:
        graph: Graph whose edges carry U(1) degrees of freedom.
        cutoff: Charge cutoff used to construct the local range.
        step: Charge spacing between adjacent local values.
        gauge_dimensions: Number of independent U(1) components per edge.
        positive_qn: Whether to use a non-negative charge range.
        qn_start: Optional explicit start value for the charge range.
        constraint: Constraint applied to U(1) states.
        local_space: Scalar local charge range.
        layout: Graph layout mapping edge components to flat sites.
        _local_spaces: Scalar local charge range for every flat site.
        _dimension: Exact Hilbert-space dimension.
        _dimension_pretty: Scientific-notation dimension string.
        _plaquettes: Minimal graph loops represented by edge indices.
        _plaquette_lmax: Maximum plaquette length.
        _plaquette_idx: Padded plaquette edge-index array.
        _plaquette_mask: Boolean mask for valid padded plaquette entries.
    """

    graph: AbstractGraph = field(pytree=False)
    """Graph whose edges carry U(1) degrees of freedom."""

    cutoff: int | float = field(static=True)
    """Charge cutoff used to construct the local range."""

    step: int | float = field(static=True, default=1)
    """Charge spacing between adjacent local values."""

    gauge_dimensions: int = field(static=True, default=1)
    """Number of independent U(1) components per edge."""

    positive_qn: bool = field(static=True, default=False)
    """Whether to use a non-negative charge range."""

    qn_start: int | float | None = field(static=True, default=None)
    """Optional explicit start value for the charge range."""

    constraint: AbstractDiscreteConstraint = field(
        static=True,
        init=False,
        default_factory=IdentityConstraint,
    )
    """Constraint applied to U(1) states."""

    local_space: LocalRange = field(static=True, init=False, default=None)
    """Scalar local charge range."""

    layout: GraphHilbertLayout = field(pytree=False, init=False, default=None)
    """Graph layout mapping edge components to flat sites."""

    _local_spaces: tuple[AbstractLocalSpace, ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    """Scalar local charge range for every flat site."""

    _dimension: int = field(static=True, init=False, default=0, repr=False)
    """Exact Hilbert-space dimension."""

    _dimension_pretty: str = field(static=True, init=False, default="", repr=False)
    """Scientific-notation dimension string."""

    _plaquettes: tuple[tuple[int, ...], ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
    )
    """Minimal graph loops represented by edge indices."""

    _plaquette_lmax: int = field(static=True, init=False, default=0, repr=False)
    """Maximum plaquette length."""

    _plaquette_idx: Any = field(
        pytree=True,
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    """Padded plaquette edge-index array."""

    _plaquette_mask: Any = field(
        pytree=True,
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    """Boolean mask for valid padded plaquette entries."""

    def __post_init__(self) -> None:
        """Builds local charge ranges, layout, dimension, and plaquette data."""
        validate_gauge_dimensions(self.gauge_dimensions)
        local_space = u1_local_range(
            self.cutoff,
            self.step,
            positive_qn=self.positive_qn,
            qn_start=self.qn_start,
        )
        edge_space = edge_space_for_gauge_dimension(self.gauge_dimensions, local_space)
        layout = GraphHilbertLayout(self.graph, edge_space=edge_space)
        dimension = int(local_space.local_size) ** int(layout.size)
        plaquette_data = plaquette_metadata(self.graph)

        object.__setattr__(self, "local_space", local_space)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_local_spaces", (local_space,) * int(layout.size))
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(self, "_dimension_pretty", scientific_int(dimension))
        object.__setattr__(self, "_plaquettes", plaquette_data[0])
        object.__setattr__(self, "_plaquette_lmax", plaquette_data[1])
        object.__setattr__(self, "_plaquette_idx", plaquette_data[2])
        object.__setattr__(self, "_plaquette_mask", plaquette_data[3])

    def __hash__(self) -> int:
        """Returns a structural hash for the U(1) Hilbert space."""
        return hash(
            (
                type(self),
                self.graph,
                self.cutoff,
                self.step,
                self.gauge_dimensions,
                self.positive_qn,
                self.qn_start,
            )
        )

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Returns an informative representation of concrete U(1) metadata.

        Concrete subclasses must include enough information for debugging and
        interactive use, such as cutoff, step, local size, graph edge count,
        gauge-component count, constrained status, and indexability. The method
        is abstract so public subclasses make their chosen representation
        explicit.
        """
