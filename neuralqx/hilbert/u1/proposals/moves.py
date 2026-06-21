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


"""Proposal move descriptions for U(1)^N Hilbert spaces.

Move descriptors are immutable configuration objects. They do not update
states themselves, but they tell the U(1) proposal dispatcher which free-edge
or plaquette proposal kernel to use.
"""

from __future__ import annotations

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.struct import field


class AbstractU1Move(Struct, metaclass=StructABCMeta):
    """Base descriptor for U(1)-specific proposal families.

    U(1) move descriptors express updates in graph and gauge-component terms
    instead of raw flat-site terms. They are intentionally immutable and do not
    mutate states themselves. The U(1) proposal mixin and package dispatcher
    inspect the concrete descriptor type and call the matching JAX update
    kernel.

    Concrete subclasses should only store static configuration needed by the
    proposal, such as how many free edges to update or whether a plaquette move
    should touch one gauge component or all components.
    """


class FreeEdgeFlipSingleGauge(AbstractU1Move):
    """Updates ``n_edges`` free variables from the pooled gauge-component set.

    The proposal chooses free base-edge variables from the flattened collection
    of all gauge components. When ``gauge_dimensions`` is greater than one,
    this means a single call can update an edge in one component without
    necessarily updating the same base edge in the other components.

    With ``adjacency=False`` the selected variables are replaced by random
    charge values from the local range. With ``adjacency=True`` the proposal
    moves selected variables to neighboring modular values, which is useful for
    local random walks in charge space.

    Args:
        n_edges: Number of free base-edge variables to update.
        adjacency: Whether to use adjacent modular values instead of random
            replacement values.

    Attributes:
        n_edges: Number of free base-edge variables to update.
        adjacency: Whether to use adjacent modular values.
    """

    n_edges: int = field(static=True, default=1)
    """Number of free base-edge variables to update."""
    adjacency: bool = field(static=True, default=False)
    """Whether to use adjacent modular values."""

    def __post_init__(self) -> None:
        """Validates the requested number of free-edge updates."""
        if self.n_edges < 0:
            raise ValueError("n_edges must be non-negative.")

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.n_edges, self.adjacency))


class FreeEdgeFlipAllGauge(AbstractU1Move):
    """Updates ``n_edges`` free variables independently in each gauge copy.

    The proposal chooses base edges and applies an update in every U(1) gauge
    component. This keeps the update scope aligned across gauge copies while
    still allowing each component to receive its own random replacement or
    adjacent modular step.

    Gauge-fixed Hilbert spaces restrict candidate base edges to the independent
    free set and reconstruct slave edges after the update. Unconstrained spaces
    treat every base edge as free and return the update directly.

    Args:
        n_edges: Number of free base-edge variables to update per gauge copy.
        adjacency: Whether to use adjacent modular values instead of random
            replacement values.

    Attributes:
        n_edges: Number of free base-edge variables to update per gauge copy.
        adjacency: Whether to use adjacent modular values.
    """

    n_edges: int = field(static=True, default=1)
    """Number of free base-edge variables to update per gauge copy."""
    adjacency: bool = field(static=True, default=False)
    """Whether to use adjacent modular values."""

    def __post_init__(self) -> None:
        """Validates the requested number of free-edge updates."""
        if self.n_edges < 0:
            raise ValueError("n_edges must be non-negative.")

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.n_edges, self.adjacency))


class PlaquetteFlipSingleGauge(AbstractU1Move):
    """Descriptor for a plaquette update in one gauge component.

    The move selects a plaquette from the graph, chooses a random update
    direction, and applies the cyclic charge shift only within one randomly
    chosen U(1) gauge component. Gauge-fixed spaces subsequently reconstruct
    dependent slave edges so the proposal remains on the valid gauge-fixing
    slice.

    The descriptor has no fields because the plaquette and gauge component are
    sampled from the supplied JAX key for each proposal call.
    """

    def __hash__(self) -> int:
        """Returns a type-level hash for the stateless move descriptor."""
        return hash(type(self))


class PlaquetteFlipAllGauge(AbstractU1Move):
    """Descriptor for plaquette updates across all gauge components.

    The move applies plaquette shifts to every U(1) gauge component of each
    state row. Each component receives its own random direction, while the
    selected plaquette structure comes from the graph metadata cached on the
    Hilbert space. Gauge-fixed spaces reimpose constructive relations after the
    update.

    The descriptor has no fields because all stochastic choices are made from
    the supplied JAX key by the proposal kernel.
    """

    def __hash__(self) -> int:
        """Returns a type-level hash for the stateless move descriptor."""
        return hash(type(self))
