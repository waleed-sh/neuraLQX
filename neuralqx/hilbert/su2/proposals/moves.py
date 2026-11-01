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


"""Proposal move descriptions for SU(2) spin-network Hilbert spaces.

SU(2) proposal moves describe how a Markov chain should explore valid
spin-network states: loop shifts change edge spins around graph loops, vertex
moves resample recoupling channels, heatbath moves try edge sectors, and mixed
moves combine those families.
"""

from __future__ import annotations

from neuralqx.hilbert.proposal.moves import AbstractProposalMove
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.struct import field


class AbstractSU2Move(AbstractProposalMove, metaclass=StructABCMeta):
    """Base descriptor for SU(2)-specific spin-network proposals.

    SU(2) proposal descriptors express admissibility-aware updates in terms of
    graph loops, vertices, edge sectors, or mixtures of those operations. They
    store static configuration only. Proposal kernels perform the stochastic
    choices and state mutation using a supplied JAX key.

    Concrete subclasses should validate their configuration in
    ``__post_init__`` and remain hashable so they can participate in JAX static
    dispatch and cached Hilbert-space workflows.
    """


class LoopFundamentalMove(AbstractSU2Move):
    """Proposal that applies ``two_j -> two_j +/- 1`` around a graph loop.

    The move picks a minimal graph loop, shifts each edge label on that loop by
    one doubled-spin unit in the chosen direction, and accepts the result only
    when the completed spin-network state remains admissible. Retrying allows a
    chain row to attempt alternate loop or direction choices before keeping the
    original state.

    Attributes:
        loop: Optional fixed loop index.
        direction: Optional fixed direction, either ``-1`` or ``1``.
        max_attempts: Maximum loop proposals tried for each chain row.
    """

    loop: int | None = field(static=True, default=None)
    """Optional fixed loop index."""

    direction: int | None = field(static=True, default=None)
    """Optional fixed direction, either ``-1`` or ``1``."""

    max_attempts: int = field(static=True, default=1)
    """Maximum loop proposals tried for each chain row."""

    def __post_init__(self) -> None:
        """Validates direction and attempt count."""
        if self.direction is not None and int(self.direction) not in (-1, 1):
            raise ValueError("direction must be -1, 1, or None.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.loop, self.direction, self.max_attempts))


class VertexRecouplingMove(AbstractSU2Move):
    """Descriptor for resampling recoupling channels at one vertex.

    The move keeps the edge doubled-spin labels fixed and chooses a new
    admissible set of left-comb channels for a selected vertex. It is useful for
    exploring the internal intertwiner degrees of freedom without changing the
    surrounding edge-spin sector.

    Attributes:
        vertex: Optional fixed vertex position or id.
    """

    vertex: int | None = field(static=True, default=None)
    """Optional fixed vertex position or id."""

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.vertex))


class EdgeSpinHeatbathMove(AbstractSU2Move):
    """Proposal that tries edge-spin heatbath sectors with channel completion.

    The move selects an edge, samples candidate doubled-spin labels, and asks
    the Hilbert space to complete affected vertex channels in an admissible
    way. If completion fails for a candidate sector, the kernel may try again
    up to ``max_attempts`` and otherwise keeps a valid state.

    Attributes:
        edge: Optional fixed edge index.
        include_current: Whether sampled labels may equal the current label.
        max_attempts: Maximum edge proposals tried for each chain row.
    """

    edge: int | None = field(static=True, default=None)
    """Optional fixed edge index."""
    include_current: bool = field(static=True, default=True)
    """Whether sampled labels may equal the current label."""
    max_attempts: int = field(static=True, default=1)
    """Maximum edge proposals tried for each chain row."""

    def __post_init__(self) -> None:
        """Validates the attempt count."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.edge, self.include_current, self.max_attempts))


class MixedSU2Move(AbstractSU2Move):
    """Mixture descriptor for SU(2) proposal families.

    ``MixedSU2Move`` combines loop, vertex, and edge heatbath proposals behind
    a single descriptor. The proposal kernel samples one family according to
    ``weights`` for each update and then delegates to the corresponding move
    descriptor. This gives Markov chains a compact way to explore both edge
    spins and intertwiner channels.

    Attributes:
        weights: Relative probabilities for loop, vertex, and edge moves.
        loop_move: Loop proposal descriptor.
        vertex_move: Vertex-channel proposal descriptor.
        edge_move: Edge-spin heatbath descriptor.
    """

    weights: tuple[float, float, float] = field(static=True, default=(0.5, 0.3, 0.2))
    """Relative probabilities for loop, vertex, and edge moves."""

    loop_move: LoopFundamentalMove = field(
        static=True, default_factory=LoopFundamentalMove
    )
    """Loop proposal descriptor."""

    vertex_move: VertexRecouplingMove = field(
        static=True, default_factory=VertexRecouplingMove
    )
    """Vertex-channel proposal descriptor."""

    edge_move: EdgeSpinHeatbathMove = field(
        static=True, default_factory=EdgeSpinHeatbathMove
    )
    """Edge-spin heatbath descriptor."""

    def __post_init__(self) -> None:
        """Normalizes and validates mixture weights."""
        weights = tuple(float(x) for x in self.weights)
        if (
            len(weights) != 3
            or any(weight < 0.0 for weight in weights)
            or sum(weights) <= 0.0
        ):
            raise ValueError(
                "weights must be three non-negative values with positive sum."
            )
        object.__setattr__(self, "weights", weights)

    def __hash__(self) -> int:
        """Returns a structural hash for the mixed move descriptor."""
        return hash(
            (type(self), self.weights, self.loop_move, self.vertex_move, self.edge_move)
        )


__all__ = [
    "AbstractSU2Move",
    "EdgeSpinHeatbathMove",
    "LoopFundamentalMove",
    "MixedSU2Move",
    "VertexRecouplingMove",
]
