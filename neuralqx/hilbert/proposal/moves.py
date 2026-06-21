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


"""Typed proposal move descriptions.

Proposal moves are immutable descriptors that tell the proposal dispatcher how
to mutate a batch of states. Generic moves operate on flat discrete sites,
while U(1) and SU(2) subpackages define domain-specific moves.
"""

from __future__ import annotations

import abc
from typing import Literal

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.struct import field


class AbstractProposalMove(Struct, metaclass=StructABCMeta):
    """Base type for immutable proposal move descriptors.

    Proposal moves describe how a Markov-chain transition should alter a batch
    of Hilbert-space states. They intentionally store only static
    configuration, while the actual update kernels live in the proposal
    dispatchers and concrete Hilbert subpackages.

    Subclasses should expose validated, hashable fields that fully determine
    the proposal family. Generic subclasses operate on flat local indices,
    while U(1) and SU(2) subclasses add physics-aware moves without changing
    this descriptor contract.
    """


class UniformSiteUpdate(AbstractProposalMove):
    """Proposal descriptor that resamples one or more flat sites.

    Args:
        n_sites: Number of distinct sites to update in each state.
        avoid_current: Whether to avoid proposing the current local value when
            the selected site has more than one available value.

    Attributes:
        n_sites: Number of distinct sites to update in each state.
        avoid_current: Whether to avoid proposing the current local value.
    """

    n_sites: int = field(static=True, default=1)
    """Number of distinct sites to update in each state."""
    avoid_current: bool = field(static=True, default=True)
    """Whether to avoid proposing the current local value."""

    def __post_init__(self) -> None:
        """Validates the number of selected sites."""
        if self.n_sites <= 0:
            raise ValueError("n_sites must be > 0.")

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.n_sites, self.avoid_current))


class AdjacentSiteUpdate(AbstractProposalMove):
    """Proposal descriptor that increments or decrements local indices by one.

    Args:
        n_sites: Number of distinct sites to update in each state.
        boundary: Boundary behavior for the local index update.

    Attributes:
        n_sites: Number of distinct sites to update in each state.
        boundary: Boundary behavior for the local index update.
    """

    n_sites: int = field(static=True, default=1)
    """Number of distinct sites to update in each state."""
    boundary: Literal["wrap", "clamp"] = field(static=True, default="wrap")
    """Boundary behavior for the local index update."""

    def __post_init__(self) -> None:
        """Validates the number of selected sites and boundary mode."""
        if self.n_sites <= 0:
            raise ValueError("n_sites must be > 0.")
        if self.boundary not in ("wrap", "clamp"):
            raise ValueError("boundary must be either 'wrap' or 'clamp'.")

    def __hash__(self) -> int:
        """Returns a structural hash for the move descriptor."""
        return hash((type(self), self.n_sites, self.boundary))


__all__ = ["AbstractProposalMove", "AdjacentSiteUpdate", "UniformSiteUpdate"]
