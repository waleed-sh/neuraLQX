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


"""Gauge-fixing relation data structures.

A constructive U(1) gauge-fixing relation designates one slave edge and
expresses it as a signed modular sum of one or more independent edges.
"""

from __future__ import annotations

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field


class GaugeRelation(Struct):
    """One relation ``lhs = sum(sign_i * rhs_i)`` on base edge indices.

    A relation declares that one slave base edge is not sampled independently.
    Instead, its value is reconstructed as a signed modular sum of source base
    edges for each U(1) gauge component. The relation stores only base-edge
    indices, while gauge-component expansion happens in the constraint and
    projection kernels.

    Args:
        lhs: Slave base-edge index reconstructed by the relation.
        rhs: Signed source base-edge indices.

    Attributes:
        lhs: Slave base-edge index reconstructed by the relation.
        rhs: Signed source base-edge indices.
    """

    lhs: int = field(static=True)
    """Slave base-edge index reconstructed by the relation."""
    rhs: tuple[tuple[int, int], ...] = field(static=True)
    """Signed source base-edge indices."""

    def __post_init__(self) -> None:
        """Normalizes integer fields and validates signs."""
        lhs = int(self.lhs)
        rhs = tuple((int(edge), int(sign)) for edge, sign in self.rhs)
        if any(sign not in (-1, 1) for _, sign in rhs):
            raise ValueError("Gauge relation signs must be +/-1.")
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)

    def __hash__(self) -> int:
        """Returns a structural hash for the gauge relation."""
        return hash((type(self), self.lhs, self.rhs))
