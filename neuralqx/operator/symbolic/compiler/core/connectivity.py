# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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


"""Backend-independent connectivity assembly plans and strategy selection."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ConnectivityDeduplicationPlan:
    """Static assembly plan for a deduplicated connectivity kernel.

    Attributes:
        raw_size: Number of emitted branches before deduplication.
        groups: Original branch indices grouped by identical target expression.
        collision_components: Groups requiring joint runtime collision checks.
        key_columns: State coordinates sufficient to distinguish each component.
        component_capacities: Finite-image upper bound for each collision component.
        key_ranges: Exact-integer coordinate ranges for lossless key packing,
            or ``None`` when a component needs full coordinate keys.
        compact: Whether amplitudes can vanish and require stable compaction.
        proof_min: Smallest intermediate integer used in collision proofs.
        proof_max: Largest intermediate integer used in collision proofs.
        proof_integer_limit: Exact-integer limit of arithmetic operands used by proofs.
        output_size: Proven upper bound on distinct nonzero targets, at least one.
    """

    raw_size: int
    groups: tuple[tuple[int, ...], ...]
    collision_components: tuple[tuple[int, ...], ...]
    key_columns: tuple[tuple[int, ...], ...]
    component_capacities: tuple[int, ...]
    key_ranges: tuple[tuple[tuple[int, int], ...] | None, ...]
    compact: bool
    proof_min: int
    proof_max: int
    proof_integer_limit: int
    output_size: int

    def as_dict(self) -> dict[str, Any]:
        """Returns a compact summary suitable for artifact metadata."""
        return {
            "raw_size": self.raw_size,
            "target_class_count": len(self.groups),
            "output_size": self.output_size,
            "runtime_component_sizes": tuple(map(len, self.collision_components)),
            "runtime_key_widths": tuple(map(len, self.key_columns)),
            "runtime_component_capacities": self.component_capacities,
            "packed_key_widths": tuple(
                None if ranges is None else packed_key_width(ranges)
                for ranges in self.key_ranges
            ),
            "compact": self.compact,
            "proof_range": (self.proof_min, self.proof_max),
            "proof_integer_limit": self.proof_integer_limit,
        }


def packed_key_width(ranges: tuple[tuple[int, int], ...]) -> int:
    """Returns the number of 32-bit words needed for lossless coordinate keys."""
    return (sum((hi - lo).bit_length() for lo, hi in ranges) + 31) // 32


def collision_strategy(size: int, width: int, *, key_bits: int | None = None) -> str:
    """Selects an exact runtime algorithm from its bounded temporary footprint."""
    if (
        key_bits is not None
        and size >= 8
        and key_bits <= 12
        and 2**key_bits <= 4 * size
    ):
        return "workspace"
    if size * size * max(1, width) <= 16_384:
        return "pairwise"
    return "sort"


__all__ = ["ConnectivityDeduplicationPlan", "collision_strategy", "packed_key_width"]
