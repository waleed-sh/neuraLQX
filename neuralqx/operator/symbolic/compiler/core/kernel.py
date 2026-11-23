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

"""Immutable structured kernel plans, independent of JAX and operator wrappers."""

from __future__ import annotations

import dataclasses
from typing import Any

from neuralqx.operator.symbolic.compiler.core.actions import _Node
from neuralqx.operator.symbolic.compiler.core.actions import _Target
from neuralqx.operator.symbolic.compiler.core.connectivity import (
    ConnectivityDeduplicationPlan,
)
from neuralqx.operator.symbolic.ir.update import UpdateProgram


@dataclasses.dataclass(frozen=True)
class KernelAction:
    """One instantiated emission with ordered scalar and sparse target expressions.

    Attributes:
        origin: Original term, iterator-row, and emission positions.
        bindings: Static iterator bindings, in declaration order.
        target: Final sparse coordinate expressions or an opaque update key.
        expressions: Term guard, emission guard, and completed-target amplitude.
        program: Original program, retained for dynamic-index fallback.
        mapped: Whether the original emission is mapped over iterator visits.
        valid: Whether the program is free of top-level invalidation.
    """

    origin: tuple[int, int, int]
    bindings: tuple[tuple[str, int], ...]
    target: _Target
    expressions: tuple[_Node, _Node, _Node]
    program: UpdateProgram
    mapped: bool
    valid: bool


@dataclasses.dataclass(frozen=True)
class ScalarBatch:
    """Compatible scalar DAGs evaluated over an explicit action axis.

    Only source read indices are parameterized. Constants retain their values,
    scalar types, and weak-promotion behavior. Repeated reads use the same slot.

    Attributes:
        actions: Original branch positions in this batch.
        expressions: Shared expression templates with input-slot leaves.
        indices: Source coordinate bound to each input slot per action.
        mapped: Preserve iterator mapping for guarded scalar evaluation.
    """

    actions: tuple[int, ...]
    expressions: tuple[_Node, ...]
    indices: tuple[tuple[int, ...], ...]
    mapped: bool


@dataclasses.dataclass(frozen=True)
class KernelPlan:
    """Proof-preserving separation of scalar work and target materialization.

    Attributes:
        actions: Instantiated emissions in original output order.
        scalar_batches: Compatible amplitude and guard DAGs.
        representatives: Branch providing each target-equivalence class.
        connectivity: Existing exact assembly plan, or ``None`` in raw mode.
        local_states: Finite local basis used by specialization-time proofs.
        wrap_spec: Declared modular minimum and period, when available.
        dense_table_limit: Maximum coordinate-map cells for dense selection.
        fallback_reason: Reason the generic kernel is required, or ``None``.
    """

    actions: tuple[KernelAction, ...]
    scalar_batches: tuple[ScalarBatch, ...]
    representatives: tuple[int, ...]
    connectivity: ConnectivityDeduplicationPlan | None
    local_states: tuple[Any, ...]
    wrap_spec: tuple[int, int] | None
    dense_table_limit: int = 65_536
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Returns bounded, versioned metadata without expression payloads."""
        return {
            "version": 1,
            "enabled": self.fallback_reason is None,
            "raw_actions": len(self.actions),
            "target_classes": len(self.representatives),
            "scalar_batches": len(self.scalar_batches),
            "scalar_input_slots": sum(len(b.indices[0]) for b in self.scalar_batches),
            "opaque_targets": sum(a.target.opaque is not None for a in self.actions),
            "target_coordinate_expressions": sum(
                len(self.actions[i].target.coordinates) for i in self.representatives
            ),
            "dense_table_limit": self.dense_table_limit,
            "fallback_reason": self.fallback_reason,
            "schedule": "specialized_by_shape_and_dtype",
        }


def scalar_template(
    nodes: tuple[_Node, ...],
) -> tuple[tuple[_Node, ...], tuple[int, ...]]:
    """Abstracts source indices while retaining typed operations and aliasing.

    Args:
        nodes: Ordered scalar expression roots sharing one source environment.

    Returns:
        Shared expression roots and their ordered source-index bindings.
    """
    slots: dict[int, int] = {}

    def visit(node: _Node) -> _Node:
        if node[0] == "read":
            site = int(node[1])
            return ("input", slots.setdefault(site, len(slots)))
        if node[0] in {"const", "array", "convert", "symbol", "target_read"}:
            if node[0] == "convert":
                return ("convert", node[1], visit(node[2]))
            return node
        return (
            node[0],
            *(visit(arg) if isinstance(arg, tuple) else arg for arg in node[1:]),
        )

    return tuple(visit(n) for n in nodes), tuple(slots)
