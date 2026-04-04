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


"""Symbolic term-fusion planning compiler pass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
    AbstractSymbolicPass,
)


def _fusion_key(term: Any) -> tuple:
    """
    Computes the fusion grouping key for an IR term.

    Two terms are fusion-compatible when they share the same iterator kind and
    neither contains an ``invalidate_branch`` op (which would require separate
    branch handling). Fusing compatible terms reduces loop count in the
    generated kernel.

    Returns a tuple ``(iterator_kind, has_invalidate)`` used as the grouping
    key.
    """
    return (
        term.iterator.kind,
        term.update_program.has_invalidate(),
    )


class SymbolicFusionPass(AbstractSymbolicPass):
    """
    Groups IR terms into fusion-compatible clusters.

    This post-cache pass (only runs on cache misses) plans which terms can be
    lowered inside a single vmapped loop rather than separate loops.

    When ``enable_fusion`` is ``False`` in the compiler options, each term is
    placed in its own singleton group.

    Analysis keys written:
        ``"fusion_groups"``, ``list[list[str]]``, each inner list is a group
        of term names that can be lowered in a single fused kernel loop.

    The lowerer reads ``"fusion_groups"`` to decide loop structure.
    """

    @property
    def name(self) -> str:
        return "symbolic_fusion_planning"

    def run(
        self,
        context: SymbolicCompilationContext,
    ) -> Mapping[str, Any] | None:
        if not context.options.enable_fusion:
            # Each term in its own singleton group
            fusion_groups = [[term.name] for term in context.ir.terms]
            context.set_analysis("fusion_groups", fusion_groups)
            return {
                "fusion_enabled": False,
                "group_count": len(fusion_groups),
            }

        # Group by fusion key, preserving encounter order within each group
        groups_by_key: dict[tuple, list[str]] = {}
        for term in context.ir.terms:
            key = _fusion_key(term)
            if key not in groups_by_key:
                groups_by_key[key] = []
            groups_by_key[key].append(term.name)

        fusion_groups = list(groups_by_key.values())
        context.set_analysis("fusion_groups", fusion_groups)

        return {
            "fusion_enabled": True,
            "group_count": len(fusion_groups),
            "groups": fusion_groups,
        }


__all__ = ["SymbolicFusionPass"]
