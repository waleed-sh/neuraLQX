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


"""Symbolic fanout-analysis compiler pass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
    AbstractSymbolicPass,
)


def _compute_term_fanout(term: Any, _hilbert_size: int) -> int:
    """
    Computes an upper-bound fanout for one IR term.

    Fanout = M * E where M is the number of index tuples and E is the number
    of emissions per tuple.
    """
    if term.fanout_hint is not None:
        return int(term.fanout_hint)

    E = len(term.effective_emissions)
    M = len(term.iterator.index_sets)
    return max(1, M * E)


class SymbolicFanoutAnalysisPass(AbstractSymbolicPass):
    """
    Computes per-term and total fanout (max connected states) bounds.

    Analysis keys written:
        ``"term_fanouts"``  - ``dict[unique_key, int]``
        ``"total_fanout"``  - ``int``
    """

    @property
    def name(self) -> str:
        return "symbolic_fanout_analysis"

    def run(
        self,
        context: SymbolicCompilationContext,
    ) -> Mapping[str, Any] | None:
        hilbert_size = context.ir.hilbert_size
        term_fanouts: dict[str, int] = {}

        for idx, term in enumerate(context.ir.terms):
            # Key by (index, name) to avoid collisions from identically-named terms
            unique_key = f"{idx}:{term.name}"
            term_fanouts[unique_key] = _compute_term_fanout(term, hilbert_size)

        total_fanout = sum(term_fanouts.values())

        context.set_analysis("term_fanouts", term_fanouts)
        context.set_analysis("total_fanout", total_fanout)

        return {
            "term_fanouts": term_fanouts,
            "total_fanout": total_fanout,
        }


__all__ = ["SymbolicFanoutAnalysisPass"]
