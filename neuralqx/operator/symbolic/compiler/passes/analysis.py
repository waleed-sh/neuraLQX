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


"""Symbolic max-connection-size analysis compiler pass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.passes.base import AbstractSymbolicPass
from neuralqx.operator.symbolic.debug import event as debug_event


def _compute_term_max_conn_size(term: Any, _hilbert_size: int) -> int:
    """
    Computes an upper bound on max connected states for one IR term.

    Bound = M * E where M is the number of index tuples and E is the number
    of emissions per tuple.
    """
    if term.max_conn_size_hint is not None:
        return int(term.max_conn_size_hint)

    E = len(term.effective_emissions)
    M = len(term.iterator.index_sets)
    return max(1, M * E)


def _compute_term_fanout(term: Any, _hilbert_size: int) -> int:
    """Backward-compatible alias for ``_compute_term_max_conn_size``."""
    return _compute_term_max_conn_size(term, _hilbert_size)


class SymbolicMaxConnSizeAnalysisPass(AbstractSymbolicPass):
    """
    Computes per-term and total max-connection-size bounds.

    Analysis keys written:
        ``"term_max_conn_sizes"``  - ``dict[unique_key, int]``
        ``"total_max_conn_size"``  - ``int``
    """

    @property
    def name(self) -> str:
        return "symbolic_max_conn_size_analysis"

    def run(
        self,
        context: SymbolicCompilationContext,
    ) -> Mapping[str, Any] | None:
        hilbert_size = context.ir.hilbert_size
        term_max_conn_sizes: dict[str, int] = {}

        for idx, term in enumerate(context.ir.terms):
            # Key by (index, name) to avoid collisions from identically-named terms
            unique_key = f"{idx}:{term.name}"
            term_max_conn_sizes[unique_key] = _compute_term_max_conn_size(
                term, hilbert_size
            )

        total_max_conn_size = sum(term_max_conn_sizes.values())

        context.set_analysis("term_max_conn_sizes", term_max_conn_sizes)
        context.set_analysis("total_max_conn_size", total_max_conn_size)
        # Backward-compatible analysis keys.
        context.set_analysis("term_fanouts", term_max_conn_sizes)
        context.set_analysis("total_fanout", total_max_conn_size)
        debug_event(
            "computed max_conn_size analysis",
            scope="passes",
            pass_name=self.name,
            tag="PASS",
            term_count=len(term_max_conn_sizes),
            total_max_conn_size=total_max_conn_size,
            hilbert_size=hilbert_size,
        )

        return {
            "term_max_conn_sizes": term_max_conn_sizes,
            "total_max_conn_size": total_max_conn_size,
        }


SymbolicFanoutAnalysisPass = SymbolicMaxConnSizeAnalysisPass


__all__ = [
    "SymbolicFanoutAnalysisPass",
    "SymbolicMaxConnSizeAnalysisPass",
    "_compute_term_fanout",
    "_compute_term_max_conn_size",
]
