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


"""IR traversal helpers used by diagnostic rules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.operator.symbolic.ir.predicates import PredicateExpr
from neuralqx.operator.symbolic.ir.term import SymbolicIRTerm
from neuralqx.operator.symbolic.ir.update import UpdateOp


def iter_amplitude_nodes(root: AmplitudeExpr) -> Iterable[AmplitudeExpr]:
    """Yields one amplitude-expression subtree in pre-order.

    Args:
        root: Amplitude-expression root node.

    Yields:
        Amplitude nodes in traversal order.
    """
    yield root
    for arg in root.args:
        if isinstance(arg, AmplitudeExpr):
            yield from iter_amplitude_nodes(arg)
        elif isinstance(arg, tuple):
            for item in arg:
                if isinstance(item, AmplitudeExpr):
                    yield from iter_amplitude_nodes(item)


def iter_predicate_amplitude_nodes(predicate: PredicateExpr) -> Iterable[AmplitudeExpr]:
    """Yields amplitude-expression leaves reachable from one predicate.

    Args:
        predicate: Predicate-expression root node.

    Yields:
        Amplitude-expression leaves from this predicate.
    """
    for arg in predicate.args:
        if isinstance(arg, PredicateExpr):
            yield from iter_predicate_amplitude_nodes(arg)
        elif isinstance(arg, AmplitudeExpr):
            yield from iter_amplitude_nodes(arg)


def iter_update_amplitude_nodes(update_op: UpdateOp) -> Iterable[AmplitudeExpr]:
    """Yields amplitude nodes reachable from one update operation.

    Args:
        update_op: Update operation.

    Yields:
        Amplitude nodes reachable from operation parameters.
    """

    def _walk(value: Any) -> Iterable[AmplitudeExpr]:
        if isinstance(value, AmplitudeExpr):
            yield from iter_amplitude_nodes(value)
            return
        if isinstance(value, PredicateExpr):
            yield from iter_predicate_amplitude_nodes(value)
            return
        if isinstance(value, UpdateOp):
            for _key, item in value.params:
                yield from _walk(item)
            return
        if isinstance(value, tuple):
            for item in value:
                yield from _walk(item)

    for _key, item in update_op.params:
        yield from _walk(item)


def iter_term_amplitude_nodes(term: SymbolicIRTerm) -> Iterable[AmplitudeExpr]:
    """Yields all amplitude nodes reachable from one symbolic IR term.

    Args:
        term: Symbolic operator term.

    Yields:
        Amplitude-expression nodes appearing in predicates, emissions, and updates.
    """
    yield from iter_predicate_amplitude_nodes(term.predicate)
    for emission in term.effective_emissions:
        yield from iter_predicate_amplitude_nodes(emission.predicate)
        yield from iter_amplitude_nodes(emission.amplitude)
        for op in emission.update_program.ops:
            yield from iter_update_amplitude_nodes(op)


def iter_term_static_index_nodes(term: SymbolicIRTerm) -> Iterable[tuple[str, int]]:
    """Yields all static-index uses in one term.

    Args:
        term: Symbolic operator term.

    Yields:
        Tuples ``(op_name, flat_index)`` for static source/target index nodes.
    """
    for amplitude in iter_term_amplitude_nodes(term):
        if amplitude.op in {"static_index", "static_emitted_index"}:
            yield amplitude.op, int(amplitude.args[0])


__all__ = [
    "iter_amplitude_nodes",
    "iter_predicate_amplitude_nodes",
    "iter_term_amplitude_nodes",
    "iter_term_static_index_nodes",
    "iter_update_amplitude_nodes",
]
