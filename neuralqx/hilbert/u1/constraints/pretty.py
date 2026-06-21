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


"""Readable formatting for U(1) gauge-fixing relations.

Formatting helpers turn resolved edge-index relations back into graph-edge
representations so users can inspect the constructive gauge-fixing slice.
"""

from __future__ import annotations

from collections.abc import Sequence

from neuralqx.graph import AbstractGraph

from .relation import GaugeRelation


def pretty_gauge_fixing(
    graph: AbstractGraph, relations: Sequence[GaugeRelation]
) -> str:
    """Formats parsed gauge fixing in terms of graph edges.

    Args:
        graph: Graph used to convert edge indices back to edge objects.
        relations: Resolved constructive gauge relations.

    Returns:
        Multiline human-readable gauge-fixing description.
    """
    if not relations:
        return "No gauge-fixing relations."
    return "\n".join(_format_relation(graph, relation) for relation in relations)


def _format_relation(graph: AbstractGraph, relation: GaugeRelation) -> str:
    """Formats one constructive gauge relation."""
    lhs = graph.index_to_edge(relation.lhs)
    rhs = []
    for edge, sign in relation.rhs:
        prefix = "+" if sign > 0 else "-"
        rhs.append(f"{prefix} {graph.index_to_edge(edge)!r}")
    rhs_text = " ".join(rhs).lstrip("+").strip() if rhs else "0"
    return f"{lhs!r} = {rhs_text}"
