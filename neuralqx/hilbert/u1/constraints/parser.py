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


"""Parser for user-facing U(1) gauge-fixing declarations.

The parser accepts edge indices, edge-like literals, and signed string tokens
from compact gauge-fixing specifications. It resolves all references to base
graph edge indices and records whether each right-hand side edge is added or
subtracted.
"""

from __future__ import annotations

from typing import Any

import ast

from neuralqx.graph import AbstractGraph
from neuralqx.utils.typing import Edge

from .errors import U1GaugeFixingError
from .relation import GaugeRelation
from .types import GaugeFixingSpec


def parse_gauge_fixing(
    graph: AbstractGraph,
    spec: GaugeFixingSpec,
) -> tuple[GaugeRelation, ...]:
    """Parses a gauge-fixing array into base-edge relations.

    Args:
        graph: Graph used to resolve edge-like tokens.
        spec: User-facing gauge-fixing declaration.

    Returns:
        Tuple of constructive gauge relations.
    """
    if spec is None:
        raise U1GaugeFixingError("Gauge fixing is None.")

    relations: list[GaugeRelation] = []
    lhs_seen: set[int] = set()
    for condition in spec:
        if len(condition) != 2:
            raise U1GaugeFixingError(
                "Each gauge condition must be [lhs_list, rhs_list]."
            )
        lhs_list, rhs_list = condition
        if len(lhs_list) != 1:
            raise U1GaugeFixingError(
                "Each gauge condition must have exactly one LHS edge."
            )
        lhs = edge_to_index(graph, lhs_list[0])
        if lhs in lhs_seen:
            raise U1GaugeFixingError(f"Duplicate LHS edge {lhs} in gauge fixing.")
        lhs_seen.add(lhs)

        subtract_mode = False
        rhs: list[tuple[int, int]] = []
        for token in rhs_list:
            if isinstance(token, str) and token.strip() == "N":
                subtract_mode = True
                continue
            edge_token, explicit_minus = _split_signed_token(token)
            sign = -1 if (int(subtract_mode) + int(explicit_minus)) % 2 else 1
            rhs.append((edge_to_index(graph, edge_token), sign))

        relations.append(GaugeRelation(lhs=lhs, rhs=tuple(rhs)))

    return tuple(relations)


def edge_to_index(graph: AbstractGraph, token: Any) -> int:
    """Resolves an edge token or edge index.

    Args:
        graph: Graph used for edge lookup.
        token: Integer edge index, edge-like value, or string representation.

    Returns:
        Integer base edge index.
    """
    if isinstance(token, int) and not isinstance(token, bool):
        if token < 0 or token >= graph.n_edges:
            raise U1GaugeFixingError(f"Edge index {token} out of range.")
        return token
    if isinstance(token, str):
        maybe = _maybe_int_from_string(token)
        if maybe is not None:
            return edge_to_index(graph, maybe)
        token = _literal_edge(token)
    try:
        return graph.edge_to_index(Edge.from_like(token))
    except Exception as exc:
        raise U1GaugeFixingError(
            f"Cannot resolve gauge-fixing edge token {token!r}."
        ) from exc


def _split_signed_token(token: Any) -> tuple[Any, bool]:
    """Splits an optional leading sign from a gauge-fixing token."""
    if not isinstance(token, str):
        return token, False
    stripped = token.strip()
    if stripped.startswith("-"):
        return stripped[1:].strip(), True
    if stripped.startswith("+"):
        return stripped[1:].strip(), False
    return stripped, False


def _maybe_int_from_string(token: str) -> int | None:
    """Parses an integer string token when possible."""
    stripped = token.strip()
    if stripped.startswith(("+", "-")):
        return int(stripped) if stripped[1:].isdigit() else None
    return int(stripped) if stripped.isdigit() else None


def _literal_edge(token: str) -> Any:
    """Parses a literal edge token when possible."""
    try:
        return ast.literal_eval(token)
    except (SyntaxError, ValueError):
        return token
