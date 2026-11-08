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


"""Validation helpers for shipped prebuilt operators."""

from __future__ import annotations

from typing import Any

from neuralqx.graph import AbstractGraph
from neuralqx.graph import HalfLadderGraph
from neuralqx.hilbert.u1.spaces.abstract import AbstractU1Hilbert


def require_u1_hilbert(
    hilbert: Any, *, operator: str | None = None
) -> AbstractU1Hilbert:
    r"""Require a Hilbert space with the U(1) charge and layout interface.

    Prebuilt U(1) operators need charge bounds, base-edge indices, and gauge-copy
    metadata. A generic discrete Hilbert space does not guarantee these fields.
    This check runs during construction, before any JAX kernel is traced.

    Args:
        hilbert: Candidate Hilbert-space object.
        operator: Optional name to include in the diagnostic.

    Returns:
        result (AbstractU1Hilbert): The original object, without copying it.

    Raises:
        TypeError: If the object does not implement AbstractU1Hilbert.
    """
    # Validate the U(1) contract before reading any charge metadata.
    if not isinstance(hilbert, AbstractU1Hilbert):
        # Use the factory name when available to identify the failing construction.
        name = "Prebuilt U(1) operators" if operator is None else operator
        # Report the actual type rather than allowing a later missing-attribute error.
        raise TypeError(
            f"{name} require an AbstractU1Hilbert. Got {type(hilbert).__name__}."
        )
    # Preserve object identity shared by the model and its operators.
    return hilbert


def require_gauge_dimensions(
    hilbert: AbstractU1Hilbert,
    expected: int,
    *,
    operator: str | None = None,
) -> None:
    r"""Require the number of independent U(1) components used by a formula.

    The single-copy operators and the U(1)^3 volume and Thiemann formulas have
    different component contractions. Accepting the wrong count would change
    those contractions or index the wrong charge block.

    Args:
        hilbert: Validated U(1) space.
        expected: Required number of gauge copies.
        operator: Optional factory name for the error message.

    Raises:
        ValueError: If the actual number of copies differs from `expected`.
    """
    # Compare the model component count before building copy-dependent metadata.
    if int(hilbert.gauge_dimensions) != int(expected):
        # Attach the failed requirement to the requesting operator when possible.
        name = "operator" if operator is None else operator
        # Give both counts so the Hilbert-space configuration can be corrected.
        raise ValueError(
            f"{name} requires gauge_dimensions={expected}. Got {hilbert.gauge_dimensions}."
        )


def require_planar_graph(graph: AbstractGraph, *, model: str) -> None:
    r"""Require a graph declared planar for the spatially two-dimensional model.

    This checks the graph's model flag, not a new geometric planarity test. The
    flag controls which orientation table and volume prescription the graph
    provides to its operators.

    Args:
        graph: Graph used by the Hilbert space.
        model: Name of the model or operator requesting planar geometry.

    Raises:
        ValueError: If `graph.is_planar` is false.
    """
    # Use the graph declaration that also selects its sign-table construction.
    if not bool(graph.is_planar):
        # Reject the wrong dimensional prescription at operator construction.
        raise ValueError(f"{model} requires a planar graph.")


def require_nonplanar_graph(graph: AbstractGraph, *, model: str) -> None:
    r"""Require the graph flag used by the spatially three-dimensional model.

    The 3+1 volume uses triples of edge tangents. A graph declared planar uses a
    different sign-table prescription and cannot be passed to that kernel.
    This function does not test whether every individual triple is nondegenerate.

    Args:
        graph: Graph used by the Hilbert space.
        model: Name of the requesting model or operator.

    Raises:
        ValueError: If `graph.is_planar` is true.
    """
    # A planar declaration selects the incompatible two-dimensional prescription.
    if bool(graph.is_planar):
        # Fail before assembling triples from the wrong graph model.
        raise ValueError(f"{model} requires a non-planar graph.")


def require_half_ladder_graph(
    graph: AbstractGraph, *, operator: str
) -> HalfLadderGraph:
    r"""Require the radial-chain geometry used by the spherical operators.

    Spherical stencils depend on the ordered radial vertices and transverse-edge
    accessors of HalfLadderGraph. A general graph with similar coordinates does
    not provide that contract.

    Args:
        graph: Candidate graph.
        operator: Name of the spherical operator being constructed.

    Returns:
        result (HalfLadderGraph): The original half-ladder graph.

    Raises:
        ValueError: If the graph has another type.
    """
    # Check the radial-stencil interface before accessing its specialized methods.
    if not isinstance(graph, HalfLadderGraph):
        # Identify the graph type that cannot provide the spherical neighborhood.
        raise ValueError(
            f"{operator} requires HalfLadderGraph. Got {type(graph).__name__}."
        )
    # Return the same graph object after establishing its type.
    return graph


def require_single_vertex_three_edges(
    hilbert: AbstractU1Hilbert, *, operator: str
) -> None:
    r"""Check the charge-space assumptions of the three-direction QR formulas.

    The reduced kernels use exactly three base labels in one gauge copy and take
    fractional powers of non-negative spins. The check enforces those numerical
    requirements. Despite the historical helper name, it does not validate graph
    incidence or require the graph to contain only one Vertex object. A star
    with three edges has the expected three labels and is a valid input.

    Args:
        hilbert: Validated U(1) space supplied to a QR constraint.
        operator: Name to include in construction errors.

    Raises:
        ValueError: If there is more than one gauge copy, the base size is not
            three, or negative local spins are allowed.
    """
    # The three reduced labels represent directions, not three gauge-copy blocks.
    require_gauge_dimensions(hilbert, 1, operator=operator)
    # Every QR kernel indexes precisely the three base sites.
    if int(hilbert.tiny_size) != 3:
        # Explain the incompatible label count or negative-spin domain.
        raise ValueError(
            f"{operator} requires a three-edge single-vertex Hilbert space. "
            f"got {hilbert.tiny_size} base edges."
        )
    # Fractional flux powers in the reduced constraint require non-negative spins.
    if hilbert.q_min < 0:
        # Explain the incompatible label count or negative-spin domain.
        raise ValueError(
            f"{operator} requires non-negative QR spins. "
            "Construct the Hilbert space with positive_qn=True."
        )


def validate_site(hilbert: AbstractU1Hilbert, site: int, *, operator: str) -> int:
    r"""Validate a flat state-component index at operator construction.

    A site addresses the entire copy-major state, unlike an edge index confined
    to one base block. Booleans and non-Python integer types are rejected rather
    than silently converted. This keeps accidental fractional or logical inputs
    from selecting a physical component.

    Args:
        hilbert: U(1) space defining the flat state size.
        site: Proposed zero-based flat index.
        operator: Name of the requesting operator.

    Returns:
        result (int): The validated site.

    Raises:
        TypeError: If `site` is not a Python int or is a bool.
        ValueError: If the site is negative or outside the full configuration.
    """
    # Validate the type before any integer coercion could hide an invalid input.
    if isinstance(site, bool) or not isinstance(site, int):
        # Expose the original argument in the diagnostic.
        raise TypeError(f"{operator} site must be an integer. Got {site!r}.")
    # Use full state size because later gauge-copy blocks are valid site targets.
    if not 0 <= site < int(hilbert.size):
        # Reject Python-style negative indexing for physical site selection.
        raise ValueError(
            f"{operator} site must be in [0, {hilbert.size - 1}], got {site}."
        )
    # Return a static integer suitable for indexing inside a traced kernel.
    return int(site)


def validate_copy(hilbert: AbstractU1Hilbert, copy: int, *, operator: str) -> int:
    r"""Validate a zero-based gauge-component selector.

    The selector identifies a complete base-edge block in the copy-major state.
    It must be supplied as a Python integer. Validation precedes site arithmetic,
    so a boolean or fractional value cannot silently choose a component.

    Args:
        hilbert: U(1) space defining the number of gauge copies.
        copy: Proposed gauge-component index.
        operator: Name to include in error messages.

    Returns:
        result (int): The validated component index.

    Raises:
        TypeError: If `copy` is not a Python int or is a bool.
        ValueError: If the copy is outside the available component range.
    """
    # Reject boolean and fractional inputs before constructing any flat indices.
    if isinstance(copy, bool) or not isinstance(copy, int):
        # Identify the original invalid component selector.
        raise TypeError(f"{operator} copy must be an integer. Got {copy!r}.")
    # A component index must address an existing gauge-copy block.
    if not 0 <= copy < int(hilbert.gauge_dimensions):
        # Report the valid inclusive copy range alongside the supplied value.
        raise ValueError(
            f"{operator} copy must be in [0, {hilbert.gauge_dimensions - 1}], got {copy}."
        )
    # Return the static component index used by site-layout helpers.
    return int(copy)


__all__ = [
    "require_gauge_dimensions",
    "require_half_ladder_graph",
    "require_nonplanar_graph",
    "require_planar_graph",
    "require_single_vertex_three_edges",
    "require_u1_hilbert",
    "validate_copy",
    "validate_site",
]
