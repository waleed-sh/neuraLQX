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


"""U(1) lattice arithmetic and flat-layout helpers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.hilbert.u1.spaces.abstract import AbstractU1Hilbert

from .validation import validate_copy


def q_bounds(
    hilbert: AbstractU1Hilbert,
) -> tuple[int | float, int | float, int | float, int]:
    r"""Expose the finite charge lattice used by the U(1) Hilbert space.

    The local values are $q_{\min}+nh$ for $n=0,\ldots,L-1$, where
    $h=q_{\mathrm{step}}$ and $L$ is the local size. Cyclic arithmetic wraps
    with period $Lh$, while truncated QR shifts use the inclusive end bounds.

    Args:
        hilbert: U(1) space defining the local charge range.

    Returns:
        result (tuple[int | float, int | float, int | float, int]): Minimum,
            maximum, lattice spacing, and number of allowed local values.
    """
    # Use Hilbert-space metadata so all kernels share exactly the same cutoff.
    return hilbert.q_min, hilbert.q_max, hilbert.q_step, int(hilbert.local_size)


def edge_index(hilbert: AbstractU1Hilbert, edge: Any) -> int:
    r"""Resolve an edge to its base index before applying a gauge-copy offset.

    Integers address `hilbert.tiny_size` base edges. Other inputs use the graph's
    edge lookup, which also resolves a reversed edge to its stored representative.
    The returned index therefore does not encode traversal direction. Holonomy
    callers must retain that sign separately in their loop or segment metadata.

    Args:
        hilbert: Space whose base graph defines edge indexing.
        edge: Base-edge integer index or an edge-like value accepted by the graph.

    Returns:
        result (int): Index in the first gauge-copy block.

    Raises:
        TypeError: If a boolean is supplied as an index.
        ValueError: If an integer falls outside the base-edge range.
    """
    # Booleans must not alias base-edge indices zero and one.
    if isinstance(edge, bool):
        # Reject the ambiguous input before integer dispatch.
        raise TypeError("Edge index must not be a bool.")
    # An integer names a base edge, not a site in a later gauge copy.
    if isinstance(edge, int):
        # Validate against the base graph size rather than the full state size.
        if not 0 <= edge < int(hilbert.tiny_size):
            # Include the valid inclusive range in the indexing error.
            raise ValueError(
                f"Edge index must be in [0, {hilbert.tiny_size - 1}], got {edge}."
            )
        # The validated index is already in the canonical base-edge block.
        return int(edge)
    # Let the graph resolve endpoint notation and keyed parallel edges.
    return int(hilbert.graph.edge_to_index(edge))


def canonical_edge(hilbert: AbstractU1Hilbert, edge: Any):
    r"""Return the stored graph representative of an edge.

    An edge supplied with reversed endpoints resolves to the same base edge.
    This helper deliberately returns stored orientation. It must not be used to
    infer a loop traversal sign after that information has been discarded.

    Args:
        hilbert: Space whose graph owns the edge.
        edge: Base-edge index or edge-like input.

    Returns:
        result (Edge): The graph edge at the resolved base index.
    """
    # Resolve through the graph index to retain the canonical key and orientation.
    return hilbert.graph.index_to_edge(edge_index(hilbert, edge))


def site_index(hilbert: AbstractU1Hilbert, edge: Any, copy: int = 0) -> int:
    r"""Locate one edge component in a flat, copy-major charge configuration.

    For $E$ base edges, component $c$ of edge $e$ is stored at $cE+e$.
    A base-edge index and a full state-site index are therefore interchangeable
    only in the first gauge copy.

    Args:
        hilbert: U(1) space defining the number of edges and copies.
        edge: Base-edge index or edge-like value.
        copy: Zero-based gauge-copy index.

    Returns:
        result (int): Flat state position of the selected component.

    Raises:
        TypeError: If `copy` is not an integer or is a boolean.
        ValueError: If the copy or base-edge index is outside its allowed range.
    """
    # Validate before arithmetic so fractional and boolean copies cannot be coerced.
    copy = validate_copy(hilbert, copy, operator="U(1) site lookup")
    # Offset the validated base edge by one complete edge block per gauge copy.
    return copy * int(hilbert.tiny_size) + edge_index(hilbert, edge)


def copy_sites(hilbert: AbstractU1Hilbert, edge: Any) -> tuple[int, ...]:
    r"""Locate every gauge component of a single geometric edge.

    The result follows component order. In the U(1)^3 models, gathering these
    three positions produces the charge vector used in area and volume formulas.

    Args:
        hilbert: U(1) space using the copy-major state layout.
        edge: Base-edge index or edge-like value.

    Returns:
        result (tuple[int, ...]): One flat site per gauge component.
    """
    # Resolve the geometric edge once before adding component offsets.
    base = edge_index(hilbert, edge)
    # Each gauge copy occupies one complete base-edge block.
    tiny = int(hilbert.tiny_size)
    # List the corresponding position in each block in gauge-component order.
    return tuple(base + copy * tiny for copy in range(int(hilbert.gauge_dimensions)))


def wrap_add_value(
    value: jax.Array,
    delta: int | float | jax.Array,
    *,
    q_min: int | float,
    q_step: int | float,
    local_size: int,
) -> jax.Array:
    r"""Add a charge displacement on the finite cyclic U(1) lattice.

    For $L$ allowed values separated by $h$, the result is

    $$
    q_{\min}+h\left[\left(\frac{q-q_{\min}}{h}
    +\frac{\Delta}{h}\right)\bmod L\right].
    $$

    The period is $Lh$, not $q_{\max}-q_{\min}$. Inputs and displacements must
    lie on this lattice. The helper performs no rounding or off-lattice
    validation and supports JAX tracing and array broadcasting.

    Args:
        value: Scalar or array of charges, with a dtype representing the lattice.
        delta: Physical charge displacement, not a count of lattice steps.
        q_min: Smallest allowed local charge.
        q_step: Positive separation between allowed charges.
        local_size: Number of allowed charges, including both endpoints.

    Returns:
        result (jax.Array): Wrapped charges, cast back to `value.dtype`.
    """
    # Represent the physical lattice spacing in the charge dtype.
    step = jnp.asarray(q_step, dtype=value.dtype)
    # The lattice is indexed relative to its lower endpoint.
    start = jnp.asarray(q_min, dtype=value.dtype)
    # The modular index range includes every allowed local state.
    size = jnp.asarray(local_size, dtype=value.dtype)
    # Convert the displacement without leaving the traced JAX computation.
    delta_arr = jnp.asarray(delta, dtype=value.dtype)
    # Translate physical charge to its zero-based lattice index.
    value_index = (value - start) / step
    # Express the physical displacement in the same index units.
    delta_index = delta_arr / step
    # Wrap the index, restore charge units, and preserve the input dtype.
    return (jnp.mod(value_index + delta_index, size) * step + start).astype(value.dtype)


def shift_sites(
    state: jax.Array,
    sites: tuple[int, ...],
    directions: tuple[int, ...],
    *,
    q_min: int | float,
    q_step: int | float,
    local_size: int,
    scale: int = 1,
) -> jax.Array:
    r"""Apply a sequence of cyclic component shifts to one configuration.

    Each pair `(site, direction)` adds `scale * direction` lattice steps.
    Repeated sites are updated in sequence, so repeated traversal of an edge
    accumulates its signed winding. The input is immutable. An empty site list
    returns the unchanged state. This routine only moves labels and does not
    choose whether those labels describe ket action or matrix-row targets.

    Args:
        state: Flat configuration of shape `(hilbert.size,)`.
        sites: Flat indices to update, in traversal order.
        directions: Signed integer step counts, one per listed site.
        q_min: Smallest allowed charge.
        q_step: Physical charge increment for one lattice step.
        local_size: Number of allowed charges at one site.
        scale: Common integer multiplier, usually +1 or -1 for a loop or its inverse.

    Returns:
        result (jax.Array): Updated configuration with the input shape and dtype.

    Raises:
        ValueError: If the site and direction sequences have different lengths.
    """
    # JAX updates return new arrays, leaving the caller's state unchanged.
    out = state
    # Convert signed step counts to physical charge increments.
    unit = jnp.asarray(q_step, dtype=out.dtype)
    # Pair each traversal with its direction and reject inconsistent metadata.
    for site, direction in zip(sites, directions, strict=True):
        # Update sequentially so repeated sites accumulate rather than overwrite shifts.
        out = out.at[site].set(
            # Use the common cyclic rule at both charge cutoffs.
            wrap_add_value(
                out[site],
                # The branch multiplier reverses all loop steps when the inverse is requested.
                int(scale) * int(direction) * unit,
                q_min=q_min,
                q_step=q_step,
                local_size=local_size,
            )
        )
    # Return the configuration after the entire ordered shift sequence.
    return out


def noncyclic_shift_site(
    state: jax.Array,
    site: int,
    delta: int | float | jax.Array,
    *,
    q_min: int | float,
    q_max: int | float,
) -> tuple[jax.Array, jax.Array]:
    r"""Propose a bounded QR shift and report whether its target exists.

    Unlike cyclic U(1) holonomies, a truncated QR shift vanishes at the boundary.
    This helper keeps the original label for an invalid proposal, ensuring that
    padding is still a valid configuration. The caller must multiply its matrix
    element by the returned mask. The helper checks bounds, not lattice spacing.

    Args:
        state: Flat input configuration.
        site: Component to shift.
        delta: Physical charge displacement on the local lattice.
        q_min: Inclusive lower charge bound.
        q_max: Inclusive upper charge bound.

    Returns:
        result (tuple[jax.Array, jax.Array]): Proposed configuration and scalar
            boolean validity mask. Invalid proposals leave the input unchanged.
    """
    # Read the original charge for both the proposal and invalid-target fallback.
    value = state[site]
    # Apply the physical displacement without wrapping at the cutoff.
    new_value = value + jnp.asarray(delta, dtype=value.dtype)
    # Both inclusive bounds must hold for a nonzero matrix element.
    valid = (new_value >= jnp.asarray(q_min, dtype=value.dtype)) & (
        new_value <= jnp.asarray(q_max, dtype=value.dtype)
    )
    # Keep a legal padding state and return the mask needed to zero its amplitude.
    return state.at[site].set(jnp.where(valid, new_value, value)), valid


__all__ = [
    "canonical_edge",
    "copy_sites",
    "edge_index",
    "noncyclic_shift_site",
    "q_bounds",
    "shift_sites",
    "site_index",
    "wrap_add_value",
]
