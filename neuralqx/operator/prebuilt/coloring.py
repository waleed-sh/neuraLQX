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


r"""Charge-label projectors and their products on U(1)^N Hilbert spaces.

``ChargeColoringOperator`` selects one flat component of a configuration.
``ColoringOperator`` selects every gauge component of one base edge. Their
expectation values give marginal probabilities under the normalized
distribution proportional to :math:`|\psi(\sigma)|^2`. Products select joint
events and retain correlations between the specified edges.

The operators use the same definitions as upstream neuraLQX's computational
coloring operators. All kernels are JAX-native and preserve the input charge
dtype. Construction validates the complete local charge range, including its
spacing and any nonzero starting value.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import copy_sites
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import validate_site
from neuralqx.operator.wrappers import ProductOperator
from neuralqx.utils.struct import field

from .wcl.identity import IdentityOperator


def _canonical_charge(hilbert, charge, *, operator: str) -> tuple[int | float, int]:
    """Return a validated stored charge label and its local basis index."""
    value = np.asarray(charge)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise TypeError(f"{operator} charge must be a real scalar, got {charge!r}.")
    if not np.isfinite(value):
        raise ValueError(f"{operator} charge must be finite, got {charge!r}.")
    raw_index = (float(value) - hilbert.q_min) / hilbert.q_step
    if not np.isfinite(raw_index):
        raise ValueError(
            f"{operator} charge {charge!r} is outside the allowed charge lattice."
        )
    index = round(raw_index)
    charge_dtype = np.dtype(hilbert.dtype)
    precision = max(
        np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) else 0.0
        for dtype in (value.dtype, charge_dtype, np.dtype(np.float64))
    )
    # Permit representational roundoff while still selecting one basis label.
    # Off-lattice charges are never truncated to integer charges.
    on_lattice = abs(raw_index - index) <= 8 * precision * max(1, abs(raw_index))
    if not 0 <= index < hilbert.local_size or not on_lattice:
        raise ValueError(
            f"{operator} charge {charge!r} is outside the allowed lattice "
            f"{hilbert.q_min} + k * {hilbert.q_step}, with 0 <= k < {hilbert.local_size}."
        )
    start = np.asarray(hilbert.q_min, dtype=charge_dtype)
    step = np.asarray(hilbert.q_step, dtype=charge_dtype)
    return (start + np.asarray(index, dtype=charge_dtype) * step).item(), index


@jax.jit
def _coloring_connections(states, sites, indices, start, step):
    """Return unchanged configurations and one indicator per batch row."""
    values = jnp.take(states, sites, axis=-1)
    if jnp.issubdtype(values.dtype, jnp.integer) and jnp.issubdtype(
        step.dtype, jnp.integer
    ):
        # Integer arithmetic also distinguishes labels beyond float32's exact range.
        matches = ((values - start) // step == indices) & ((values - start) % step == 0)
    else:
        # Configurations belong to the Hilbert lattice. Compare their discrete
        # indices so fused and unfused charge arithmetic give the same projector.
        matches = jnp.rint((values - start) / step) == indices
    matches = jnp.all(matches, axis=-1)
    return states[:, None, :], matches.astype(jnp.float64)[:, None]


class _ColoringProjector(ComputationalOperator):
    """Shared diagonal action for a conjunction of charge-label conditions.

    Subclasses resolve their public selectors into matching flat-site and
    local-index tuples during construction. Each padded row then contains the
    original configuration with amplitude zero or one.
    """

    _sites: tuple[int, ...] = field(static=True, init=False, default=(), repr=False)
    """Flat component indices whose labels must match simultaneously."""

    _indices: tuple[int, ...] = field(static=True, init=False, default=(), repr=False)
    """Local basis indices of the selected charges, in the same order as `_sites`."""

    @property
    def max_conn_size(self) -> int:
        """One diagonal slot, including a zero-amplitude slot for a nonmatch."""
        return 1

    @property
    def dtype(self) -> jnp.dtype:
        """Real float64 dtype of the zero-or-one matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """The real diagonal projector equals its conjugate transpose."""
        return True

    @property
    def is_diagonal(self) -> bool:
        """The operator leaves every basis configuration unchanged."""
        return True

    @property
    def adjoint(self) -> _ColoringProjector:
        r"""Return this projector, since :math:`P^\dagger=P`."""
        return self

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the original state and its scalar indicator in one padded slot."""
        targets, values = self._get_conn_padded_batch_kernel(state[None, :])
        return targets[0], values[0]

    def _get_conn_padded_batch_kernel(
        self, states_2d: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate a rank-two configuration batch without changing its charge dtype."""
        return _coloring_connections(
            states_2d,
            jnp.asarray(self._sites, dtype=jnp.int32),
            jnp.asarray(self._indices, dtype=jnp.int32),
            jnp.asarray(self.hilbert.q_min, dtype=self.hilbert.dtype),
            jnp.asarray(self.hilbert.q_step, dtype=self.hilbert.dtype),
        )

    def _get_conn_padded_direct(
        self, states_2d: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Reuse compiled indicator kernels across site and charge selections."""
        return self._get_conn_padded_batch_kernel(states_2d)


class ChargeColoringOperator(_ColoringProjector):
    r"""Project onto a specified charge at one flat U(1) component.

    For flat site :math:`s` and charge :math:`q`, the operator acts as

    .. math::

        P_{s,q}|\sigma\rangle=\delta_{\sigma_s,q}|\sigma\rangle,
        \qquad P_{s,q}^2=P_{s,q}=P_{s,q}^\dagger.

    Its expectation value is the probability of finding that charge at the
    selected component. Other components are unrestricted. In a U(1)^N space
    with :math:`E` base edges, component :math:`c` of edge :math:`e` has flat
    index :math:`s=cE+e`.

    Both unconstrained and gauge-invariant U(1) Hilbert spaces are supported.
    The projector restricts the existing state space and preserves its gauge
    conditions. Charges may be fractional when the Hilbert space permits
    them. Floating-point roundoff is normalized to the stored lattice label
    during construction. The kernel compares local basis indices, so roundoff
    in floating-point configurations does not change the selected basis label.

    Args:
        hilbert: U(1)^N Hilbert space carrying the charge configurations.
        site: Python integer in ``[0, hilbert.size)``. This is a flat component
            index and may select any gauge copy.
        charge: Finite real label belonging to the Hilbert space's local range.

    Raises:
        TypeError: If the Hilbert space, site type, or charge type is invalid.
        ValueError: If the site is out of bounds or the charge is not allowed.

    Examples:
        Select charge one on edge zero in the second gauge copy::

            projector = ChargeColoringOperator(hilbert, hilbert.tiny_size, 1)
            probability = state.expect(projector)
    """

    site: int = field(static=True)
    """Flat component index, counting all edge blocks of the gauge copies."""

    charge: int | float = field(static=True)
    """Selected local charge, stored in the Hilbert space's numeric representation."""

    def __post_init__(self) -> None:
        """Validate the flat site and charge before constructing their indicator."""
        name = type(self).__name__
        hilbert = require_u1_hilbert(self.hilbert, operator=name)
        site = validate_site(hilbert, self.site, operator=name)
        charge, index = _canonical_charge(hilbert, self.charge, operator=name)
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "_sites", (site,))
        object.__setattr__(self, "_indices", (index,))
        super().__post_init__()

    def __repr__(self) -> str:
        """Show the selected flat component and its charge label."""
        return f"ChargeColoringOperator(site={self.site}, charge={self.charge!r})"


class ColoringOperator(_ColoringProjector):
    r"""Project onto a complete charge vector on one U(1)^N graph edge.

    For base edge :math:`e` and requested labels :math:`\mathbf q=(q_0,\ldots,q_{N-1})`,

    .. math::

        P_{e,\mathbf q}|\sigma\rangle
        =\left[\prod_{c=0}^{N-1}\delta_{\sigma_{cE+e},q_c}\right]|\sigma\rangle.

    Here :math:`E` is the number of base edges. All specified components must
    match in the same configuration. The expectation therefore gives the
    joint probability of the edge's charge vector, including correlations
    between its gauge components. Every positive number of gauge copies is
    supported, including a single U(1) copy.

    This is a real orthogonal projector with :math:`P^2=P=P^\dagger`. It is
    diagonal on both unconstrained and gauge-invariant Hilbert spaces. Its
    kernel returns one unchanged configuration with matrix element one for
    a match and zero otherwise. No basis enumeration or sparse matrix
    construction is needed.

    Args:
        hilbert: U(1)^N Hilbert space with the component-major layout.
        site: Python integer naming an edge in the base graph, in
            ``[0, hilbert.tiny_size)``. Gauge-copy offsets are supplied by
            the operator.
        charge_vector: Sequence of exactly ``hilbert.gauge_dimensions``
            allowed local labels, in gauge-copy order. Fractional labels
            are supported and normalized as in `ChargeColoringOperator`.

    Raises:
        TypeError: If the Hilbert space, site type, or charge types are invalid.
        ValueError: If the base edge, vector length, or any charge is invalid.

    Examples:
        Select an edge coloring and a joint event on two edges::

            first = ColoringOperator(hilbert, 0, (1, 0, -1))
            second = ColoringOperator(hilbert, 2, (0, 1, 0))
            probability = state.expect(first)
            joint_probability = state.expect(first @ second)
    """

    site: int = field(static=True)
    """Base-edge index, shared by all components of the selected charge vector."""

    charge_vector: tuple[int | float, ...] = field(static=True, converter=tuple)
    """Allowed charge labels ordered by gauge copy and normalized at construction."""

    def __post_init__(self) -> None:
        """Resolve the base edge and validate one charge for each gauge copy."""
        name = type(self).__name__
        hilbert = require_u1_hilbert(self.hilbert, operator=name)
        site = validate_site(hilbert, self.site, operator=name)
        if site >= hilbert.tiny_size:
            raise ValueError(
                f"{name} site must name a base edge in [0, {hilbert.tiny_size}), got {site}."
            )
        if len(self.charge_vector) != hilbert.gauge_dimensions:
            raise ValueError(
                f"{name} requires {hilbert.gauge_dimensions} charges, got {len(self.charge_vector)}."
            )
        selections = tuple(
            _canonical_charge(hilbert, charge, operator=name)
            for charge in self.charge_vector
        )
        charges, indices = zip(*selections, strict=True)
        object.__setattr__(self, "charge_vector", charges)
        object.__setattr__(self, "_sites", copy_sites(hilbert, site))
        object.__setattr__(self, "_indices", indices)
        super().__post_init__()

    def __repr__(self) -> str:
        """Show the base edge and its complete charge-vector selection."""
        return (
            f"ColoringOperator(site={self.site}, charge_vector={self.charge_vector!r})"
        )


charge_coloring = ChargeColoringOperator
"""Constructor alias retaining upstream's single-component factory name."""

coloring = ColoringOperator
"""Constructor alias retaining upstream's edge-coloring factory name."""


def n_point_function(
    hilbert,
    operator: Callable[..., ComputationalOperator],
    edges: Sequence[int],
    charge_vectors: Sequence[Any],
) -> ComputationalOperator:
    r"""Build a product of local coloring selections for a joint expectation.

    Each factor is constructed as ``operator(hilbert, edge, charge_vector)``.
    For `ColoringOperator`, evaluating the result with ``state.expect`` gives

    .. math::

        \left\langle\prod_{i=0}^{n-1}P_{e_i,\mathbf q_i}\right\rangle
        =\Pr_\psi\!\left(\sigma_{e_i}=\mathbf q_i\text{ for every }i\right).

    The supplied edge selections may repeat. Identical repeated projectors
    have no additional effect, and incompatible labels on the same edge
    give the zero operator. An empty selection gives the identity.

    This returns an operator, leaving sampling and expectation evaluation to
    the variational state. To preserve upstream's convention for a general
    factory, factors are ordered as :math:`O_{n-1}\cdots O_1O_0`. Coloring
    projectors commute, so this ordering does not affect their joint event.
    Explicit products with ``@`` can be used when a different order is needed.

    Args:
        hilbert: Hilbert space passed to every local factory call.
        operator: Callable such as `ColoringOperator` or `ChargeColoringOperator`.
        edges: Base-edge indices for `ColoringOperator`, or flat-site indices
            for `ChargeColoringOperator`.
        charge_vectors: One vector per edge for `ColoringOperator`. Supply
            scalar charges instead when using `ChargeColoringOperator`.

    Returns:
        ComputationalOperator: The local factor, a lazy product of factors,
            or the identity for an empty selection.

    Raises:
        ValueError: If the selection lengths differ or factors act on a
            different Hilbert space. Local factories also validate labels.
        TypeError: If `operator` is not callable or returns a noncomputational
            operator.

    Examples:
        Estimate the probability of two specified edge colorings::

            joint = n_point_function(hilbert, ColoringOperator, (0, 2), ((1, 0, -1), (0, 1, 0)))
            probability = state.expect(joint)
    """
    if not callable(operator):
        raise TypeError("n_point_function requires an operator factory.")
    edges, charge_vectors = tuple(edges), tuple(charge_vectors)
    if len(edges) != len(charge_vectors):
        raise ValueError(
            "Each edge must have exactly one corresponding charge selection."
        )
    factors = tuple(
        operator(hilbert, edge, charges)
        for edge, charges in zip(edges, charge_vectors, strict=True)
    )
    for factor in factors:
        if not isinstance(factor, ComputationalOperator):
            raise TypeError("The local factory must return a ComputationalOperator.")
        if factor.hilbert != hilbert:
            raise ValueError(
                "Every local factor must act on the supplied Hilbert space."
            )
    if not factors:
        return IdentityOperator(hilbert)
    if len(factors) == 1:
        return factors[0]
    return ProductOperator(tuple(reversed(factors)))


__all__ = [
    "ChargeColoringOperator",
    "ColoringOperator",
    "charge_coloring",
    "coloring",
    "n_point_function",
]
