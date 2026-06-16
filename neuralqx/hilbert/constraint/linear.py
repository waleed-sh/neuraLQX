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


"""Sparse linear constraints for discrete Hilbert spaces.

Linear constraints express one or more equalities over selected flat sites.
They are useful for small conservation laws and simple sum rules that can be
evaluated directly on JAX arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from neuralqx.utils.struct import field

from .abstract import AbstractDiscreteConstraint


class LinearConstraint(AbstractDiscreteConstraint):
    """Sparse equalities of the form ``sum(weights * state[sites]) == target``.

    Args:
        sites: Per-row site indices participating in each equality.
        weights: Per-row coefficients paired with ``sites``.
        targets: Target value for each equality row.
        atol: Absolute tolerance used for approximate comparison.

    Attributes:
        sites: Per-row site indices participating in each equality.
        weights: Per-row coefficients paired with ``sites``.
        targets: Target value for each equality row.
        atol: Absolute tolerance used for approximate comparison.
    """

    sites: tuple[tuple[int, ...], ...] | Sequence[Sequence[int]] = field(static=True)
    """Per-row site indices participating in each equality."""
    weights: tuple[tuple[int | float, ...], ...] | Sequence[Sequence[int | float]] = (
        field(static=True)
    )
    """Per-row coefficients paired with ``sites``."""
    targets: tuple[int | float, ...] | Sequence[int | float] = field(static=True)
    """Target value for each equality row."""
    atol: float = field(static=True, default=0.0)
    """Absolute tolerance used for approximate comparison."""

    def __post_init__(self) -> None:
        """Normalizes rows and validates shape consistency."""
        sites = tuple(tuple(int(site) for site in row) for row in self.sites)
        weights = tuple(tuple(weight for weight in row) for row in self.weights)
        targets = tuple(self.targets)
        if not sites:
            raise ValueError("LinearConstraint needs at least one row.")
        if len(sites) != len(weights) or len(sites) != len(targets):
            raise ValueError(
                "sites, weights, and targets must have the same row count."
            )
        for site_row, weight_row in zip(sites, weights, strict=True):
            if not site_row:
                raise ValueError("LinearConstraint rows may not be empty.")
            if len(site_row) != len(weight_row):
                raise ValueError("Every row needs matching sites and weights.")
            if any(site < 0 for site in site_row):
                raise ValueError("LinearConstraint sites must be non-negative.")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "targets", targets)

    def __hash__(self) -> int:
        """Returns a structural hash for the linear constraint."""
        return hash((type(self), self.sites, self.weights, self.targets, self.atol))

    @classmethod
    def sum(cls, sites: Sequence[int], target: int | float) -> LinearConstraint:
        """Builds a unit-weight sum constraint.

        Args:
            sites: Flat sites participating in the sum.
            target: Required sum value.

        Returns:
            Linear constraint with one row and unit weights.
        """
        return cls(
            sites=(tuple(sites),), weights=((1,) * len(sites),), targets=(target,)
        )

    def validate_hilbert(self, hilbert: Any) -> None:
        """Checks that all constrained sites exist in a Hilbert space.

        Args:
            hilbert: Hilbert-space instance exposing a flat ``size``.

        Raises:
            IndexError: If any referenced site is outside the flat state range.
        """
        size = hilbert.size
        for row in self.sites:
            if any(site >= size for site in row):
                raise IndexError(
                    f"Constraint site out of range for Hilbert size {size}."
                )

    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates all sparse equalities on a batch of states.

        Args:
            states: State array whose trailing dimension contains flat site
                values.

        Returns:
            Boolean mask where every equality row is satisfied.
        """
        arr = jnp.asarray(states)
        ok = jnp.ones(arr.shape[:-1], dtype=jnp.bool_)
        for site_row, weight_row, target in zip(
            self.sites,
            self.weights,
            self.targets,
            strict=True,
        ):
            sites = jnp.asarray(site_row, dtype=jnp.int32)
            weights = jnp.asarray(weight_row, dtype=arr.dtype)
            value = jnp.sum(jnp.take(arr, sites, axis=-1) * weights, axis=-1)
            target_arr = jnp.asarray(target, dtype=arr.dtype)
            if self.atol == 0.0:
                ok = ok & (value == target_arr)
            else:
                ok = ok & jnp.isclose(value, target_arr, atol=self.atol)
        return ok


__all__ = ["LinearConstraint"]
