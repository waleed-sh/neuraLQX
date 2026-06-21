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


"""Plum-dispatched random-state generation.

This module provides the generic ``random_state`` entry point for discrete
Hilbert spaces. Unconstrained spaces sample directly from local domains,
specialized U(1) and SU(2) spaces delegate to their own samplers, and generic
constrained spaces fall back to bounded rejection sampling.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from plum import dispatch

from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.constraint import IdentityConstraint
from neuralqx.hilbert.space import DiscreteHilbertSpace
from neuralqx.hilbert.space import HeterogeneousDiscreteHilbert
from neuralqx.hilbert.space import HomogeneousDiscreteHilbert
from neuralqx.hilbert.su2.spaces import SU2GaugeInvariantHilbert
from neuralqx.hilbert.u1.spaces import U1GaugeInvariantHilbert
from neuralqx.hilbert.u1.spaces import U1Hilbert
from neuralqx.hilbert.utils import batch_shape


def random_state(
    space: DiscreteHilbertSpace,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Generates random state arrays for a discrete Hilbert space.

    Args:
        space: Discrete Hilbert space to sample from.
        key: JAX pseudo-random key.
        size: Optional leading batch shape for generated states.
        dtype: Optional dtype for returned state values.
        max_trials: Maximum attempts for generic rejection sampling.

    Returns:
        Random state array with trailing dimension ``space.size``.
    """
    return _nqx_random_state(
        space, space.constraint, key, size, dtype=dtype, max_trials=max_trials
    )


@dispatch
def _nqx_random_state(
    space: HomogeneousDiscreteHilbert,
    constraint: IdentityConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Samples an unconstrained homogeneous space directly from local domains."""
    del constraint, max_trials
    return _random_unconstrained_jit(space, key, size, dtype)


@dispatch
def _nqx_random_state(
    space: HeterogeneousDiscreteHilbert,
    constraint: IdentityConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Samples an unconstrained heterogeneous space directly from local domains."""
    del constraint, max_trials
    return _random_unconstrained_jit(space, key, size, dtype)


@dispatch
def _nqx_random_state(
    space: U1Hilbert,
    constraint: IdentityConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Delegates unconstrained U(1) sampling to the U(1) Hilbert implementation."""
    del constraint, max_trials
    return space.random_state(key, size=size, dtype=dtype)


@dispatch
def _nqx_random_state(
    space: SU2GaugeInvariantHilbert,
    constraint: AbstractDiscreteConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Delegates SU(2) gauge-invariant sampling to the SU(2) implementation."""
    del constraint
    return space.random_state(key, size=size, dtype=dtype, max_trials=max_trials)


@dispatch
def _nqx_random_state(
    space: U1GaugeInvariantHilbert,
    constraint: AbstractDiscreteConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Delegates U(1) gauge-invariant sampling to the U(1) implementation."""
    del constraint, max_trials
    return space.random_state(key, size=size, dtype=dtype)


@dispatch
def _nqx_random_state(
    space: DiscreteHilbertSpace,
    constraint: AbstractDiscreteConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None = None,
    *,
    dtype: Any | None = None,
    max_trials: int = 1024,
) -> jax.Array:
    """Samples a generic constrained discrete space by rejection.

    Raises:
        RuntimeError: If any requested state is not accepted within
            ``max_trials`` attempts.
    """
    states, accepted = _random_rejection_jit(
        space, constraint, key, size, dtype, max_trials
    )
    if not bool(jnp.all(accepted)):
        raise RuntimeError(
            "Failed to sample constrained Hilbert state by rejection within "
            f"{max_trials} trials. Register a specialized random_state dispatch "
            "for this constraint."
        )
    return states


@partial(jax.jit, static_argnames=("space", "size", "dtype"))
def _random_unconstrained_jit(
    space: DiscreteHilbertSpace,
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: Any | None,
) -> jax.Array:
    """JIT wrapper for direct unconstrained sampling."""
    return _random_unconstrained(space, key, size, dtype)


@partial(
    jax.jit, static_argnames=("space", "constraint", "size", "dtype", "max_trials")
)
def _random_rejection_jit(
    space: DiscreteHilbertSpace,
    constraint: AbstractDiscreteConstraint,
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: Any | None,
    max_trials: int,
) -> tuple[jax.Array, jax.Array]:
    """JIT rejection sampler for generic constrained spaces.

    Args:
        space: Discrete Hilbert space whose local domains are sampled.
        constraint: Constraint used to accept or reject proposed states.
        key: JAX pseudo-random key.
        size: Optional leading batch shape.
        dtype: Optional dtype for generated state values.
        max_trials: Maximum number of rejection iterations.

    Returns:
        Pair containing sampled states and a boolean acceptance mask.
    """
    key, subkey = jax.random.split(key)
    states = _random_unconstrained(space, subkey, size, dtype)
    accepted = constraint(states)

    def cond(carry: tuple[int, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        """Returns whether rejection sampling should continue."""
        trial, _, _, ok = carry
        return (trial < max_trials) & ~jnp.all(ok)

    def body(carry: tuple[int, jax.Array, jax.Array, jax.Array]):
        """Performs one rejection-sampling iteration."""
        trial, loop_key, current, ok = carry
        loop_key, proposal_key = jax.random.split(loop_key)
        proposal = _random_unconstrained(space, proposal_key, size, dtype)
        proposal_ok = constraint(proposal)
        replace = ~ok & proposal_ok
        current = jnp.where(replace[..., None], proposal, current)
        return trial + 1, loop_key, current, ok | proposal_ok

    _, _, states, accepted = jax.lax.while_loop(cond, body, (0, key, states, accepted))
    return states, accepted


def _random_unconstrained(
    space: DiscreteHilbertSpace,
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: Any | None,
) -> jax.Array:
    """Samples an unconstrained generic discrete Hilbert space.

    Args:
        space: Homogeneous or heterogeneous discrete Hilbert space.
        key: JAX pseudo-random key.
        size: Optional leading batch shape.
        dtype: Optional dtype for generated state values.

    Returns:
        Random state array with trailing dimension ``space.size``.
    """
    if isinstance(space, HomogeneousDiscreteHilbert):
        unique, groups = _group_local_spaces(space.local_spaces)
        return _random_from_groups(
            local_spaces=unique,
            groups=groups,
            key=key,
            size=size,
            dtype=space.dtype if dtype is None else np.dtype(dtype),
        )

    if isinstance(space, HeterogeneousDiscreteHilbert):
        return _random_heterogeneous(space, key, size, dtype)

    raise TypeError(f"Unsupported discrete Hilbert type {type(space).__name__}.")


def _random_heterogeneous(
    space: HeterogeneousDiscreteHilbert,
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: Any | None,
) -> jax.Array:
    """Samples a heterogeneous space from grouped local domains.

    Args:
        space: Heterogeneous discrete Hilbert space.
        key: JAX pseudo-random key.
        size: Optional leading batch shape.
        dtype: Optional dtype for generated state values.

    Returns:
        Random state array with trailing dimension ``space.size``.
    """
    out_dtype = space.dtype if dtype is None else np.dtype(dtype)
    return _random_from_groups(
        local_spaces=space.unique_local_spaces,
        groups=space.groups,
        key=key,
        size=size,
        dtype=out_dtype,
    )


def _random_from_groups(
    *,
    local_spaces: tuple[Any, ...],
    groups: tuple[tuple[int, ...], ...],
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: np.dtype,
) -> jax.Array:
    """Samples flat sites grouped by common local domain.

    Args:
        local_spaces: Unique local spaces to sample.
        groups: Flat site groups for each unique local space.
        key: JAX pseudo-random key.
        size: Optional leading batch shape.
        dtype: Dtype for generated state values.

    Returns:
        Random state array assembled in flat site order.
    """
    keys = jax.random.split(key, len(local_spaces))
    n_sites = sum(len(group) for group in groups)
    out = jnp.empty((*batch_shape(size), n_sites), dtype=dtype)
    for subkey, local_space, sites in zip(keys, local_spaces, groups, strict=True):
        site_idx = jnp.asarray(sites, dtype=jnp.int32)
        values = local_space.random_values(
            subkey,
            (*batch_shape(size), len(sites)),
            dtype=dtype,
        )
        out = out.at[..., site_idx].set(values)
    return out


def _group_local_spaces(
    spaces: tuple[Any, ...],
) -> tuple[tuple[Any, ...], tuple[tuple[int, ...], ...]]:
    """Groups flat sites by equal local-space object.

    Args:
        spaces: Local space for every flat site.

    Returns:
        Unique local spaces and matching flat site groups.
    """
    unique: list[Any] = []
    groups: list[list[int]] = []
    for site, local_space in enumerate(spaces):
        try:
            idx = unique.index(local_space)
        except ValueError:
            unique.append(local_space)
            groups.append([])
            idx = len(unique) - 1
        groups[idx].append(site)
    return tuple(unique), tuple(tuple(group) for group in groups)


__all__ = ["random_state"]
