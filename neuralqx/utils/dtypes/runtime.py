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


"""
Runtime entry points for effective dtype policy and JAX array initialisation.

This module exposes two layers of helpers:

Dtype accessors: return the effective :class:`jnp.dtype` for each role
defined in :class:`~neuralqx.utils.dtypes.policy.DTypePolicy`::

    jax_real_dtype()  # e.g. jnp.float64
    jax_complex_dtype()  # e.g. jnp.complex128
    jax_index_dtype()  # e.g. jnp.int64

Array producers: thin wrappers around :mod:`jax.numpy` that pre-fill the
``dtype`` argument from the active policy so call sites don't have to repeat
it::

    zeros_real((3, 4))  # jnp.zeros((3, 4), dtype=jax_real_dtype())
    ones_complex((2,))  # jnp.ones((2,), dtype=jax_complex_dtype())
    array_real([1.0, 2.0])  # jnp.array([1., 2.], dtype=jax_real_dtype())

Each also accepts an optional ``config_manager`` keyword so callers
operating under a custom config override can propagate it without touching
global state.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neuralqx.utils.dtypes.policy import DTypePolicy


def get_dtype_policy(config_manager: Any | None = None) -> DTypePolicy:
    """
    Returns the effective dtype policy.

    Args:
        config_manager: Optional config manager. If omitted, uses the global
            :data:`neuralqx.configs.cfg` singleton.

    Returns:
        Effective :class:`~neuralqx.utils.dtypes.policy.DTypePolicy`.
    """
    return DTypePolicy.from_config(config_manager=config_manager)


def dtype_name_map(config_manager: Any | None = None) -> dict[str, str]:
    """
    Returns effective dtype names as a plain dictionary.

    Args:
        config_manager: Optional config manager.

    Returns:
        Mapping with keys ``real``, ``complex``, ``index``.
    """
    return get_dtype_policy(config_manager=config_manager).as_dict()


def jax_real_dtype(config_manager: Any | None = None) -> jnp.dtype:
    """Returns the effective JAX real dtype."""
    return jnp.dtype(get_dtype_policy(config_manager=config_manager).real)


def jax_complex_dtype(config_manager: Any | None = None) -> jnp.dtype:
    """Returns the effective JAX complex dtype."""
    return jnp.dtype(get_dtype_policy(config_manager=config_manager).complex)


def jax_index_dtype(config_manager: Any | None = None) -> jnp.dtype:
    """Returns the effective JAX index dtype."""
    return jnp.dtype(get_dtype_policy(config_manager=config_manager).index)


def zeros_real(
    shape: int | tuple[int, ...],
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns a zero-filled real array with the effective real dtype."""
    return jnp.zeros(shape, dtype=jax_real_dtype(config_manager))


def ones_real(
    shape: int | tuple[int, ...],
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns a one-filled real array with the effective real dtype."""
    return jnp.ones(shape, dtype=jax_real_dtype(config_manager))


def full_real(
    shape: int | tuple[int, ...],
    fill: float,
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns an array filled with *fill* using the effective real dtype."""
    return jnp.full(shape, fill, dtype=jax_real_dtype(config_manager))


def array_real(
    values: Any,
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Converts *values* to a JAX array with the effective real dtype."""
    return jnp.array(values, dtype=jax_real_dtype(config_manager))


def zeros_complex(
    shape: int | tuple[int, ...],
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns a zero-filled complex array with the effective complex dtype."""
    return jnp.zeros(shape, dtype=jax_complex_dtype(config_manager))


def ones_complex(
    shape: int | tuple[int, ...],
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns a one-filled complex array with the effective complex dtype."""
    return jnp.ones(shape, dtype=jax_complex_dtype(config_manager))


def full_complex(
    shape: int | tuple[int, ...],
    fill: complex,
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns an array filled with *fill* using the effective complex dtype."""
    return jnp.full(shape, fill, dtype=jax_complex_dtype(config_manager))


def array_complex(
    values: Any,
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Converts *values* to a JAX array with the effective complex dtype."""
    return jnp.array(values, dtype=jax_complex_dtype(config_manager))


def zeros_index(
    shape: int | tuple[int, ...],
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Returns a zero-filled integer array with the effective index dtype."""
    return jnp.zeros(shape, dtype=jax_index_dtype(config_manager))


def array_index(
    values: Any,
    *,
    config_manager: Any | None = None,
) -> jnp.ndarray:
    """Converts *values* to a JAX array with the effective index dtype."""
    return jnp.array(values, dtype=jax_index_dtype(config_manager))


__all__ = [
    "array_complex",
    "array_index",
    "array_real",
    "dtype_name_map",
    "full_complex",
    "full_real",
    "get_dtype_policy",
    "jax_complex_dtype",
    "jax_index_dtype",
    "jax_real_dtype",
    "ones_complex",
    "ones_real",
    "zeros_complex",
    "zeros_index",
    "zeros_real",
]
