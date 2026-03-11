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
NetKet ordering and local-basis conversion utilities for U(1) Hilbert spaces.

This module is the canonical home for U(1)-specific NetKet conversion helpers
used by :mod:`neuralqx.hilbert.u1.index.mapping`.
"""

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

import netket as nk
from netket.utils import StaticRange

from .utils._array_normalise import _as_numpy
from .utils._types import Order


def _infer_netket_site_order(local_states: Any) -> Order:
    r"""Infer NetKet's site-significance convention (``"C"`` vs ``"F"``)."""

    if local_states.length > 1000:
        local_states = StaticRange(
            start=local_states.start, step=local_states.step, length=100
        )

    vals = _as_numpy(local_states.all_states())
    L = int(vals.shape[0])
    if L <= 1:
        return "C"

    h2 = nk.hilbert.HomogeneousHilbert(local_states=local_states, N=2)

    s0 = vals[0]
    s1 = vals[1]

    st01 = jnp.asarray([[s0, s1]])
    st10 = jnp.asarray([[s1, s0]])

    try:
        n01 = jnp.asarray(h2.states_to_numbers(st01))[0]
        n10 = jnp.asarray(h2.states_to_numbers(st10))[0]
    except Exception as e:
        raise RuntimeError(
            "Could not infer NetKet ordering: calling states_to_numbers on a tiny (N=2) "
            "HomogeneousHilbert failed: {e}".format(e=e)
        ) from e

    L_j = jnp.asarray(L, dtype=n10.dtype)

    cond_C = (n10 == L_j) & (n01 == jnp.asarray(1, dtype=n10.dtype))
    cond_F = (n10 == jnp.asarray(1, dtype=n10.dtype)) & (n01 == L_j)

    code = jnp.select(
        [cond_C, cond_F], [jnp.int8(0), jnp.int8(1)], default=jnp.int8(-1)
    )

    try:
        code_i = int(np.asarray(code))
    except Exception:
        return code  # type: ignore[return-value]

    if code_i == 0:
        return "C"
    if code_i == 1:
        return "F"

    raise RuntimeError(
        f"Unexpected NetKet numbering strides for N=2: num([1,0])={np.asarray(n10)}, "
        f"num([0,1])={np.asarray(n01)}, L={L}. Cannot determine a consistent C/F ordering."
    )


def _local_values_to_digits(local_states: Any, values2d: np.ndarray) -> np.ndarray:
    """Convert physical local values to integer digits compatible with NetKet."""
    flat = values2d.reshape((-1,))
    try:
        digits_flat = _as_numpy(local_states.states_to_numbers(flat))
    except Exception as e:
        raise ValueError("Input contains values not in the local basis.") from e
    return _as_numpy(digits_flat).reshape(values2d.shape)


def _local_digits_to_values(local_states: Any, digits2d: np.ndarray) -> np.ndarray:
    """Convert integer digits to physical local values using NetKet's local basis mapping."""
    flat = digits2d.reshape((-1,))
    try:
        vals_flat = _as_numpy(local_states.numbers_to_states(flat))
    except Exception as e:
        raise ValueError("Input contains invalid local digits.") from e
    return _as_numpy(vals_flat).reshape(digits2d.shape)


def _local_values_to_digits_jax(local_states: Any, values2d: jax.Array) -> jax.Array:
    """Map local physical values to digits [0, L) — JAX path."""
    if (
        hasattr(local_states, "start")
        and hasattr(local_states, "step")
        and hasattr(local_states, "length")
    ):
        start = jnp.asarray(local_states.start, dtype=values2d.dtype)
        step = jnp.asarray(local_states.step, dtype=values2d.dtype)
        return jnp.rint((values2d - start) / step).astype(jnp.int64)

    vals = jnp.asarray(local_states.all_states())
    matches = values2d[..., jnp.newaxis] == vals
    return jnp.argmax(matches, axis=-1).astype(jnp.int64)


def _local_digits_to_values_jax(local_states: Any, digits2d: jax.Array) -> jax.Array:
    """Map digits [0, L) to local physical values — JAX path."""
    if (
        hasattr(local_states, "start")
        and hasattr(local_states, "step")
        and hasattr(local_states, "length")
    ):
        start = jnp.asarray(local_states.start)
        step = jnp.asarray(local_states.step)
        return (start + step * digits2d).astype(start.dtype)

    vals = jnp.asarray(local_states.all_states())
    return jnp.take(vals, digits2d, axis=0)
