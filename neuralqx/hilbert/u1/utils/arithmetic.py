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


"""Cyclic arithmetic for finite U(1) quantum-number ranges.

The U(1) Hilbert spaces represent charges on a finite arithmetic range. These
helpers perform addition and signed sums in that range and wrap results back
onto the configured local domain.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def wrap_values(
    values: Any,
    *,
    q_min: int | float,
    step: int | float,
    local_size: int,
    dtype: Any | None = None,
) -> jax.Array:
    """Wraps values onto ``q_min + step * n`` modulo ``local_size``.

    Args:
        values: Candidate charge values.
        q_min: Minimum charge value in the finite range.
        step: Charge spacing.
        local_size: Number of allowed charge values.
        dtype: Optional dtype for the wrapped result.

    Returns:
        Values wrapped back onto the finite U(1) local range.
    """
    out_dtype = jnp.asarray(q_min).dtype if dtype is None else dtype
    vals = jnp.asarray(values, dtype=out_dtype)
    q0 = jnp.asarray(q_min, dtype=out_dtype)
    qstep = jnp.asarray(step, dtype=out_dtype)
    raw = (vals - q0) / qstep
    idx = jnp.rint(raw).astype(jnp.int32)
    wrapped = jnp.mod(idx, int(local_size))
    return (q0 + wrapped.astype(out_dtype) * qstep).astype(out_dtype)


def modular_add(
    lhs: Any,
    rhs: Any,
    *,
    q_min: int | float,
    step: int | float,
    local_size: int,
    dtype: Any | None = None,
) -> jax.Array:
    """Adds two U(1) labels and wraps the result.

    Args:
        lhs: Left charge values.
        rhs: Right charge values.
        q_min: Minimum charge value in the finite range.
        step: Charge spacing.
        local_size: Number of allowed charge values.
        dtype: Optional dtype for the wrapped result.

    Returns:
        Modular sum of ``lhs`` and ``rhs`` in the configured range.
    """
    out_dtype = jnp.result_type(lhs, rhs) if dtype is None else dtype
    return wrap_values(
        jnp.asarray(lhs, dtype=out_dtype) + jnp.asarray(rhs, dtype=out_dtype),
        q_min=q_min,
        step=step,
        local_size=local_size,
        dtype=out_dtype,
    )


def signed_modular_sum(
    values: Any,
    signs: Any,
    *,
    q_min: int | float,
    step: int | float,
    local_size: int,
    dtype: Any | None = None,
) -> jax.Array:
    """Computes a signed sum of labels followed by U(1) wrapping.

    Args:
        values: Charge values to combine along the trailing axis.
        signs: Sign coefficients paired with ``values``.
        q_min: Minimum charge value in the finite range.
        step: Charge spacing.
        local_size: Number of allowed charge values.
        dtype: Optional dtype for the wrapped result.

    Returns:
        Wrapped signed sum along the trailing axis.
    """
    vals = jnp.asarray(values)
    out_dtype = vals.dtype if dtype is None else dtype
    signed = vals.astype(out_dtype) * jnp.asarray(signs, dtype=out_dtype)
    return wrap_values(
        jnp.sum(signed, axis=-1),
        q_min=q_min,
        step=step,
        local_size=local_size,
        dtype=out_dtype,
    )
