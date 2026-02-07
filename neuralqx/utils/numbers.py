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

from __future__ import annotations

from typing import Any
from typing import Tuple

import numbers

import numpy as np

import jax.numpy as jnp


def is_scalar_like(x: Any) -> bool:
    """True for Python numbers, NumPy scalars, and 0-d arrays (NumPy/jax)."""

    if isinstance(x, numbers.Number):
        return True

    if isinstance(x, np.generic):
        return True
    # JAX/NumPy arrays
    ndim = getattr(x, "ndim", None)
    if ndim is not None:
        return int(ndim) == 0

    return False


def canonical_jax_dtype(x: Any | None) -> jnp.dtype | None:
    """
    Try to return a jnp.dtype for `x` (dtype, scalar, array, or None).
    Returns None if not deducible.
    """

    if x is None:
        return None

    if isinstance(x, (np.dtype, jnp.dtype)):
        return jnp.dtype(x)

    if hasattr(x, "dtype"):
        return jnp.dtype(x.dtype)

    if isinstance(x, numbers.Number) or isinstance(x, np.generic):
        return jnp.asarray(x).dtype

    return None


def _is_real_dtype(dt: jnp.dtype) -> bool:
    return jnp.issubdtype(dt, jnp.floating) or jnp.issubdtype(dt, jnp.integer)


def _to_0d_jax_array(x: Any) -> jnp.ndarray:
    """Convert to a 0-d JAX array (raises if not scalar-like)."""

    if not is_scalar_like(x):
        raise TypeError(
            f"Expected a scalar-like value, got shape-bearing type: {type(x)}"
        )

    arr = jnp.asarray(x)
    # ensure () shape
    if arr.ndim != 0:
        arr = jnp.asarray(arr).reshape(())

    return arr


def promote_constant_for_op_dtype(
    op_dtype: jnp.dtype,
    const: Any,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> Tuple[jnp.dtype, jnp.ndarray]:
    """
    Given an operator dtype and a scalar-like `const`, return
    (target_dtype, const_cast) where const_cast is a 0-d JAX array of target_dtype.

    Rules:
      - If operator is complex -> cast const to operator dtype (promote real->complex).
      - If operator is real:
          * If const is real -> cast to operator dtype.
          * If const is complex with non-zero imag (beyond tolerance) -> error.
            (User should cast operator to complex or pass a real constant.)
          * If const is complex with ~zero imag -> drop imag safely and cast to op dtype.

    This keeps the operator dtype stable unless we explicitly choose to upcast elsewhere.
    """

    op_dtype = jnp.dtype(op_dtype)
    c = _to_0d_jax_array(const)

    if jnp.issubdtype(op_dtype, jnp.complexfloating):
        # promote constant into operator's complex dtype (preserve precision of op)
        return op_dtype, c.astype(op_dtype)

    # operator is real
    if jnp.iscomplexobj(c):
        if jnp.all(jnp.isclose(jnp.imag(c), 0.0, rtol=rtol, atol=atol)):
            c_real = jnp.real(c)
            return op_dtype, c_real.astype(op_dtype)
        raise TypeError(
            f"Cannot add complex constant with non-zero imaginary part to a real operator "
            f"(op dtype={op_dtype}, const dtype={c.dtype}). "
            f"Cast the operator to a complex dtype or provide a real constant."
        )

    # both real: just cast to op dtype
    return op_dtype, c.astype(op_dtype)


def hermitian_flag_for_sum_with_scalar(
    op_is_hermitian: bool,
    const_cast: jnp.ndarray,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> bool:
    """
    For (op + c·I), result is Hermitian iff:
      - op is Hermitian, AND
      - imag(c) ~ 0  (within tolerance).
    """

    if not op_is_hermitian:
        return False

    if jnp.iscomplexobj(const_cast):
        return bool(
            jnp.all(jnp.isclose(jnp.imag(const_cast), 0.0, rtol=rtol, atol=atol))
        )

    return True


def to_python_scalar(x0d: jnp.ndarray) -> numbers.Number:
    """
    Convert a 0-d JAX array to a Python scalar (float/complex/int).
    """

    if not (hasattr(x0d, "ndim") and int(x0d.ndim) == 0):
        raise TypeError("to_python_scalar expects a 0-d array.")

    return x0d.item()
