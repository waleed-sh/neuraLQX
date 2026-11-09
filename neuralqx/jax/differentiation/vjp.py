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


"""VJP wrappers with explicit real/complex-output semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.tree import tree_any_complex


def vjp(
    fun: Callable[..., Any],
    *primals: Any,
    conjugate: bool = False,
    has_aux: bool = False,
):
    """Return a VJP that handles real-parameter complex-output functions.

    JAX's raw VJP is correct for many complex cases, but VMC force estimators
    need consistent behavior across R->R, R->C, C->R, and C->C call signatures.
    For real primals and complex outputs, the pullback returns the full
    complex contraction ``sum(cotangent * derivative)``. Both channels share
    one forward trace. Complex primals retain JAX's VJP convention.
    ``conjugate=True`` conjugates the resulting contraction, so it is useful
    for covariance forces but must not be used for a direct complex derivative
    with respect to real coordinates.
    """
    if has_aux:
        output, pullback, aux = jax.vjp(fun, *primals, has_aux=True)
    else:
        output, pullback = jax.vjp(fun, *primals)
    real_to_complex = not tree_any_complex(primals) and jnp.iscomplexobj(output)

    def wrapped(cotangent: Any):
        if real_to_complex:
            # For real coordinates JAX returns Re(c J). A second pullback
            # with i c recovers -Im(c J), using the same forward trace.
            cotangent = jnp.asarray(cotangent, dtype=output.dtype)
            real = pullback(cotangent)
            minus_imag = pullback(1j * cotangent)
            result = jax.tree_util.tree_map(
                lambda r, i: _complex_from_parts(r, i, imag_sign=-1), real, minus_imag
            )
        elif jnp.iscomplexobj(cotangent) and not jnp.iscomplexobj(output):
            real = pullback(jnp.asarray(jnp.real(cotangent), dtype=output.dtype))
            imag = pullback(jnp.asarray(jnp.imag(cotangent), dtype=output.dtype))
            result = jax.tree_util.tree_map(_complex_from_parts, real, imag)
        else:
            result = pullback(jnp.asarray(cotangent, dtype=output.dtype))
        if conjugate:
            result = jax.tree_util.tree_map(_conjugate, result)
        return result

    if has_aux:
        return output, wrapped, aux
    return output, wrapped


def _complex_from_parts(real: Any, imag: Any, *, imag_sign: int = 1) -> Any:
    real_arr = jnp.asarray(real)
    imag_arr = jnp.asarray(imag)
    if jax.dtypes.issubdtype(real_arr.dtype, jax.dtypes.float0):
        return real
    if jax.dtypes.issubdtype(imag_arr.dtype, jax.dtypes.float0):
        return real
    return real + (imag_sign * 1j) * imag


def _conjugate(value: Any) -> Any:
    if jax.dtypes.issubdtype(jnp.asarray(value).dtype, jax.dtypes.float0):
        return value
    return jnp.conjugate(value)


__all__ = ["vjp"]
