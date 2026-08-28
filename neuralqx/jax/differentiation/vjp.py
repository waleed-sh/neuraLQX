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
    For real primals and complex outputs, this wrapper differentiates real and
    imaginary output channels separately and recombines the pullback so callers
    can pass a complex cotangent directly.
    """
    if has_aux:
        output_shape, _ = jax.eval_shape(fun, *primals)
    else:
        output_shape = jax.eval_shape(fun, *primals)
    primals_are_complex = tree_any_complex(primals)
    output_is_complex = bool(jnp.iscomplexobj(output_shape))
    if primals_are_complex or not output_is_complex:
        return _vjp_standard(
            fun,
            *primals,
            conjugate=conjugate,
            has_aux=has_aux,
        )
    return _vjp_real_to_complex(
        fun,
        *primals,
        conjugate=conjugate,
        has_aux=has_aux,
    )


def _vjp_standard(
    fun: Callable[..., Any],
    *primals: Any,
    conjugate: bool,
    has_aux: bool,
):
    if has_aux:
        output, pullback, aux = jax.vjp(fun, *primals, has_aux=True)
    else:
        output, pullback = jax.vjp(fun, *primals)

    def wrapped(cotangent: Any):
        if jnp.iscomplexobj(cotangent) and not jnp.iscomplexobj(output):
            real = pullback(jnp.asarray(jnp.real(cotangent), dtype=output.dtype))
            imag = pullback(jnp.asarray(jnp.imag(cotangent), dtype=output.dtype))
            result = jax.tree_util.tree_map(_complex_from_parts, real, imag)
        else:
            result = pullback(jnp.asarray(cotangent, dtype=output.dtype))
        if conjugate:
            result = jax.tree_util.tree_map(jnp.conjugate, result)
        return result

    if has_aux:
        return output, wrapped, aux
    return output, wrapped


def _vjp_real_to_complex(
    fun: Callable[..., Any],
    *primals: Any,
    conjugate: bool,
    has_aux: bool,
):
    if has_aux:

        def real_fun(*args: Any):
            value, aux = fun(*args)
            return jnp.real(value), aux

        def imag_fun(*args: Any):
            value, aux = fun(*args)
            return jnp.imag(value), aux

        real_output, real_pullback, aux = jax.vjp(real_fun, *primals, has_aux=True)
        imag_output, imag_pullback, _ = jax.vjp(imag_fun, *primals, has_aux=True)
    else:
        real_fun = lambda *args: jnp.real(fun(*args))
        imag_fun = lambda *args: jnp.imag(fun(*args))
        real_output, real_pullback = jax.vjp(real_fun, *primals)
        imag_output, imag_pullback = jax.vjp(imag_fun, *primals)
        aux = None

    output = real_output + 1j * imag_output

    def wrapped(cotangent: Any):
        cotangent = jnp.asarray(cotangent, dtype=output.dtype)
        real_real = real_pullback(
            jnp.asarray(jnp.real(cotangent), dtype=real_output.dtype)
        )
        real_imag = real_pullback(
            jnp.asarray(jnp.imag(cotangent), dtype=real_output.dtype)
        )
        imag_real = imag_pullback(
            jnp.asarray(jnp.real(cotangent), dtype=imag_output.dtype)
        )
        imag_imag = imag_pullback(
            jnp.asarray(jnp.imag(cotangent), dtype=imag_output.dtype)
        )

        real_part = jax.tree_util.tree_map(_complex_from_parts, real_real, real_imag)
        imag_part = jax.tree_util.tree_map(_complex_from_parts, imag_real, imag_imag)
        result = jax.tree_util.tree_map(_complex_from_parts, real_part, imag_part)
        if conjugate:
            result = jax.tree_util.tree_map(jnp.conjugate, result)
        return result

    if has_aux:
        return output, wrapped, aux
    return output, wrapped


def _complex_from_parts(real: Any, imag: Any) -> Any:
    real_arr = jnp.asarray(real)
    imag_arr = jnp.asarray(imag)
    if jax.dtypes.issubdtype(real_arr.dtype, jax.dtypes.float0):
        return real
    if jax.dtypes.issubdtype(imag_arr.dtype, jax.dtypes.float0):
        return real
    return real + 1j * imag


__all__ = ["vjp"]
