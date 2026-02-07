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
This file contains numba jitted implementations of various functions
"""

from typing import Union
from numba import njit

import numpy as np
import jax.numpy as jnp


def _domain_is_integer(q_min, step):
    qs = jnp.array([q_min + k * step for k in range(3)])
    return jnp.all(jnp.isclose(qs, jnp.round(qs), rtol=0.0, atol=1e-12))


@njit
def mod_add_njit(
    m: Union[np.ndarray, int, float],
    n: Union[np.ndarray, int, float],
    q_min: Union[np.ndarray, int, float],
    q_max: Union[np.ndarray, int, float],
    step: Union[np.ndarray, int, float] = 1,
):
    r"""
    A function which imposes the q-deformed representation label addition for the :math:`U_q(1)`
    group. The modded addition follows the following prescription:

    .. math::
        m \oplus n := (m + n + N) mod(2N + 1) - N

    for two representation labels m and n and a cutoff of N.

    :param m: an int or float representing a representation label, or a matrix
    :param n: an int or float representing a representation label, or a matrix
    :param q_min: the smallest allowed DoF
    :param q_max: the largest allowed DoF
    :param step: the step size from one DoF to another
    """
    span = q_max - q_min + step
    return ((m + n - q_min) % span) + q_min

    # TODO: values must lie on a discrete lattice defined by step then?
    # return q_min + step * (((m + n - q_min) // step) % ((q_max - q_min) // step + 1))


def mod_add_jax(m, n, *, q_min=None, q_max=None, step=1, cutoff=None):
    """
    Perform modular addition over a discrete quantum-number domain in a JAX-friendly way.

    This is the JAX-safe counterpart of :func:`neuralqx.utils.misc.arithmetic.mod_add`. It computes
    ``m (+) n`` and wraps the result into the specified lattice domain using modular arithmetic,
    while keeping dtypes stable under JAX transformations (``jit``, ``vmap``, ``grad``).

    Domain specification:

    - If ``q_min`` and ``q_max`` are provided, the allowed domain is interpreted as the lattice
      ``{q_min + k * step}`` with endpoints ``[q_min, q_max]`` (inclusive).
    - Otherwise, if ``cutoff`` is provided, the domain is the symmetric interval ``[-cutoff, cutoff]``
      using the given ``step``.

    The wrapping formula used is::

        res = ((m + n - q_min) % (q_max - q_min + step)) + q_min

    Dtype behavior:

    - Computation is performed in an "operation dtype" that can represent the domain parameters.
    - If the probed domain is integer-valued (based on ``q_min`` and ``step``) and the result type
      of ``m`` and ``n`` is integer, the result is cast back to that integer dtype. Otherwise, the
      result remains in the operation dtype.

    :param m: First addend. Can be a scalar or array; must be broadcast-compatible with ``n``.
    :param n: Second addend. Can be a scalar or array; must be broadcast-compatible with ``m``.
    :param q_min: Lower bound of the domain (inclusive). Must be provided together with ``q_max``
                  unless ``cutoff`` is used.
    :param q_max: Upper bound of the domain (inclusive). Must be provided together with ``q_min``
                  unless ``cutoff`` is used.
    :param step: Lattice spacing of the domain. Defaults to 1.
    :param cutoff: If provided and ``q_min``/``q_max`` are not, sets ``q_min=-cutoff`` and
                   ``q_max=cutoff``.
    :returns: The modular sum wrapped into the specified domain, with the broadcasted shape of
              ``m`` and ``n``.
    :raises ValueError: If neither ``(q_min, q_max)`` nor ``cutoff`` are provided.
    """

    if q_min is None or q_max is None:
        if cutoff is None:
            raise ValueError("Must provide either `cutoff` or (`q_min`, `q_max`).")
        q_min = -cutoff
        q_max = cutoff

    span = q_max - q_min + step
    domain_is_int = _domain_is_integer(float(q_min), float(step))

    # do arithmetic in an op dtype that can represent q_min/span
    op_dtype = jnp.result_type(m, n, jnp.asarray(q_min), jnp.asarray(span))
    q_min_ = jnp.asarray(q_min, dtype=op_dtype)
    span_ = jnp.asarray(span, dtype=op_dtype)

    res = ((m.astype(op_dtype) + n.astype(op_dtype) - q_min_) % span_) + q_min_

    # cast back only if the domain is integer-valued AND inputs are integer
    if domain_is_int and jnp.issubdtype(jnp.result_type(m, n), jnp.integer):
        res = res.astype(jnp.result_type(m, n))

    return res


def mod_sum_jax(values, *, q_min=None, q_max=None, step=1, cutoff=None):
    """
    Compute a modular sum over the last axis of ``values`` in a JAX-transformation-friendly way.

    This function reduces ``values`` along ``axis=-1`` and wraps the result back into a discrete
    quantum-number domain using modular arithmetic. It is designed to work under JAX
    transformations (``jit``, ``vmap``, ``grad``) by avoiding control flow that would cause
    dtype/shape changes during tracing.

    Domain specification:

    - If ``q_min`` and ``q_max`` are provided, the allowed domain is interpreted as the lattice
      ``{q_min + k * step}`` with endpoints ``[q_min, q_max]`` (inclusive).
    - Otherwise, if ``cutoff`` is provided, the domain is the symmetric interval ``[-cutoff, cutoff]``
      using the given ``step``.

    The wrapping formula used is::

        out = ((s - q_min) % (q_max - q_min + step)) + q_min

    where ``s = sum(values, axis=-1)``.

    Dtype behavior:

    - Computation is performed in an "operation dtype" that can safely represent the modular
      arithmetic with the provided domain parameters.
    - If the probed domain is integer-valued (based on ``q_min`` and ``step``) and ``values`` has an
      integer dtype, the output is cast back to ``values.dtype``. Otherwise, the output remains in
      the operation dtype.

    :param values: Array of values to modular-sum. The reduction is performed over the last axis.
                   Can be 1D (returns a scalar) or batched with arbitrary leading dimensions.
    :param q_min: Lower bound of the domain (inclusive). Must be provided together with ``q_max``
                  unless ``cutoff`` is used.
    :param q_max: Upper bound of the domain (inclusive). Must be provided together with ``q_min``
                  unless ``cutoff`` is used.
    :param step: Lattice spacing of the domain. Defaults to 1.
    :param cutoff: If provided and ``q_min``/``q_max`` are not, sets ``q_min=-cutoff`` and
                   ``q_max=cutoff``.
    :returns: The modular sum wrapped into the specified domain. Shape is ``values.shape[:-1]``.
    :raises ValueError: If neither ``(q_min, q_max)`` nor ``cutoff`` are provided.
    """
    if q_min is None or q_max is None:
        if cutoff is None:
            raise ValueError("Provide either `cutoff` or (`q_min`, `q_max`)")
        q_min = -cutoff
        q_max = cutoff

    span = q_max - q_min + step
    domain_is_int = _domain_is_integer(
        jnp.asarray(q_min, dtype=float),
        jnp.asarray(step, dtype=float),
    )

    op_dtype = jnp.result_type(values.dtype, jnp.asarray(q_min), jnp.asarray(span))
    q_min_ = jnp.asarray(q_min, dtype=op_dtype)
    span_ = jnp.asarray(span, dtype=op_dtype)

    s = jnp.sum(values.astype(op_dtype), axis=-1)
    out = ((s - q_min_) % span_) + q_min_

    out_int = out.astype(values.dtype)

    is_int_domain = domain_is_int & jnp.issubdtype(values.dtype, jnp.integer)

    out = jnp.where(is_int_domain, out_int, out)

    return out.astype(values.dtype)
