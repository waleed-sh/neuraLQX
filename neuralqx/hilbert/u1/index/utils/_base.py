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
Utilities for ranking and unranking digit vectors in a fixed radix.

This module provides small, allocation-conscious helpers for converting between:

- a vector of digits :math:`(d_0, d_1, \\dots, d_{N-1})` with :math:`0 \\le d_i < \\text{base}`, and
- its corresponding non-negative integer label :math:`n \\in [0, \\text{base}^N)`.

Two digit order conventions are supported via :class:`~.Order`:

- ``"C"`` order: ``d_0`` is the most-significant digit (MSD).
- ``"F"`` order: ``d_0`` is the least-significant digit (LSD).

The helpers are intended to underpin NetKet-compatible state indexing, where a configuration is
treated as a mixed-radix (or fixed-radix) numeral system.

The functions in this module deliberately keep validation minimal for performance; callers may
choose to validate digit ranges and shapes at a higher level.
"""

from __future__ import annotations

from typing import Union

import math
import numpy as np

import jax
import jax.numpy as jnp

from ._types import Order


def _choose_return_dtype(
    base: int,
    n_digits: int,
    requested: Union[str, np.dtype, None],
) -> np.dtype:
    """
    Choose the NumPy dtype used for returned ranked numbers.

    This helper selects a safe return dtype for representing integers in the range

    .. math::

        0 \\le n < \\text{base}^{n_\\text{digits}}.

    The selection rules are:

    - If ``requested`` is ``None`` or ``"auto"``:
      choose ``np.int64`` when the maximum representable value
      :math:`\\text{base}^{n_\\text{digits}} - 1` fits in signed 64-bit integer range, and
      ``dtype=object`` otherwise (to use Python arbitrary-precision integers).
    - If ``requested`` is ``"int64"`` (or ``"i8"``): always return ``np.int64`` (may overflow if the
      range does not fit).
    - If ``requested`` is ``"object"`` (or ``"pyint"``, ``"python"``): always return ``dtype=object``.
    - If ``requested`` is a NumPy dtype (or dtype-like): return ``np.dtype(requested)``.

    The overflow check for ``"auto"`` uses logarithms to avoid evaluating
    :math:`\\text{base}^{n_\\text{digits}}` explicitly.

    :param base: Radix :math:`\\text{base}` of the positional number system.
    :param n_digits: Number of digits :math:`n_\\text{digits}`.
    :param requested: Desired dtype specifier (``"auto"``, ``"int64"``, ``"object"``, a dtype, or
        ``None``).

    :raises ValueError: If ``requested`` is a string that is not recognised.

    :return: The selected NumPy dtype for ranked numbers.
    """

    if requested is None or (isinstance(requested, str) and requested == "auto"):
        if base <= 1:
            return np.int64
        # numbers are in [0, base^n_digits), need base^n_digits - 1 <= int64_max
        # use logs to avoid huge pow
        int64_max = np.iinfo(np.int64).max

        # base^n_digits <= int64_max+1
        if n_digits == 0:
            return np.int64
        if n_digits * math.log(base) <= math.log(int64_max + 1):
            return np.int64
        return np.dtype(object)

    if isinstance(requested, str):
        if requested.lower() in ("object", "pyint", "python"):
            return np.dtype(object)
        if requested.lower() in ("int64", "i8"):
            return np.int64
        raise ValueError(
            f"Unknown return_dtype='{requested}'. Use 'auto', 'int64', or 'object'."
        )

    return np.dtype(requested)


def _rank_digits_base(
    digits2d: np.ndarray,
    base: int,
    order: Order,
    out_dtype: np.dtype,
) -> np.ndarray:
    """
    Rank base-``base`` digits into integer labels.

    Given a batch of digit vectors ``digits2d`` with shape ``(B, N)``, this function computes the
    corresponding integer representation in radix :math:`\\text{base}`:

    .. math::

        n = \\sum_{i=0}^{N-1} d_i\\,\\text{base}^{N-1-i} \\quad (\\text{C order})

    where each :math:`d_i \\in \\{0, 1, \\dots, \\text{base}-1\\}`.

    The mapping depends on ``order``:

    - ``order="C"``: ``digits2d[:, 0]`` is the most-significant digit (MSD), and ``digits2d[:, -1]``
      is the least-significant digit (LSD).
    - ``order="F"``: ``digits2d[:, 0]`` is the least-significant digit (LSD), and ``digits2d[:, -1]``
      is the most-significant digit (MSD). Equivalently, the digits are interpreted in reverse.

    If ``N == 0`` (no digits), the output is the zero vector of length ``B``.
    The output dtype is controlled by ``out_dtype``:

    - ``np.int64``: uses 64-bit integer arithmetic.
    - ``dtype=object``: uses Python arbitrary-precision integers.

    :param digits2d: Array of digits with shape ``(B, N)``.
    :param base: Radix :math:`\\text{base}`.
    :param order: Digit significance convention (``"C"`` or ``"F"``).
    :param out_dtype: Output dtype, typically ``np.int64`` or ``dtype=object``.

    :return: Ranked numbers of shape ``(B,)``.
    """

    B, N = digits2d.shape
    if N == 0:
        return np.zeros((B,), dtype=out_dtype)

    if order == "F":
        digits_iter = digits2d[:, ::-1]
    else:
        digits_iter = digits2d

    if out_dtype == np.int64:
        out = np.zeros((B,), dtype=np.int64)
        for i in range(N):
            out = out * base + digits_iter[:, i].astype(np.int64)
        return out

    # object dtype: Python big-int
    out_obj = np.zeros((B,), dtype=object)
    for i in range(N):
        # ensure Python ints
        di = digits_iter[:, i].astype(np.int64)
        out_obj = out_obj * base + di.astype(object)
    return out_obj


def _unrank_numbers_base(
    numbers1d: np.ndarray,
    base: int,
    n_digits: int,
    order: Order,
    *,
    digits_dtype: np.dtype = np.int64,
) -> np.ndarray:
    """
    Unrank integer labels into base-``base`` digits.

    Given a batch of non-negative integers ``numbers1d`` with shape ``(B,)``, this function returns
    their radix-:math:`\\text{base}` digit expansions with a fixed number of digits ``n_digits``,
    as an array of shape ``(B, n_digits)``.

    For each entry :math:`n`, the computed digits :math:`d_0,\\dots,d_{n_\\text{digits}-1}` satisfy

    .. math::

        n = \\sum_{i=0}^{n_\\text{digits}-1} d_i\\,\\text{base}^{n_\\text{digits}-1-i}

    in ``"C"`` order, with :math:`0 \\le d_i < \\text{base}`.

    The ``order`` convention determines where the least-significant digit appears:

    - ``order="C"``: the returned array is ``[\\text{MSD}, \\dots, \\text{LSD}]``.
    - ``order="F"``: the returned array is ``[\\text{LSD}, \\dots, \\text{MSD}]``.

    Range and validity checks:

    - All entries of ``numbers1d`` must be :math:`\\ge 0`.
    - Each entry must satisfy :math:`n < \\text{base}^{n_\\text{digits}}`; otherwise the number
      cannot be represented in ``n_digits`` digits and a ``ValueError`` is raised.

    The computation is performed per element (rather than vectorised) to support Python big-integers
    cleanly when ``numbers1d`` has ``dtype=object``.

    :param numbers1d: Array of non-negative integers with shape ``(B,)``.
    :param base: Radix :math:`\\text{base}`.
    :param n_digits: Fixed number of digits to emit.
    :param order: Digit significance convention (``"C"`` or ``"F"``).
    :param digits_dtype: NumPy dtype of the returned digits array.

    :raises ValueError: If any number is negative.
    :raises ValueError: If any number is :math:`\\ge \\text{base}^{n_\\text{digits}}`.

    :return: Digits array of shape ``(B, n_digits)`` in the requested order.
    """

    B = int(numbers1d.shape[0])
    digits_iter = np.zeros((B, n_digits), dtype=digits_dtype)

    # work per element to support python big-ints cleanly
    for b in range(B):
        n = int(numbers1d[b])
        if n < 0:
            raise ValueError("numbers must be >= 0.")
        for i in range(n_digits - 1, -1, -1):
            n, r = divmod(n, base)
            digits_iter[b, i] = r
        if n != 0:
            # overflow: number too large to fit in n_digits
            raise ValueError(
                "numbers contains an entry >= base**n_digits (out of range for this space)."
            )

    if order == "F":
        return digits_iter[:, ::-1]
    return digits_iter


def _rank_digits_base_jax(digits2d: jax.Array, base: int, order: Order) -> jax.Array:
    """Rank digits (B,N) -> numbers (B,) using int64 arithmetic."""
    d = jnp.asarray(digits2d, dtype=jnp.int64)
    if order == "F":
        d = d[:, ::-1]

    base_i = jnp.int64(base)
    B, N = d.shape
    out0 = jnp.zeros((B,), dtype=jnp.int64)

    def body(i, acc):
        return acc * base_i + d[:, i]

    return jax.lax.fori_loop(0, N, body, out0)


def _unrank_numbers_base_jax(
    numbers1d: jax.Array, base: int, n_digits: int, order: Order
) -> jax.Array:
    """Unrank numbers (B,) -> digits (B,n_digits) using int64 arithmetic."""
    n = jnp.asarray(numbers1d, dtype=jnp.int64)
    base_i = jnp.int64(base)
    B = n.shape[0]
    digits = jnp.zeros((B, n_digits), dtype=jnp.int64)

    def body(i, carry):
        ncur, dcur = carry
        r = jnp.remainder(ncur, base_i)
        nnext = jnp.floor_divide(ncur, base_i)
        dcur = dcur.at[:, n_digits - 1 - i].set(r)  # fill from LSD upward
        return (nnext, dcur)

    _, digits = jax.lax.fori_loop(0, n_digits, body, (n, digits))
    if order == "F":
        digits = digits[:, ::-1]
    return digits
