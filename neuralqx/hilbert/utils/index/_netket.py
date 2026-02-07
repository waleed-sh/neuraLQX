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
NetKet ordering and local-basis conversion utilities.

This module provides helper functions for interfacing NumPy/JAX arrays with NetKet's
:class:`netket.hilbert.HomogeneousHilbert` indexing conventions. It focuses on two related tasks:

1. **Inferring NetKet's site significance order** for multi-site basis numbering.

   For a configuration :math:`\\mathbf{d} = (d_0, \\dots, d_{N-1})` with local digits
   :math:`d_i \\in \\{0, \\dots, L-1\\}`, NetKet maps :math:`\\mathbf{d}` to an integer
   :math:`n(\\mathbf{d})`. The ordering determines which site is treated as more significant:

   - **C-order**: :math:`d_0` is most significant and :math:`d_{N-1}` is least significant.
   - **F-order**: :math:`d_0` is least significant and :math:`d_{N-1}` is most significant.

   The helper ``_infer_netket_site_order`` determines this convention by constructing a tiny
   two-site Hilbert space and comparing the numbers assigned to two basis states.

2. **Mapping between physical local values and integer digits**.

   Many models use physical local values (e.g. quantum numbers) rather than integer digits.
   The helpers in this module convert between:

   - physical values :math:`v \\in \\mathcal{S}` (where ``local_states`` enumerates
     :math:`\\mathcal{S}`), and
   - integer digits :math:`d \\in \\{0, \\dots, L-1\\}` consistent with NetKet's internal ordering.

   These conversions use ``local_states.states_to_numbers`` and
   ``local_states.numbers_to_states`` to guarantee compatibility with NetKet.

The functions in this module are designed to be used by higher-level indexing code that needs to
match NetKet's ``states_to_numbers`` / ``numbers_to_states`` behaviour exactly, including the
site-order convention.
"""

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

import netket as nk
from netket.utils import StaticRange

from ._array_normalise import _as_numpy
from ._types import Order


def _infer_netket_site_order(local_states: Any) -> Order:
    r"""
    Infer NetKet's site-significance convention (``"C"`` vs ``"F"``) for a
    :class:`netket.hilbert.HomogeneousHilbert` constructed from the provided ``local_states``.

    NetKet maps a configuration of local basis values :math:`(x_0, x_1, \dots, x_{N-1})` to a single
    integer label via ``states_to_numbers``. For a fixed local basis of size :math:`L`, this induces a
    mixed-radix ranking of the corresponding local *digits* :math:`(d_0, \dots, d_{N-1})` with
    :math:`d_i \in \{0, \dots, L-1\}`. The remaining ambiguity is the **site order** (which site is most
    significant):

    - ``"C"`` order: :math:`d_0` is most significant and :math:`d_{N-1}` is least significant.
    - ``"F"`` order: :math:`d_0` is least significant and :math:`d_{N-1}` is most significant.

    This helper constructs a tiny :math:`N=2` reference Hilbert space and compares the integer labels of
    two basis states built from the first two local values. For :math:`L = |\mathcal{S}|`:

    - If ``num([s_0, s_1]) = 1`` and ``num([s_1, s_0]) = L``, the convention is ``"C"``.
    - If ``num([s_0, s_1]) = L`` and ``num([s_1, s_0]) = 1``, the convention is ``"F"``.

    To avoid accidentally creating an enormous reference space, very large ``local_states`` objects may
    be truncated to a smaller representative subset before probing.

    :param local_states: NetKet local state container (e.g. :class:`netket.utils.StaticRange`) providing
      ``all_states()``, ``length``, and suitable mapping behaviour for ``states_to_numbers``.
    :return: The inferred ordering flag, either ``"C"`` or ``"F"``.
    :raises RuntimeError: If the probe Hilbert cannot be constructed, if NetKet's
      ``states_to_numbers`` fails for the reference states, or if the observed numbering strides are
      inconsistent with a ``"C"``/``"F"`` convention.
    """

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
        # traced: return a JAX scalar
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
    r"""
    Convert physical local values to integer digits compatible with NetKet's local basis indexing.

    Given a 2D array of local values ``values2d`` with shape :math:`(B, N)`, this function maps each
    entry :math:`x \in \mathcal{S}` (the local basis set encoded by ``local_states``) to its associated
    integer digit :math:`d \in \{0, \dots, L-1\}` such that

    .. math::

        d = \mathrm{num}_{\text{local}}(x),
        \qquad x = \mathrm{state}_{\text{local}}(d),

    where the conversions are those implemented by NetKet via
    ``local_states.states_to_numbers`` / ``local_states.numbers_to_states``.

    Using NetKet's mapping ensures that downstream ranking/unranking of full configurations is
    consistent with NetKet's own ``states_to_numbers`` convention.

    :param local_states: NetKet local state container providing ``states_to_numbers`` and representing
      the allowed local basis values.
    :param values2d: Array of shape :math:`(B, N)` containing physical local values drawn from the
      provided basis.
    :return: Integer digits array of shape :math:`(B, N)` with entries in ``[0, local_size)``.
    :raises ValueError: If any entry of ``values2d`` is not contained in the local basis represented by
      ``local_states``.
    """

    flat = values2d.reshape((-1,))
    try:
        digits_flat = _as_numpy(local_states.states_to_numbers(flat))
    except Exception as e:
        raise ValueError("Input contains values not in the local basis.") from e
    return _as_numpy(digits_flat).reshape(values2d.shape)


def _local_digits_to_values(local_states: Any, digits2d: np.ndarray) -> np.ndarray:
    r"""
    Convert integer digits to physical local values using NetKet's local basis mapping.

    Given a 2D array ``digits2d`` with shape :math:`(B, N)` and entries
    :math:`d \in \{0, \dots, L-1\}`, this function returns the corresponding physical local values
    :math:`x \in \mathcal{S}` according to NetKet's local basis enumeration:

    .. math::

        x = \mathrm{state}_{\text{local}}(d).

    This is the inverse of :func:`_local_values_to_digits` when both are applied with the same
    ``local_states``.

    :param local_states: NetKet local state container providing ``numbers_to_states`` and representing
      the allowed local basis values.
    :param digits2d: Array of shape :math:`(B, N)` containing integer digits.
    :return: Array of shape :math:`(B, N)` containing the corresponding physical local values.
    :raises ValueError: If any entry of ``digits2d`` is outside the valid digit range for the provided
      ``local_states`` (i.e. not in ``[0, local_size)``).
    """

    flat = digits2d.reshape((-1,))
    try:
        vals_flat = _as_numpy(local_states.numbers_to_states(flat))
    except Exception as e:
        raise ValueError("Input contains invalid local digits.") from e
    return _as_numpy(vals_flat).reshape(digits2d.shape)


def _local_values_to_digits_jax(local_states: Any, values2d: jax.Array) -> jax.Array:
    """
    Map local physical values to digits [0, L).

    Supports:
        - StaticRange-like objects with attributes (start, step, length).
        - Generic objects exposing all_states().

    The generic path uses equality matching against enumerated values. This is robust for small
    local bases and is jit-compatible.
    """

    # fast path for StaticRange (and similar): arithmetic mapping
    if (
        hasattr(local_states, "start")
        and hasattr(local_states, "step")
        and hasattr(local_states, "length")
    ):
        start = jnp.asarray(local_states.start, dtype=values2d.dtype)
        step = jnp.asarray(local_states.step, dtype=values2d.dtype)

        # rint tolerates float representational quirks (e.g. half-integers stored as float)
        return jnp.rint((values2d - start) / step).astype(jnp.int64)

    # generic path: match against enumerated local values
    vals = jnp.asarray(local_states.all_states())
    matches = values2d[..., jnp.newaxis] == vals
    return jnp.argmax(matches, axis=-1).astype(jnp.int64)


def _local_digits_to_values_jax(local_states: Any, digits2d: jax.Array) -> jax.Array:
    """Map digits [0, L) to local physical values."""
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
