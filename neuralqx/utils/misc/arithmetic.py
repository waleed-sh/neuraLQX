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


from typing import Optional, Union

import jax
import numpy as np


def mod_add(
    m: Union[np.ndarray, int, float, jax.Array],
    n: Union[np.ndarray, int, float, jax.Array],
    *,
    q_min: Union[int, float] = None,
    q_max: Union[int, float] = None,
    step: Union[int, float] = 1,
    cutoff: Optional[Union[int, float]] = None,
) -> Union[np.ndarray, int, float, jax.Array]:
    """
    Modular addition over a discrete quantum-number domain.

    This function computes ``m (+) n`` and wraps the result back into the allowed domain using
    modular arithmetic. The domain is interpreted as a lattice ``{q_min + k * step}`` with
    endpoints ``[q_min, q_max]`` (inclusive). If ``q_min``/``q_max`` are not provided, a symmetric
    domain ``[-cutoff, cutoff]`` is used.

    The wrapping formula is::

        res = ((m + n - q_min) % (q_max - q_min + step)) + q_min

    Dtype behavior:

    - The function probes a short prefix of the lattice (three points) to determine whether the
      domain is integer-valued.
    - If the domain is integer-valued, the result is cast back to the NumPy/JAX result type of
      ``m`` and ``n`` (e.g. to preserve integer dtypes for integer domains).
    - If the domain is fractional (e.g. half-integer), the result is returned without forcing an
      integer cast.

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
            raise ValueError("Must provide either `cutoff or (`q_min`, `q_max`).")
        q_min = -cutoff
        q_max = cutoff

    span = q_max - q_min + step
    res = ((m + n - q_min) % span) + q_min

    # DOMAIN-BASED CAST DECISION
    # construct a short segment of the domain
    # (3 points is enough: q_min, q_min + step, q_min + 2*step)
    qs = np.array([q_min + k * step for k in range(3)])

    # check whether all allowed values are integer-valued
    domain_is_integer = np.all(np.isclose(qs, np.round(qs)))

    if domain_is_integer:
        # safe to cast back to integer type
        target_dtype = np.result_type(m, n)

        if isinstance(res, np.ndarray):
            res = res.astype(target_dtype)
        else:
            import jax.numpy as jnp

            res = jnp.asarray(res).astype(target_dtype)

    return res


def mod_sum(
    array,
    *,
    q_min=None,
    q_max=None,
    step=1,
    cutoff=None,
):
    """
    Compute a modular sum of an iterable using :func:`mod_add`.

    This function behaves like Python's ``sum()``, except that addition is performed with modular
    wrapping into the specified quantum-number domain. Summation is performed sequentially
    (left-fold) using::

        s <- mod_add(s, e, ...)

    Domain specification:

    - If ``q_min`` and ``q_max`` are provided, the allowed domain is the lattice
      ``{q_min + k * step}`` with endpoints ``[q_min, q_max]`` (inclusive).
    - Otherwise, if ``cutoff`` is provided, the domain is the symmetric interval
      ``[-cutoff, cutoff]`` using the given ``step``.

    :param array: Iterable of values to modular-sum.
    :param q_min: Lower bound of the domain (inclusive). Must be provided together with ``q_max``
                  unless ``cutoff`` is used.
    :param q_max: Upper bound of the domain (inclusive). Must be provided together with ``q_min``
                  unless ``cutoff`` is used.
    :param step: Lattice spacing of the domain. Defaults to 1.
    :param cutoff: If provided and ``q_min``/``q_max`` are not, sets ``q_min=-cutoff`` and
                   ``q_max=cutoff``.
    :returns: The modular sum wrapped into the specified domain.
    :raises ValueError: If neither ``(q_min, q_max)`` nor ``cutoff`` are provided.
    """

    s = 0
    for e in array:
        s = mod_add(s, e, q_min=q_min, q_max=q_max, step=step, cutoff=cutoff)
    return s


def get_sgn(n: Union[int, float]) -> int:
    """
    Return the sign of a number as ``-1`` or ``+1``.

    Note: zero is treated as non-negative and returns ``+1``.

    :param n: Number whose sign should be returned.
    :returns: ``-1`` if ``n < 0``, otherwise ``+1``.
    """

    return -1 if n < 0 else 1


def get_signed_value(s: str) -> int:
    """
    Parse a leading sign from a string and return it as ``-1`` or ``+1``.

    This helper inspects only the first character of the string:
    if ``s`` starts with ``"-"`` it returns ``-1``, otherwise it returns ``+1``.
    It does not validate that the rest of the string is numeric.

    :param s: Input string potentially beginning with a minus sign.
    :returns: ``-1`` if ``s`` starts with ``"-"``, otherwise ``+1``.
    """

    if s.startswith("-"):
        return -1

    return 1


def factorial(n: int) -> int:
    """
    Compute the factorial of a non-negative integer.

    This implementation uses recursion and returns ``n!``. For ``n == 0`` the function returns 1.

    :param n: Non-negative integer for which to compute the factorial.
    :returns: The factorial ``n!``.
    :raises ValueError: If ``n`` is negative.
    """

    if n < 0:
        raise ValueError("Cannot compute factorial of a negative number.")

    if n == 0:
        return 1

    return n * factorial(n - 1)


def generate_plus_minus_one(key, shape=()):
    """
    Generate random values in ``{+1, -1}`` using JAX PRNG.

    The function samples integer bits in ``{0, 1}`` and maps them to ``{-1, +1}`` via
    ``2 * bits - 1``. The output is compatible with JAX transformations.

    :param key: JAX PRNGKey used for sampling.
    :param shape: Output shape. Defaults to ``()`` (a scalar).
    :returns: A JAX array of the given shape containing only ``-1`` and ``+1``.
    """

    bits = jax.random.randint(key, shape=shape, minval=0, maxval=2)
    return 2 * bits - 1


def plus_key(
    t: Union[tuple, list],
    number: float,
) -> tuple:
    """
    Add a constant to all but the last element of a tuple/list key.

    This helper returns a new tuple where ``number`` is added to each element of ``t[:-1]``.
    The last element ``t[-1]`` is preserved unchanged. This is useful when the last component is a
    label/metadata entry that should not be shifted.

    :param t: Input tuple/list. The last element is kept unchanged.
    :param number: Constant to add to each element of ``t[:-1]``.
    :returns: A new tuple with shifted elements and the original last element.
    """

    return (*tuple(element + number for element in t[:-1]), t[-1])


def minus_key(
    t: Union[tuple, list],
    number: float,
) -> tuple:
    """
    Subtract a constant from all but the last element of a tuple/list key.

    This helper returns a new tuple where ``number`` is subtracted from each element of ``t[:-1]``.
    The last element ``t[-1]`` is preserved unchanged. This is useful when the last component is a
    label/metadata entry that should not be shifted.

    :param t: Input tuple/list. The last element is kept unchanged.
    :param number: Constant to subtract from each element of ``t[:-1]``.
    :returns: A new tuple with shifted elements and the original last element.
    """

    return (*tuple(element - number for element in t[:-1]), t[-1])
