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
Utilities for gauge-fixing checks and JAX-friendly constraint evaluation.

This module provides:

- :func:`check_one_state_jax`, a JIT-compiled predicate that tests whether a single configuration
  satisfies a collection of preprocessed gauge-fixing constraints in a modular arithmetic setting;
- :func:`make_hashable`, a small helper that converts common Python/JAX containers into hashable
  equivalents (useful for caching and memoisation of processed constraint data).

The constraint checks are formulated in terms of modular sums over signed edge variables. For each
constraint, the incoming and outgoing signed sums are compared after applying the modular reduction.
"""

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.utils.jit import mod_sum_jax


@jax.jit
def check_one_state_jax(state, constraints, q_min, q_max, q_step):
    """
    Check all gauge-fixing constraints for a single configuration.

    Given a state ``state`` of shape ``(N,)`` and a preprocessed list of constraint specifications,
    this function verifies that every constraint is satisfied. Each constraint is represented as a pair
    ``(inc_data, out_data)``, where both entries are lists of ``(idx, sgn)`` pairs. The value
    ``state[idx]`` is multiplied by its sign ``sgn`` and then combined using a modular sum.

    For a single constraint, define the signed incoming and outgoing aggregates

    .. math::

        S_{\\mathrm{in}} = \\bigoplus_{(i, s) \\in \\mathrm{inc}} (s\\,\\sigma_i), \\qquad
        S_{\\mathrm{out}} = \\bigoplus_{(j, s) \\in \\mathrm{out}} (s\\,\\sigma_j),

    where :math:`\\sigma_i = \\texttt{state}[i]` and :math:`\\oplus` denotes the modular addition
    implemented by :func:`neuralqx.utils.jit.mod_sum_jax` on the discrete domain
    ``[q_min, q_max]`` with step ``q_step``. The constraint is satisfied iff

    .. math::

        S_{\\mathrm{in}} = S_{\\mathrm{out}}.

    All constraints must pass for the state to be accepted.

    :param state: One configuration array of shape ``(N,)``.
    :param constraints: Preprocessed constraints as produced by the corresponding preprocessing helper.
        Each item is ``(inc_data, out_data)`` with ``inc_data`` and ``out_data`` lists of
        ``(idx, sgn)`` pairs.
    :param q_min: Minimum allowed local quantum number for the modular domain.
    :param q_max: Maximum allowed local quantum number for the modular domain.
    :param q_step: Step size between allowed local quantum numbers.
    :return: A boolean JAX scalar indicating whether all constraints are satisfied.
    """

    is_satisfied = True
    for inc_data, out_data in constraints:
        # compute the incoming sum
        inc_vals = []
        for idx, sgn in inc_data:
            inc_vals.append(sgn * state[idx])
        inc_vals = jnp.array(inc_vals)
        inc_sum = mod_sum_jax(inc_vals, q_min=q_min, q_max=q_max, step=q_step)

        # compute the outgoing sum
        out_vals = []
        for idx, sgn in out_data:
            out_vals.append(sgn * state[idx])
        out_vals = jnp.array(out_vals)
        out_sum = mod_sum_jax(out_vals, q_min=q_min, q_max=q_max, step=q_step)

        match = inc_sum == out_sum
        is_satisfied = jnp.logical_and(is_satisfied, match)

    return is_satisfied


def make_hashable(obj: Any) -> Any:
    """
    Recursively convert an object into a hashable representation.

    This helper is intended for building stable cache keys from nested container structures commonly
    encountered in constraint specifications and preprocessed data.

    Conversion rules:

    - ``list``/``tuple`` are converted into tuples of recursively converted items;
    - ``dict`` is converted into a sorted tuple of ``(key, value)`` pairs, with values converted
      recursively;
    - array-like objects exposing ``.tolist()`` (for example JAX or NumPy arrays) are converted via
      their Python list representation and then processed recursively;
    - all other objects are returned unchanged.

    :param obj: Any Python object (possibly nested).
    :return: A hashable equivalent suitable for use as a dictionary key.
    """

    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif hasattr(obj, "tolist"):
        return make_hashable(obj.tolist())
    else:
        return obj
