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
Utilities for reducing and reconstructing gauge-fixed configurations.

This module provides small helpers used by indexing and conversion routines when working with
gauge-fixed constrained Hilbert-space cores. In a gauge-fixed setting, a full configuration
:math:`\\sigma \\in \\mathbb{R}^{N}` (or :math:`\\mathbb{Z}^{N}`) is flattened in *gauge blocks*:

.. math::

    N = G E, \\qquad
    \\sigma = (\\sigma^{(0)}, \\sigma^{(1)}, \\dots, \\sigma^{(G-1)}),

where :math:`G` is the number of gauge copies and :math:`E` is the number of edge degrees of freedom
per copy. Each block :math:`\\sigma^{(g)} \\in \\mathbb{R}^{E}` contains a subset of *free* variables
(index set ``gauge_fixing.free``) and a complementary subset of *slave* variables, which are
deterministically fixed by the gauge-fixing prescription.

The helpers in this module support two common operations:

1. Extract a reduced representation consisting only of the free variables across all gauge copies,
   ordered as ``[copy0 | copy1 | ...]``.
2. Reconstruct full gauge-invariant configurations by placing the free variables into the correct
   sites and calling ``reimpose_gauge_fixing`` to fill the dependent variables.

These functions are intentionally lightweight and rely on a small structural contract: a compatible
space exposes ``gauge_fixing`` (with a ``free`` index array) and a method
``reimpose_gauge_fixing`` that completes configurations deterministically.
"""

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp


def _space_is_gauge_fixed(space: Any) -> bool:
    """
    Return whether a Hilbert-space object looks like a gauge-fixed constrained core.

    A gauge-fixed constrained core is expected to expose:

    - ``gauge_fixing``: an object describing the deterministic gauge-fixing structure, and
    - ``reimpose_gauge_fixing``: a method that reconstructs a full gauge-invariant configuration by
      filling the dependent (``slave``) variables from the independent (``free``) variables.

    This helper performs a lightweight structural check (``hasattr`` only). It does not validate the
    semantics of the attributes and is intended for fast branching in indexing/conversion utilities.

    :param space: Candidate Hilbert-space core or interface object.

    :return: ``True`` if the required attributes are present, otherwise ``False``.
    """

    return hasattr(space, "gauge_fixing") and hasattr(space, "reimpose_gauge_fixing")


def _extract_reduced_free_values(space: Any, states2d: np.ndarray) -> np.ndarray:
    """
    Extract the reduced set of gauge-free variables from full configurations.

    For gauge-fixed constrained cores, a full configuration :math:`\\sigma` of shape ``(B, N)`` can be
    viewed as ``(B, G, E)``, where:

    - :math:`G` is the number of gauge copies (``space.gauge_dimensions``),
    - :math:`E` is the number of edge sites per copy (``space.tiny_size``),
    - :math:`N = G E` is the flattened configuration length.

    The gauge-fixing object ``space.gauge_fixing`` defines a set of *free* edge indices
    ``gauge_fixing.free`` within a single copy (size :math:`F`). The remaining edges are *slaves* and
    are determined deterministically from the free variables by reimposing the gauge fixing.

    This function returns the free variables in a stable, block-concatenated order:

    .. math::

        [\\text{copy}_0\\ (f_0,\\dots,f_{F-1})\\ \\vert\\
          \\text{copy}_1\\ (f_0,\\dots,f_{F-1})\\ \\vert\\ \\cdots\\ \\vert\\
          \\text{copy}_{G-1}\\ (f_0,\\dots,f_{F-1})]

    so that the output has shape ``(B, G*F)``.

    This helper assumes that ``gauge_fixing.free`` is already in a deterministic order; it simply
    indexes the reshaped state accordingly.

    :param space: Gauge-fixed constrained core providing ``gauge_fixing.free``, ``gauge_dimensions``,
      and ``tiny_size``.
    :param states2d: Full configurations as a NumPy array of shape ``(B, N)`` with ``N = G*E``.

    :raises AttributeError: If ``space`` does not provide the required gauge-fixing attributes.
    :raises ValueError: If ``states2d`` cannot be reshaped consistently to ``(B, G, E)`` (e.g. if
      ``N != G*E``).

    :return: Free-variable array of shape ``(B, G*F)`` in the block order described above.
    """

    gf = space.gauge_fixing

    # base-edge indices, sorted by construction
    free = np.asarray(gf.free, dtype=np.int64)
    G = int(space.gauge_dimensions)
    E = int(space.tiny_size)
    F = int(free.shape[0])

    v = states2d.reshape((states2d.shape[0], G, E))

    # (B, G, F)
    free_vals = v[:, :, free]
    return free_vals.reshape((states2d.shape[0], G * F))


def _reconstruct_from_reduced_free_values(
    space: Any, free_vals2d: np.ndarray
) -> "jnp.ndarray":
    """
    Reconstruct full gauge-invariant configurations from reduced free variables.

    Given free variables arranged as ``(B, G*F)``, this function builds a full configuration
    ``sigma`` of shape ``(B, G*E)`` by:

    1. creating a zero-initialised view ``v`` of shape ``(B, G, E)``,
    2. inserting the free values at the edge indices ``gauge_fixing.free`` for each gauge copy, and
    3. calling ``space.reimpose_gauge_fixing`` to deterministically fill the dependent (slave) variables
       and enforce the gauge-fixing conditions.

    The resulting array is returned as a JAX array, matching the space dtype.

    This is the inverse operation of :func:`_extract_reduced_free_values` up to the deterministic
    completion performed by ``reimpose_gauge_fixing``.

    :param space: Gauge-fixed constrained core providing ``gauge_fixing.free``,
      ``reimpose_gauge_fixing``, ``gauge_dimensions``, ``tiny_size``, and dtype metadata.
    :param free_vals2d: Reduced free variables of shape ``(B, G*F)`` in block order
      ``[copy0 | copy1 | ...]``.

    :raises AttributeError: If ``space`` does not provide the required gauge-fixing attributes.
    :raises ValueError: If ``free_vals2d`` cannot be reshaped to ``(B, G, F)`` consistently.
    :raises ValueError: If indices in ``gauge_fixing.free`` are out of range for a copy of size ``E``.
    :raises Exception: Propagates any error raised by ``space.reimpose_gauge_fixing``.

    :return: Full gauge-invariant configurations as a JAX array of shape ``(B, G*E)``.
    """

    gf = space.gauge_fixing
    free = np.asarray(gf.free, dtype=np.int64)
    G = int(space.gauge_dimensions)
    E = int(space.tiny_size)
    B = int(free_vals2d.shape[0])
    F = int(free.shape[0])

    v = np.zeros((B, G, E), dtype=space.allowed_basis_states.dtype)
    v[:, :, free] = free_vals2d.reshape((B, G, F))

    sigma = jnp.asarray(v.reshape((B, G * E)), dtype=space.dtype)

    # reimpose to fill slaves deterministically
    sigma_gi = space.reimpose_gauge_fixing(sigma)
    return sigma_gi


def _extract_reduced_free_values_jax(space: Any, states2d: jax.Array) -> jax.Array:
    gf = space.gauge_fixing
    free = jnp.asarray(gf.free, dtype=jnp.int64)
    G = int(space.gauge_dimensions)
    E = int(space.tiny_size)
    F = int(free.shape[0])

    v = states2d.reshape((states2d.shape[0], G, E))
    free_vals = jnp.take(v, free, axis=2)  # (B, G, F)
    return free_vals.reshape((states2d.shape[0], G * F))


def _reconstruct_from_reduced_free_values_jax(
    space: Any, free_vals2d: jax.Array
) -> jax.Array:
    gf = space.gauge_fixing
    free = jnp.asarray(gf.free, dtype=jnp.int64)
    G = int(space.gauge_dimensions)
    E = int(space.tiny_size)
    B = int(free_vals2d.shape[0])
    F = int(free.shape[0])

    v = jnp.zeros((B, G, E), dtype=space.allowed_basis_states.dtype)
    v = v.at[:, :, free].set(free_vals2d.reshape((B, G, F)))
    sigma = v.reshape((B, G * E)).astype(space.dtype)
    return space.reimpose_gauge_fixing(sigma)
