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

import jax
import jax.numpy as jnp

import numpy as np


def _as_numpy(x: Any) -> np.ndarray:
    # works for NumPy arrays, JAX arrays, python lists/tuples
    return np.asarray(x)


def _ensure_2d_states(states: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Normalise a state array to a 2D ``(B, N)`` representation.

    Many indexing and enumeration routines operate on batches of configurations. This helper
    standardises the incoming array so downstream code can assume a batch dimension is present.

    Accepted input shapes:

    - ``(N,)``: a single configuration of length :math:`N`.
    - ``(B, N)``: a batch of :math:`B` configurations.
    - ``(..., N)``: any higher-rank array whose final axis has length :math:`N`; all leading axes
      are flattened into the batch dimension.

    The returned array always has shape ``(B, N)`` for some :math:`B \\ge 1`. A boolean flag
    indicates whether the input was a single configuration (shape ``(N,)``), which is useful for
    restoring shapes later.

    :param states: Array of configurations with final axis length :math:`N`.

    :return: A tuple ``(states2d, was_single)`` where

      - ``states2d`` has shape ``(B, N)``.
      - ``was_single`` is ``True`` iff the input had shape ``(N,)``.
    """

    if states.ndim == 1:
        return states[None, :], True
    if states.ndim == 2:
        return states, False
    # flatten leading dims into batch
    return states.reshape((-1, states.shape[-1])), False


def _restore_states_shape(
    states2d: np.ndarray, orig_shape: Tuple[int, ...]
) -> np.ndarray:
    """
    Restore a batch-normalised state array to its original shape.

    This is the inverse utility to :func:`_ensure_2d_states`. After processing states in a
    batch-normalised form ``(B, N)``, this function reshapes the result back to the original input
    shape recorded by the caller.

    - If ``orig_shape == (N,)``, the output is reshaped back to ``(N,)``.
    - Otherwise the output is reshaped to ``orig_shape`` (typically ``(B, N)`` or ``(..., N)``).

    :param states2d: Array in batch form with shape ``(B, N)``.
    :param orig_shape: Original shape of the input states.

    :return: ``states2d`` reshaped to ``orig_shape``.
    """
    if len(orig_shape) == 1:
        return states2d.reshape((orig_shape[0],))
    return states2d.reshape(orig_shape)


def _ensure_1d_numbers(nums: Any) -> Tuple[np.ndarray, bool]:
    """
    Normalise integer-like labels to a 1D NumPy array.

    Indexing utilities often accept either a single label (scalar) or a batch of labels. This
    helper coerces the input into a one-dimensional NumPy array so downstream code can treat all
    cases uniformly.

    Accepted input forms include:

    - Python integers,
    - NumPy (or JAX) scalars,
    - list/tuple/array-like sequences of numbers,
    - higher-rank arrays, which are flattened to one dimension.

    A boolean flag indicates whether the original input was scalar, which is useful when deciding
    whether to return a scalar or an array to the user.

    :param nums: Scalar or array-like collection of labels.

    :return: A tuple ``(nums1d, was_scalar)`` where

      - ``nums1d`` is a 1D NumPy array of shape ``(K,)``.
      - ``was_scalar`` is ``True`` iff the input was a scalar.
    """
    arr = _as_numpy(nums)
    if arr.ndim == 0:
        return arr.reshape((1,)), True
    if arr.ndim == 1:
        return arr, False
    return arr.reshape((-1,)), False


def _ensure_2d_states_jax(states: jax.Array) -> Tuple[jax.Array, bool]:
    """Normalise a state array to shape (B, N). Returns (states2d, was_single)."""
    if states.ndim == 1:
        return states[jnp.newaxis, :], True
    if states.ndim == 2:
        return states, False
    return states.reshape((-1, states.shape[-1])), False


def _ensure_1d_numbers_jax(numbers: Any) -> Tuple[jax.Array, bool, Tuple[int, ...]]:
    """Normalise labels to a 1D JAX array. Returns (nums1d, was_scalar, orig_shape)."""
    arr = jnp.asarray(numbers)
    orig_shape = arr.shape
    if arr.ndim == 0:
        return arr.reshape((1,)), True, orig_shape
    if arr.ndim == 1:
        return arr, False, orig_shape
    return arr.reshape((-1,)), False, orig_shape
