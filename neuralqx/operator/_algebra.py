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


"""Small helpers shared by operator algebra wrappers."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

import jax


def is_scalar_like(value: Any) -> bool:
    """Return whether ``value`` should be treated as an operator scalar."""
    if isinstance(value, numbers.Number | np.number):
        return True
    if isinstance(value, jax.Array | np.ndarray):
        return value.shape == ()
    return False


def is_known_real_scalar(value: Any) -> bool:
    """Return ``True`` only when a scalar is concretely known to be real."""
    try:
        array = np.asarray(value)
    except Exception:
        return False
    if array.shape != ():
        return False
    try:
        return bool(np.all(np.isreal(array)))
    except Exception:
        return False


def is_zero_scalar_like(value: Any) -> bool:
    """Return whether ``value`` is a concretely known scalar zero."""
    if not is_scalar_like(value):
        return False
    try:
        return bool(np.asarray(value) == 0)
    except Exception:
        return False


__all__ = ["is_known_real_scalar", "is_scalar_like", "is_zero_scalar_like"]
