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


"""JAX-oriented type aliases."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TypeAlias

import jax
import numpy as np

Array: TypeAlias = jax.Array
"""Runtime JAX array class used for array annotations and ``isinstance`` checks."""

Key: TypeAlias = int | np.integer | jax.Array | np.ndarray | Sequence[int]
"""Input accepted where a JAX PRNG key or deterministic seed is expected."""

ConnectedKernel: TypeAlias = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
"""Single-state connected-component kernel returning states and matrix elements."""

ConnectedStateKernel: TypeAlias = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
"""Vectorization-compatible connected-component kernel for one state."""

__all__ = [
    "Array",
    "ConnectedKernel",
    "ConnectedStateKernel",
    "Key",
]
