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

import jax
import numpy as np

from neuralqx.utils.struct import Struct

class AbstractLocalSpace(Struct):
    @property
    def local_size(self) -> int: ...
    @property
    def dtype(self) -> np.dtype: ...
    def all_values(self, *, dtype: Any | None = None) -> jax.Array: ...
    def indices_to_values(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array: ...
    def values_to_indices(self, values: Any) -> jax.Array: ...
    def contains(self, values: Any) -> jax.Array: ...
    def random_indices(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array: ...
    def random_values(
        self,
        key: jax.Array,
        shape: tuple[int, ...],
        *,
        dtype: Any | None = None,
    ) -> jax.Array: ...
