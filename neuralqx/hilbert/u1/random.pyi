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

def u1_random_state_jit(
    space: Any,
    key: jax.Array,
    size: int | tuple[int, ...] | None,
    dtype: Any,
) -> jax.Array: ...
def reimpose_gauge_fixing_jit(space: Any, states: jax.Array) -> jax.Array: ...
