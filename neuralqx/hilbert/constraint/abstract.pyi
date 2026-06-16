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

import jax.numpy as jnp

from neuralqx.utils.struct import Struct

class AbstractConstraint(Struct):
    def __call__(self, states: Any) -> jnp.ndarray: ...
    def dimension(self, hilbert: Any) -> int | None: ...
    def __and__(self, other: AbstractConstraint) -> AbstractConstraint: ...
    def __or__(self, other: AbstractConstraint) -> AbstractConstraint: ...
    def __invert__(self) -> AbstractConstraint: ...

class AbstractDiscreteConstraint(AbstractConstraint):
    def validate_hilbert(self, hilbert: Any) -> None: ...
