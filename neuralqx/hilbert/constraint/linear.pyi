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

from collections.abc import Sequence
from typing import Any
from typing import Self

import jax.numpy as jnp

from .abstract import AbstractDiscreteConstraint

class LinearConstraint(AbstractDiscreteConstraint):
    sites: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[int | float, ...], ...]
    targets: tuple[int | float, ...]
    atol: float
    def __init__(
        self,
        sites: Sequence[Sequence[int]],
        weights: Sequence[Sequence[int | float]],
        targets: Sequence[int | float],
        atol: float = 0.0,
    ) -> None: ...
    @classmethod
    def sum(cls, sites: Sequence[int], target: int | float) -> Self: ...
    def validate_hilbert(self, hilbert: Any) -> None: ...
    def __call__(self, states: Any) -> jnp.ndarray: ...
