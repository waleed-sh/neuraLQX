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

from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.proposal import AbstractProposalMove

from .abstract import AbstractHilbertSpace

class DiscreteHilbertSpace(AbstractHilbertSpace):
    constraint: AbstractDiscreteConstraint
    @property
    def local_sizes(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...
    def states_to_local_indices(self, states: Any) -> jax.Array: ...
    def local_indices_to_states(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array: ...
    def local_states_valid(self, states: Any) -> jax.Array: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def constrained(self) -> bool: ...
    @property
    def unconstrained_dimension(self) -> int: ...
    @property
    def n_states(self) -> int: ...
    @property
    def is_indexable(self) -> bool: ...
    def is_valid(self, states: Any) -> jax.Array: ...
    def random_state(
        self,
        key: jax.Array,
        size: int | tuple[int, ...] | None = None,
        *,
        dtype: Any | None = None,
        max_trials: int = 1024,
    ) -> jax.Array: ...
    def propose(
        self, move: AbstractProposalMove, key: jax.Array, states: Any
    ) -> jax.Array: ...
    def states_to_numbers(
        self, states: Any, *, validate: bool = False
    ) -> jax.Array: ...
    def numbers_to_states(self, numbers: Any) -> jax.Array: ...
    def states_to_numbers_python(
        self, states: Any, *, validate: bool = False
    ) -> Any: ...
    def numbers_to_states_python(self, numbers: Any) -> jax.Array: ...
    def all_states(self, *, max_states: int = 1_000_000) -> jax.Array: ...

AbstractDiscreteHilbert = DiscreteHilbertSpace
