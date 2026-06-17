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

from neuralqx.hilbert.space import DiscreteHilbertSpace

def states_to_numbers(
    space: DiscreteHilbertSpace,
    states: Any,
    *,
    validate: bool = False,
) -> jax.Array: ...
def numbers_to_states(space: DiscreteHilbertSpace, numbers: Any) -> jax.Array: ...
def states_to_numbers_python(
    space: DiscreteHilbertSpace,
    states: Any,
    *,
    validate: bool = False,
) -> Any: ...
def numbers_to_states_python(
    space: DiscreteHilbertSpace, numbers: Any
) -> jax.Array: ...
