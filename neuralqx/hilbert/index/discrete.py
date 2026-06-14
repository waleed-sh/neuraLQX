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

from neuralqx.hilbert.space.discrete import DiscreteHilbertSpace


def states_to_numbers(
    space: DiscreteHilbertSpace, states: Any, *, validate: bool = False
) -> jax.Array:
    """Ranks unconstrained states in rightmost-fast lexicographic order.

    Args:
        space: Indexable unconstrained discrete Hilbert space.
        states: State array whose trailing dimension equals ``space.size``.
        validate: Whether to check membership before ranking.

    Returns:
        JAX integer labels matching the input batch shape.
    """
    raise NotImplementedError


def numbers_to_states(space: DiscreteHilbertSpace, numbers: Any) -> jax.Array:
    """Unranks integer labels for indexable unconstrained spaces.

    Args:
        space: Indexable unconstrained discrete Hilbert space.
        numbers: Integer labels to decode.

    Returns:
        State array with trailing dimension ``space.size``.
    """
    raise NotImplementedError


def states_to_numbers_python(
    space: DiscreteHilbertSpace, states: Any, *, validate: bool = False
) -> Any:
    """Ranks states using arbitrary-size Python integers.

    Args:
        space: Unconstrained discrete Hilbert space.
        states: State array whose trailing dimension equals ``space.size``.
        validate: Whether to check membership before ranking.

    Returns:
        Python integer for one state or an object array for batched states.
    """
    raise NotImplementedError


def numbers_to_states_python(space: DiscreteHilbertSpace, numbers: Any) -> jax.Array:
    """Unranks arbitrary-size Python integers without product materialization.

    Args:
        space: Unconstrained discrete Hilbert space.
        numbers: Python integer labels or an array of labels.

    Returns:
        State array with trailing dimension ``space.size``.
    """
    raise NotImplementedError


__all__ = [
    "numbers_to_states",
    "numbers_to_states_python",
    "states_to_numbers",
    "states_to_numbers_python",
]
