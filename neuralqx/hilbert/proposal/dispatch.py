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


def propose(
    space: DiscreteHilbertSpace,
    move: Any,
    key: jax.Array,
    states: Any,
) -> jax.Array:
    """Applies a proposal move using neuraLQX-local multiple dispatch.

    Args:
        space: Discrete Hilbert space whose states are being updated.
        move: Proposal move descriptor.
        key: JAX pseudo-random key.
        states: Input state array.

    Returns:
        Proposed state array with the same shape as ``states``.
    """
    raise NotImplementedError


__all__ = ["propose"]
