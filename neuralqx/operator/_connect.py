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


"""JIT entry points for computational operators."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
from neuralqx.jax.streaming import stream_flattened_kernel


@jax.jit
def direct_connected_components(
    operator: Any, states_2d: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Run an operator kernel on a rank-2 state batch."""
    return operator._get_conn_padded_batch_kernel(states_2d)


@partial(jax.jit, static_argnames=("chunk_size",))
def streamed_connected_components(
    operator: Any,
    states_2d: jax.Array,
    *,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Run an operator kernel over a rank-2 state batch in fixed chunks."""
    return stream_flattened_kernel(
        lambda chunk: operator._get_conn_padded_batch_kernel(chunk),
        states_2d,
        chunk_size=chunk_size,
        max_conn_size=int(operator.max_conn_size),
        matrix_element_dtype=operator.dtype,
    )


__all__ = ["direct_connected_components", "streamed_connected_components"]
