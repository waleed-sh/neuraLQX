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


"""Compatibility wrapper for JAX shard-map execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax


def shard_map(
    fn: Callable[..., Any],
    *,
    mesh: Any,
    in_specs: Any,
    out_specs: Any,
) -> Callable[..., Any]:
    """Return a shard-mapped function using the JAX 0.9+ public API when present."""
    public_shard_map = getattr(jax, "shard_map", None)
    if public_shard_map is not None:
        return public_shard_map(
            fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )

    from jax.experimental.shard_map import shard_map as experimental_shard_map

    return experimental_shard_map(fn, mesh, in_specs, out_specs)


__all__ = ["shard_map"]
