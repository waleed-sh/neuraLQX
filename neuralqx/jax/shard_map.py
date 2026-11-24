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
        from neuralqx.jax.tensor_parallel import network_sharding_enabled
        from neuralqx.jax.tensor_parallel import tensor_axis_name

        # Tensor axes retain global shapes and compiler-managed collectives.
        # Existing in/out specs describe only the manually partitioned work.
        manual_axes = (
            set(mesh.axis_names) - {tensor_axis_name()}
            if network_sharding_enabled() and tensor_axis_name() in mesh.axis_names
            else set()
        )
        return public_shard_map(
            fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            axis_names=manual_axes,
        )

    from jax.experimental.shard_map import shard_map as experimental_shard_map

    return experimental_shard_map(fn, mesh, in_specs, out_specs)


def varying_over(tree: Any, axes: str | tuple[str, ...]) -> Any:
    """Give replicated inputs independent cotangents on manual work axes.

    Use this before control flow whose branch varies between work shards.
    The transpose then sums contributions once when returning to a replicated
    input. JAX 0.9 exposes this operation as ``pvary`` and later versions expose
    it through ``pcast``.
    """
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(tree, axes, to="varying")
    return jax.lax.pvary(tree, axes)


__all__ = ["shard_map", "varying_over"]
