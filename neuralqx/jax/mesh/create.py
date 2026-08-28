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


"""Device mesh creation and size inference."""

from __future__ import annotations

from collections.abc import Sequence
from math import isqrt
from typing import Any

import numpy as np

import jax
from jax.sharding import Mesh

from .config import active_logical_axes
from .config import configured_axis_names
from .config import configured_axis_sizes


def create_device_mesh(
    *,
    active_axes: Sequence[str] | None = None,
    axis_names: dict[str, str] | None = None,
    axis_sizes: dict[str, int | None] | None = None,
    devices: Sequence[Any] | None = None,
) -> Mesh | None:
    """Create a JAX ``Mesh`` from neuraLQX config.

    Returns ``None`` when no logical mesh axes are enabled. The inferred mesh
    consumes exactly the supplied devices, so accidental partial-device meshes
    do not silently leave accelerators idle.
    """
    logical_axes = tuple(active_logical_axes() if active_axes is None else active_axes)
    if not logical_axes:
        return None

    devs = tuple(jax.devices() if devices is None else devices)
    if not devs:
        return None

    names = configured_axis_names() if axis_names is None else axis_names
    sizes = infer_axis_sizes(
        logical_axes,
        configured_axis_sizes() if axis_sizes is None else axis_sizes,
        device_count=len(devs),
    )
    mesh_shape = tuple(sizes[axis] for axis in logical_axes)
    mesh_names = tuple(names[axis] for axis in logical_axes)
    mesh_devices = np.asarray(devs, dtype=object).reshape(mesh_shape)
    return Mesh(mesh_devices, mesh_names)


def infer_axis_sizes(
    active_axes: Sequence[str],
    axis_sizes: dict[str, int | None],
    *,
    device_count: int,
) -> dict[str, int]:
    """Infer mesh sizes for active logical axes.

    Explicit config sizes are honored exactly. Remaining devices are assigned
    to unspecified axes. When the operator axis is active and unspecified, the
    heuristic gives it a factor close to ``sqrt(remaining_devices)`` so that
    sample and operator work can both receive parallelism by default.
    """
    axes = tuple(active_axes)
    if device_count <= 0:
        raise ValueError("device_count must be positive.")
    if not axes:
        return {}

    resolved: dict[str, int] = {}
    unspecified: list[str] = []
    explicit_product = 1
    for axis in axes:
        value = axis_sizes.get(axis)
        if value is None:
            unspecified.append(axis)
            continue
        size = int(value)
        if size <= 0:
            raise ValueError(f"Mesh axis size for {axis!r} must be positive.")
        resolved[axis] = size
        explicit_product *= size

    if device_count % explicit_product != 0:
        raise ValueError(
            "Explicit mesh axis sizes must divide the available device count."
        )
    remaining = device_count // explicit_product

    for axis in unspecified:
        resolved[axis] = 1

    if unspecified:
        if "operators" in unspecified and len(unspecified) > 1:
            op_size = _largest_factor_at_most(remaining, isqrt(remaining))
            resolved["operators"] = op_size
            remaining //= op_size
            unspecified.remove("operators")

        target = "samples" if "samples" in unspecified else unspecified[0]
        resolved[target] *= remaining

    product = 1
    for axis in axes:
        product *= resolved[axis]
    if product != device_count:
        raise ValueError(
            f"Inferred mesh shape product {product} does not match device count {device_count}."
        )
    return resolved


def _largest_factor_at_most(value: int, limit: int) -> int:
    if value <= 1:
        return 1
    limit = max(1, limit)
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


__all__ = ["create_device_mesh", "infer_axis_sizes"]
