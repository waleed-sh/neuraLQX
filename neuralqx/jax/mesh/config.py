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


"""Config views for JAX mesh construction."""

from __future__ import annotations

from dataclasses import dataclass

from neuralqx.config import config

LOGICAL_AXIS_ORDER = ("samples", "parameters", "operators")


@dataclass(frozen=True, slots=True)
class MeshConfig:
    """Resolved logical mesh configuration."""

    active_axes: tuple[str, ...]
    axis_names: dict[str, str]
    axis_sizes: dict[str, int | None]


def active_logical_axes() -> tuple[str, ...]:
    """Return logical mesh axes enabled by current neuraLQX config."""
    enabled = {
        "samples": bool(config.get("sample_sharding_enabled")),
        "parameters": bool(config.get("parameter_sharding_enabled")),
        "operators": bool(config.get("operator_sharding_enabled")),
    }
    return tuple(axis for axis in LOGICAL_AXIS_ORDER if enabled[axis])


def configured_axis_names() -> dict[str, str]:
    """Return configured physical mesh names for each logical axis."""
    return {
        "samples": str(config.get("mesh_sample_axis_name")),
        "parameters": str(config.get("mesh_parameter_axis_name")),
        "operators": str(config.get("mesh_operator_axis_name")),
    }


def configured_axis_sizes() -> dict[str, int | None]:
    """Return optional configured mesh sizes for each logical axis."""
    return {
        "samples": config.get("mesh_sample_axis_size"),
        "parameters": config.get("mesh_parameter_axis_size"),
        "operators": config.get("mesh_operator_axis_size"),
    }


def mesh_config() -> MeshConfig:
    """Return the currently configured logical mesh."""
    return MeshConfig(
        active_axes=active_logical_axes(),
        axis_names=configured_axis_names(),
        axis_sizes=configured_axis_sizes(),
    )


def logical_axis_name(axis: str) -> str:
    """Map a logical axis name to the configured JAX mesh axis name."""
    names = configured_axis_names()
    try:
        return names[axis]
    except KeyError as exc:
        raise ValueError(f"Unknown logical mesh axis {axis!r}.") from exc


__all__ = [
    "LOGICAL_AXIS_ORDER",
    "MeshConfig",
    "active_logical_axes",
    "configured_axis_names",
    "configured_axis_sizes",
    "logical_axis_name",
    "mesh_config",
]
