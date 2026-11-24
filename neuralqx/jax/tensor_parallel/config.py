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


"""Startup policy for compiler-partitioned neural networks."""

from neuralqx.config import config


def network_sharding_enabled() -> bool:
    """Whether tensor parallelism is enabled by the startup configuration."""
    return bool(config.get("network_sharding_enabled"))


def tensor_axis_name() -> str:
    """Return the configured physical tensor mesh axis name."""
    return str(config.get("mesh_tensor_axis_name"))


__all__ = ["network_sharding_enabled", "tensor_axis_name"]
