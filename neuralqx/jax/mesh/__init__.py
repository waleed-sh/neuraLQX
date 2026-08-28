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


"""Centralized JAX mesh construction from neuraLQX configuration."""

from .config import MeshConfig
from .config import active_logical_axes
from .config import configured_axis_names
from .config import configured_axis_sizes
from .config import logical_axis_name
from .create import create_device_mesh
from .create import infer_axis_sizes
from .sharding import named_sharding
from .sharding import operator_batch_sharding
from .sharding import operator_partition_axis_names

__all__ = [
    "MeshConfig",
    "active_logical_axes",
    "configured_axis_names",
    "configured_axis_sizes",
    "create_device_mesh",
    "infer_axis_sizes",
    "logical_axis_name",
    "named_sharding",
    "operator_batch_sharding",
    "operator_partition_axis_names",
]
