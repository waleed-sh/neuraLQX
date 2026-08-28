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


"""JAX sharding helpers used by neuraLQX subsystems."""

from .arrays import is_distributed_array
from .arrays import is_jax_array
from .arrays import is_sharded_array
from .connected import connected_output_shardings
from .connected import place_connected_outputs
from .connected import place_operator_batch
from .operators import operator_axis_partition_count
from .operators import operator_terms_shardings
from .operators import place_operator_terms
from .parameters import constrain_parameters
from .parameters import parameter_mesh
from .parameters import parameter_sharding_enabled
from .parameters import place_parameters
from .parameters import replicated_parameter_sharding
from .samples import can_shard_samples
from .samples import make_sample_zeros
from .samples import place_sample_batch
from .samples import sample_axis_name
from .samples import sample_batch_sharding
from .samples import sample_mesh
from .samples import sample_partition_count
from .samples import sample_partition_spec
from .samples import sample_sharding_enabled
from .samples import shard_sample_keys

__all__ = [
    "can_shard_samples",
    "connected_output_shardings",
    "constrain_parameters",
    "is_distributed_array",
    "is_jax_array",
    "is_sharded_array",
    "make_sample_zeros",
    "operator_axis_partition_count",
    "operator_terms_shardings",
    "parameter_mesh",
    "parameter_sharding_enabled",
    "place_connected_outputs",
    "place_operator_batch",
    "place_operator_terms",
    "place_parameters",
    "place_sample_batch",
    "replicated_parameter_sharding",
    "sample_axis_name",
    "sample_batch_sharding",
    "sample_mesh",
    "sample_partition_count",
    "sample_partition_spec",
    "sample_sharding_enabled",
    "shard_sample_keys",
]
