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

"""
JAX-process distributed helpers for neuraLQX.

This package is the migration target for code that previously depended on
``neuralqx.utils.mpi``.
"""

from .core import RuntimeInfo
from .core import available
from .core import backend
from .core import rank
from .core import n_nodes
from .core import node_number
from .core import local_rank
from .core import runtime_info
from .core import process_index
from .core import process_count
from .core import is_global_master
from .core import is_master
from .core import barrier
from .core import allgather
from .core import bcast
from .core import mpi_bcast
from .core import mpi_any
from .core import mpi_sum_jax
from .core import mpi_mean_jax
from .core import print0
from .core import detect_cpus_per_task
from .core import check_distributed
from .core import get_distributed_info_dict

from .io import block_until_ready_tree
from .io import device_get_tree
from .io import safe_replicate_for_io

__all__ = [
    "RuntimeInfo",
    "available",
    "backend",
    "rank",
    "n_nodes",
    "node_number",
    "local_rank",
    "runtime_info",
    "process_index",
    "process_count",
    "is_global_master",
    "is_master",
    "barrier",
    "allgather",
    "bcast",
    "mpi_bcast",
    "mpi_any",
    "mpi_sum_jax",
    "mpi_mean_jax",
    "print0",
    "detect_cpus_per_task",
    "check_distributed",
    "get_distributed_info_dict",
    "block_until_ready_tree",
    "device_get_tree",
    "safe_replicate_for_io",
]
