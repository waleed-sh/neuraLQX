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
Public interface for the neuraLQX MPI helpers.

Importing anything from this sub-package is always safe. When MPI support is disabled via the
environment variables ``NQX_MPI`` or ``NQX_MPI_CUDA``, or when ``mpi4py`` is missing, every
collective turns into a fast no-op and the code continues to run only serially.

Examples
--------
Disable MPI globally::

    export NQX_MPI=0

Disable CUDA-aware MPI::

    export NQX_MPI_CUDA=0
"""

import os
from neuralqx import cfg

if cfg.get("MPI") or cfg.get("MPI_CUDA"):
    if not int(os.getenv("NQX_MPI_TESTS", 0)):
        from neuralqx.utils.deprecation import deprecated_module

        deprecated_module(
            __name__,
            "neuraLQX is stopping MPI support as of version 1.1.0 in favour of JAX's sharding and to support the "
            "latest NetKet versions. This means that neuraLQX will be parallelisable by default only on GPU and only using "
            "JAX (no MPI). If you need MPI support, use a version of neuraLQX which is < 1.1.0. Note that in this case, "
            "new features will not be available.",
        )

from .core import (
    available,
    MPI,
    comm,
    n_nodes,
    rank,
    node_number,
    is_global_master,
    is_master,
    barrier,
    print0,
    hostname,
    all_hostnames,
    local_ranks,
    ranks_per_node,
    detect_cpus_per_task,
)

from .primitives import (
    mpi_sum,
    mpi_prod,
    mpi_mean,
    mpi_any,
    mpi_all,
    mpi_max,
    mpi_bcast,
    mpi_allgather,
    mpi_gather,
    mpi_scatter,
)

from .helpers import (
    tqdm,
    check_mpi,
    get_mpi_info_dict,
)

__all__ = [
    # low-level
    "available",
    "MPI",
    "comm",
    "n_nodes",
    "rank",
    "node_number",
    "hostname",
    "all_hostnames",
    "local_ranks",
    "ranks_per_node",
    "detect_cpus_per_task",
    # helpers
    "is_global_master",
    "is_master",
    "barrier",
    "print0",
    # collectives
    "mpi_sum",
    "mpi_prod",
    "mpi_mean",
    "mpi_any",
    "mpi_all",
    "mpi_max",
    "mpi_bcast",
    "mpi_allgather",
    "mpi_gather",
    "mpi_scatter",
    # helpers
    "tqdm",
    "check_mpi",
    "get_mpi_info_dict",
]
