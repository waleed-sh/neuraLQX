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


from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from functools import lru_cache

from typing import Optional
from typing import Tuple


@dataclass(frozen=True)
class RankInfo:
    rank: int
    size: int
    local_rank: int
    hostname: str
    pid: int
    backend: str  # "mpi", "jax", or "serial"


def _get_env_int(keys, default: int = 0) -> int:
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return default


def _detect_mpi() -> Optional[Tuple[int, int]]:
    try:
        from mpi4py import MPI  # type: ignore

        if not MPI.Is_initialized():
            return None
        comm = MPI.COMM_WORLD
        return int(comm.Get_rank()), int(comm.Get_size())
    except Exception:
        return None


def _detect_jax_process() -> Optional[Tuple[int, int]]:
    try:
        import jax  # type: ignore

        return int(jax.process_index()), int(jax.process_count())
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_rank_info() -> RankInfo:
    mpi = _detect_mpi()
    if mpi is not None:
        rank, size = mpi
        backend = "mpi"
    else:
        jaxp = _detect_jax_process()
        if jaxp is not None and jaxp[1] > 1:
            rank, size = jaxp
            backend = "jax"
        else:
            rank, size = 0, 1
            backend = "serial"

    local_rank = _get_env_int(
        [
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
        ],
        default=rank if size == 1 else 0,
    )

    hostname = socket.gethostname()
    pid = os.getpid()
    return RankInfo(
        rank=rank,
        size=size,
        local_rank=local_rank,
        hostname=hostname,
        pid=pid,
        backend=backend,
    )
