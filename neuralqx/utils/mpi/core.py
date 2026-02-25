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
Core MPI initialisation for neuraLQX.

This file is imported exactly once (singleton-style) and exposes the following MPI related
variables:

    - available: True when MPI is both requested (export NQX_MPI=1) and mpi4py could be imported
    - MPI: either the real mpi4py module or a lightweight stub
    - comm: duplicated COMM_WORLD (or a fake serial communicator)
    - n_nodes: an int, size of the communicator (always ≥1)
    - rank: an int, the rank in comm
    - print0: a print that only talks on rank-0
"""

from __future__ import annotations

import os
import types
import warnings
import logging
from collections import Counter
from textwrap import dedent

from ...configs import cfg
from ..errors import MPIUnavailableWarning, NetKetMPIUnavailableWarning
from ...debug import event

# figure out whether the user requested MPI or not
_MPI_REQUESTED: bool = bool(cfg.get("MPI")) or bool(cfg.get("MPI_CUDA"))
_mpi4py_available: bool = False

# if the user requested MPI, check for mpi4py availability
if _MPI_REQUESTED:
    try:
        # found :)
        from mpi4py import MPI as _REAL_MPI

        _mpi4py_available = True
    except ModuleNotFoundError:
        # not found :(
        _mpi4py_available = False

# if requested but not available, then warn and fall back
if _MPI_REQUESTED and not _mpi4py_available:
    MPIUnavailableWarning()

# either expose the real mpi4py interface or a minimal stub
if _mpi4py_available:
    # use real mpi4py
    MPI = _REAL_MPI

    # try to adopt NetKet's communicator/rank if NetKet MPI is active
    _USE_NETKET_MPI = False
    try:
        if os.environ.get("NETKET_MPI", "0") == "1":
            # import NetKet's MPI surface
            # these are module-level constants in netket.utils.mpi
            from netket.utils.mpi import (
                MPI_py_comm as _NK_COMM,
                MPI_jax_comm as _NK_JAX_COMM,
                rank as _NK_RANK,
                n_nodes as _NK_SIZE,
            )

            # sanity check: NetKet might be present but MPI disabled there
            if _NK_COMM is not None:
                comm = _NK_COMM
                comm_jax = _NK_JAX_COMM
                rank = int(_NK_RANK)
                n_nodes = int(_NK_SIZE)
                _USE_NETKET_MPI = True

                event(msg="NETKET_MPI_ENABLED", tag="NQX:MPI", level=logging.INFO)

    except Exception:
        # fall back below
        _USE_NETKET_MPI = False

    if not _USE_NETKET_MPI:
        # fallback to our own communicator
        NetKetMPIUnavailableWarning()

        # NOTE: Dup() is safer than Clone() with some CUDA-aware MPI stacks
        comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
        comm_jax = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
        n_nodes = comm.Get_size()
        rank = comm.Get_rank()

        event(msg="NETKET_MPI_NOT_FOUND", tag="NQX:MPI", level=logging.INFO)

    node_number = rank
    available = True

    # autorank using netket
    if bool(cfg.get("MPI_CUDA")):
        from netket.utils.mpi.gpu_autorank_util import autoset_default_gpu

        autoset_default_gpu(comm)
        del autoset_default_gpu

        event(msg="AUTO_SET_GPUS", tag="NQX:MPI_CUDA", level=logging.INFO)
else:
    # not available, we will do all the following to streamline MPI usage across neuraLQX

    event(msg="MPI4PY_NOT_AVAILABLE", tag="NQX:MPI", level=logging.INFO)

    # a serial stub
    FakeMPI = types.ModuleType("FakeMPI", doc="stub mpi4py replacement")

    class _FakeRequest:  # pylint: disable=too-few-public-methods
        """
        Non-blocking collectives are unsupported in stub mode
        """

        def Wait(self): ...
        def Test(self):
            return True, None

    class _FakeComm:  # pylint: disable=too-few-public-methods
        """
        Replacement for `COMM_WORLD` in serial mode
        """

        # structural subset of mpi4py's Intracomm
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        # collectives
        def Barrier(self):
            return None

        def Bcast(self, obj, root=0):
            return obj

        def Allreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[...] = sendbuf

        def Allgather(self, sendbuf, recvbuf):
            recvbuf[...] = sendbuf

        def Gather(self, sendbuf, recvbuf, root=0):
            recvbuf[...] = sendbuf

        def Scatter(self, sendbuf, recvbuf, root=0):
            recvbuf[...] = sendbuf[0]

        def Reduce(self, sendbuf, recvbuf, op=None, root=0):
            recvbuf[...] = sendbuf

        def allgather(self, sendbuf, recvbuf):
            recvbuf[...] = sendbuf

        # misc
        def Clone(self):
            return self

        def Split(self, color=0, key=0):
            return self

        def Create(self, group):
            return self

    FakeMPI.COMM_WORLD = _FakeComm()  # type: ignore[attr-defined]
    FakeMPI.IN_PLACE = None
    FakeMPI.SUM = FakeMPI.PROD = FakeMPI.MAX = None
    FakeMPI.LOR = FakeMPI.LAND = None
    FakeMPI.Request = _FakeRequest  # type: ignore[attr-defined]

    # initialise the fake MPI
    MPI = FakeMPI  # type: ignore[assignment]
    comm = FakeMPI.COMM_WORLD  # type: ignore[assignment]
    comm_jax = FakeMPI.COMM_WORLD  # type: ignore[assignment]
    n_nodes: int = 1
    rank: int = 0
    node_number: int = 0

    # set the availability flag to False
    available: bool = False

#
#
#   helper utilities


def _env_global_rank_str() -> str:
    # prefer launcher-provided global rank, fall back to our MPI rank
    return (
        os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("SLURM_PROCID")
        or os.environ.get("MV2_COMM_WORLD_RANK")
        or str(rank)
    )


def is_global_master() -> bool:
    try:
        return int(_env_global_rank_str()) == 0
    except Exception:
        return rank == 0


def is_master() -> bool:
    """
    Return ``True`` on rank-0, irrespective of serial/parallel mode
    """

    # dev: when we support GPU/JAX sharding, we have to check jax.process_index() == 0 as well
    return rank == 0


def barrier() -> None:
    """
    A global synchronisation barrier (no-op in serial)
    """

    if n_nodes > 1:
        comm.Barrier()


def print0(*args, **kwargs):
    """
    `print` only from rank-0
    """

    if is_master():
        print(*args, **kwargs)


def hostname():
    """
    Returns a normalized host name of the current rank.
    Strips any OpenMPI rank suffixes like '.local#0'.
    """

    name = MPI.Get_processor_name()
    # strip after '#' (OpenMPI-specific suffix)
    # dev: should we really? can they share name but differ in numbering on clusters?
    if "#" in name:
        name = name.split("#")[0]
    return name


def all_hostnames():
    """
    Return all hostnames for all ranks
    """
    return comm.allgather(hostname())


def ranks_per_node():
    """
    Return the number of ranks per node
    """

    return Counter(all_hostnames())


def local_ranks():
    """
    Returns the number of local ranks (per node)
    """
    return ranks_per_node()[hostname()]


def detect_cpus_per_task() -> int:
    """
    Determine the number of CPUs per MPI task.
    Priority:
        1. SLURM_CPUS_PER_TASK
        2. OMP_NUM_THREADS
        3. Default: 1
    """

    if "SLURM_CPUS_PER_TASK" in os.environ:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads and omp_threads.isdigit():
        return max(1, int(omp_threads))

    # default fallback
    return 1


#
#
#   explicit public surface


__all__ = [
    "MPI",
    "comm",
    "comm_jax",
    "available",
    "n_nodes",
    "rank",
    "node_number",
    "is_master",
    "barrier",
    "print0",
    "hostname",
    "all_hostnames",
    "ranks_per_node",
    "local_ranks",
    "detect_cpus_per_task",
]

# clean-up internals
del types, warnings, dedent, logging, cfg, _MPI_REQUESTED, _mpi4py_available
