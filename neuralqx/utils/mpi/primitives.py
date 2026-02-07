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
MPI collective helpers.

These thin wrappers guarantee identical call signatures whether you run serially, under ``mpiexec``
or with ``export NQX_MPI=0``. In serial mode every function simply returns its input, in parallel
they forward to the relevant ``mpi4py`` collective.
"""

from __future__ import annotations

from typing import Sequence, Any
import numpy as np

from .core import MPI, comm, n_nodes, rank

__all__ = [
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
]


def _as_array(x):
    """
    Cheaply view/convert x as ``numpy.ndarray``
    """

    return np.asarray(x)


#
#
#   arithmetic reductions
def mpi_sum(x, *, out=None):
    arr = _as_array(x)
    if n_nodes > 1:
        if out is None:
            out = arr.copy()
        comm.Allreduce(arr, out, op=MPI.SUM)
        return out
    return arr


def mpi_prod(x, *, out=None):
    arr = _as_array(x)
    if n_nodes > 1:
        if out is None:
            out = arr.copy()
        comm.Allreduce(arr, out, op=MPI.PROD)
        return out
    return arr


def mpi_mean(x, *, out=None):
    return mpi_sum(x, out=out) / n_nodes


def mpi_max(x, *, out=None):
    arr = _as_array(x)
    if n_nodes > 1:
        if out is None:
            out = arr.copy()
        comm.Allreduce(arr, out, op=MPI.MAX)
        return out
    return arr


#
#
#   logical reductions
def mpi_any(x, *, out=None):
    """
    Logical OR across ranks
    """

    arr = _as_array(x).astype(bool)
    if n_nodes > 1:
        if out is None:
            out = arr.copy()
        comm.Allreduce(arr, out, op=MPI.LOR)
        return out.astype(bool)
    return arr


def mpi_all(x, *, out=None):
    """
    Logical AND across ranks
    """

    arr = _as_array(x).astype(bool)
    if n_nodes > 1:
        if out is None:
            out = arr.copy()
        comm.Allreduce(arr, out, op=MPI.LAND)
        return out.astype(bool)
    return arr


#
#
#   point-to-point / gather-scatter
def mpi_bcast(x: Any, *, root: int = 0):
    """
    Broadcast x from root to every rank
    """

    return comm.bcast(x, root=root) if n_nodes > 1 else x


def mpi_allgather(x):
    """
    Gather x from every rank to every rank
    """

    if n_nodes > 1:
        if isinstance(x, np.ndarray):
            out = np.empty((n_nodes,) + x.shape, dtype=x.dtype)
            comm.Allgather(_as_array(x), out)
            return out
        return comm.allgather(x)

    # serial fallback
    return _as_array(x).reshape(1, *x.shape) if isinstance(x, np.ndarray) else (x,)


def mpi_gather(x, *, root: int = 0):
    """
    Gather x from every rank on root
    """

    if n_nodes > 1:
        if isinstance(x, np.ndarray):
            recv = None
            if rank == root:
                recv = np.empty((n_nodes,) + x.shape, dtype=x.dtype)
            comm.Gather(_as_array(x), recv, root=root)
            return recv
        return comm.gather(x, root=root)

    # serial
    return _as_array(x).reshape(1, *x.shape) if isinstance(x, np.ndarray) else (x,)


def mpi_scatter(xs: Sequence[Any], *, root: int = 0):
    """
    Scatter a sequence xs (living only on root) so that each rank receives one slice. Serial mode
    expects len(xs) == 1.
    """

    if n_nodes > 1:
        return comm.scatter(xs if rank == root else None, root=root)
    if len(xs) != 1:
        raise ValueError("Serial scatter expects a length-1 sequence.")
    return xs[0]
