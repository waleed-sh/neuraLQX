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

import jax.numpy as jnp

from netket.utils.mpi.primitives import promote_to_pytree, Token
from netket.utils.types import Array

from .core import MPI, comm, comm_jax, n_nodes, rank

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
    # JAX variants
    "mpi_sum_jax",
    "mpi_prod_jax",
    "mpi_mean_jax",
    "mpi_max_jax",
    "mpi_any_jax",
    "mpi_all_jax",
    "mpi_bcast_jax",
    "mpi_allgather_jax",
    "mpi_gather_jax",
    "mpi_scatter_jax",
]


def _as_array(x):
    """
    Cheaply view/convert x as ``numpy.ndarray``
    """

    return np.asarray(x)


#
#
#   arithmetic reductions
def mpi_sum(x, *, out=None, communicator=None):
    if communicator is None:
        communicator = comm

    arr = _as_array(x)

    if n_nodes <= 1:
        if out is None:
            return arr
        out[...] = arr
        return out

    if out is None:
        # By default, we do not mutate input
        out = arr.copy()

    if out is arr:
        communicator.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.SUM)
    else:
        communicator.Allreduce(arr, out, op=MPI.SUM)

    return out


def mpi_prod(x, *, out=None, communicator=None):
    if communicator is None:
        communicator = comm

    arr = _as_array(x)

    if n_nodes <= 1:
        if out is None:
            return arr
        out[...] = arr
        return out

    if out is None:
        # By default, do not mutate input
        out = arr.copy()

    if out is arr:
        communicator.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.PROD)
    else:
        communicator.Allreduce(arr, out, op=MPI.PROD)

    return out


def mpi_mean(x, *, out=None, communicator=None):
    return mpi_sum(x, out=out, communicator=communicator) / n_nodes


def mpi_max(x, *, out=None, communicator=None):
    if communicator is None:
        communicator = comm

    arr = _as_array(x)

    if n_nodes <= 1:
        if out is None:
            return arr
        out[...] = arr
        return out

    if out is None:
        # By default, do not mutate input
        out = arr.copy()

    if out is arr:
        communicator.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.MAX)
    else:
        communicator.Allreduce(arr, out, op=MPI.MAX)

    return out


#
#
#   logical reductions
def mpi_any(x, *, out=None, communicator=None):
    """
    Logical OR across ranks.
    """

    if communicator is None:
        communicator = comm

    arr = _as_array(x).astype(bool, copy=False)

    if n_nodes <= 1:
        if out is None:
            return arr
        out[...] = arr
        return out

    if out is None:
        # By default, do not mutate input
        out = arr.copy()

    if out is arr:
        communicator.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.LOR)
    else:
        communicator.Allreduce(arr, out, op=MPI.LOR)

    return out


def mpi_all(x, *, out=None, communicator=None):
    """
    Logical AND across ranks.
    """
    if communicator is None:
        communicator = comm

    arr = _as_array(x).astype(bool, copy=False)

    if n_nodes <= 1:
        if out is None:
            return arr
        out[...] = arr
        return out

    if out is None:
        # By default, do not mutate input
        out = arr.copy()

    if out is arr:
        communicator.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.LAND)
    else:
        communicator.Allreduce(arr, out, op=MPI.LAND)

    return out


#
#
#   point-to-point / gather-scatter
def mpi_bcast(x: Any, *, root: int = 0, communicator=None):
    """
    Broadcast x from root to every rank.
    """
    if communicator is None:
        communicator = comm

    return communicator.bcast(x, root=root) if n_nodes > 1 else x


def mpi_allgather(x, *, communicator=None):
    """
    Gather x from every rank to every rank.
    """
    if communicator is None:
        communicator = comm

    if isinstance(x, np.ndarray):
        arr = _as_array(x)

        if n_nodes > 1:
            out = np.empty((n_nodes,) + arr.shape, dtype=arr.dtype)
            communicator.Allgather(arr, out)
            return out

        # serial fallback
        return arr.reshape((1,) + arr.shape)

    if n_nodes > 1:
        return communicator.allgather(x)

    # serial fallback for Python objects
    return (x,)


def mpi_gather(x, *, root: int = 0, communicator=None):
    """
    Gather x from every rank onto root.
    """
    if communicator is None:
        communicator = comm

    if isinstance(x, np.ndarray):
        arr = _as_array(x)

        if n_nodes > 1:
            recv = None
            if communicator.Get_rank() == root:
                recv = np.empty((n_nodes,) + arr.shape, dtype=arr.dtype)
            communicator.Gather(arr, recv, root=root)
            return recv

        # serial fallback
        return arr.reshape((1,) + arr.shape)

    if n_nodes > 1:
        return communicator.gather(x, root=root)

    # serial fallback for Python objects
    return (x,)


def mpi_scatter(xs: Sequence[Any], *, root: int = 0, communicator=None):
    """
    Scatter a sequence xs (living only on root) so that each rank receives one item.
    Serial mode expects len(xs) == 1.
    """
    if communicator is None:
        communicator = comm

    if n_nodes > 1:
        rnk = communicator.Get_rank()
        return communicator.scatter(xs if rnk == root else None, root=root)

    # serial fallback
    if len(xs) != 1:
        raise ValueError("Serial scatter expects a length-1 sequence.")
    return xs[0]


#
#
#   JAX variants


@promote_to_pytree
def mpi_sum_jax(x: Array, *, token: Token = None, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.SUM, comm=communicator, token=token)


@promote_to_pytree
def mpi_prod_jax(x: Array, *, token: Token = None, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.PROD, comm=communicator, token=token)


@promote_to_pytree
def mpi_mean_jax(x: Array, *, token: Token = None, communicator=None):
    res, token = mpi_sum_jax(x, token=token, communicator=communicator)
    return res / n_nodes, token


@promote_to_pytree
def mpi_max_jax(x: Array, *, token: Token = None, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.MAX, comm=communicator, token=token)


@promote_to_pytree
def mpi_any_jax(x, *, token=None, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.LOR, comm=communicator, token=token)


@promote_to_pytree
def mpi_all_jax(x, *, token=None, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.LAND, comm=communicator, token=token)


@promote_to_pytree
def mpi_bcast_jax(x: Array, *, token: Token = None, root, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        assert root == 0
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.bcast(x, token=token, root=root, comm=communicator)


@promote_to_pytree
def mpi_allgather_jax(x: Array, *, token: Token = None, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return jnp.expand_dims(x, 0), token
    else:
        import mpi4jax

        return mpi4jax.allgather(x, token=token, comm=communicator)


@promote_to_pytree
def mpi_gather_jax(x: Array, *, token: Token = None, root: int = 0, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        return jnp.expand_dims(x, 0), token
    else:
        import mpi4jax

        return mpi4jax.gather(x, token=token, root=root, comm=communicator)


@promote_to_pytree
def mpi_scatter_jax(x: Array, *, token: Token = None, root: int = 0, communicator=None):
    if communicator is None:
        communicator = comm_jax

    if n_nodes == 1:
        if x.shape[0] != 1:
            raise ValueError("Invalid input shape for scattering.")
        return x[0], token
    else:
        import mpi4jax

        return mpi4jax.scatter(x, root=root, token=token, comm=communicator)
