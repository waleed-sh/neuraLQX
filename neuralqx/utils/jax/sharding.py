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
Some helper functions for sharding
"""

from functools import partial

import jax
from jax import ShapeDtypeStruct, pure_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, PositionalSharding

_AXIS = "d"


def _replicate_shmap_callback(f_py, x):
    """
    This is a decorator which is used in JAX-compatible custom LocalOperator types. It is designed
    to call the .get_conn_padded() of the Numba version of the operator and return its result.

    So it executes a Python callback `f_py` (returns xp, mels) on every host with its local shard of
    `x`, then stitch results together.
    """

    # dev: when supporting NetKet 3.19+, the commented lines should be used

    mesh = Mesh(jax.devices(), axis_names=("d",))
    # mesh = jax.sharding.get_abstract_mesh()

    # @partial(jax.shard_map, axis_names={'S'}, in_specs=(jax.P('S')), out_specs=(jax.P('S'), jax.P('S')))
    @partial(shard_map, mesh=mesh, in_specs=(P("d"),), out_specs=(P("d"), P("d")))
    def _per_shard(x_shard):

        local_batch, n_sites = x_shard.shape

        xp_spec = ShapeDtypeStruct((local_batch, f_py.max_conn, n_sites), x_shard.dtype)

        mels_spec = ShapeDtypeStruct((local_batch, f_py.max_conn), f_py.mel_dtype)

        xp_loc, mels_loc = pure_callback(
            f_py,
            (xp_spec, mels_spec),
            x_shard,
            vmap_method="legacy_vectorized",
        )

        return xp_loc, mels_loc

    return _per_shard(x)


def replicate_sharding(f_py):
    """
    A decorator to replicate the sharding mechanism for jax.pure_callback of get_conn_padded()
    of Numba types from their JAX-compatible types
    """
    return partial(_replicate_shmap_callback, f_py)


def get_abstract_mesh():

    return Mesh(jax.devices(), axis_names=("d",))
    #
    #
    #   uncomment when supporting latest NetKet
    # if hasattr(jax, "distributed") and hasattr(jax.distributed, "get_abstract_mesh"):
    #     return jax.distributed.get_abstract_mesh()
    # return jax.sharding.get_abstract_mesh()


def replicated_sharding():
    mesh = get_abstract_mesh()
    return NamedSharding(mesh, P())


def sharded_sharding_1d():
    mesh = get_abstract_mesh()
    return NamedSharding(mesh, P(_AXIS))


def normalize_pspec(pspec, ndim: int) -> tuple:
    ps = tuple(pspec)
    if len(ps) < ndim:
        ps = ps + (None,) * (ndim - len(ps))
    return ps


def derive_output_shardings(in_sharding, in_ndim: int, x_shape=None):

    # NamedSharding path (mesh + PartitionSpec)
    if isinstance(in_sharding, NamedSharding):
        in_pspec = normalize_pspec(in_sharding.spec, in_ndim)

        # Never allow sharding along site axis (last axis)
        if in_pspec[-1] is not None:
            return None, None

        xp_pspec = in_pspec[:-1] + (None,) + in_pspec[-1:]
        ml_pspec = in_pspec[:-1] + (None,)

        return (
            NamedSharding(in_sharding.mesh, P(*xp_pspec)),
            NamedSharding(in_sharding.mesh, P(*ml_pspec)),
        )

    # PositionalSharding path (heuristic)
    if PositionalSharding is not None and isinstance(in_sharding, PositionalSharding):
        if x_shape is not None:
            M = x_shape[-1]
            full_last = slice(0, M)
            dev_map = in_sharding.devices_indices_map(x_shape)
            for _, slc in dev_map.items():
                if slc is None:
                    continue
                if slc[-1] != full_last:
                    return None, None

        # PositionalSharding shards leading axes only, inserting K near the end
        # does not change leading-axis sharding. Reuse it
        return in_sharding, in_sharding

    return None, None


def is_distributed_array(x: jax.Array) -> bool:
    sh = getattr(x, "sharding", None)
    if sh is None:
        return False

    devset = getattr(sh, "device_set", None)
    if devset is None:
        return False

    return len(devset) > 1


def is_sharded_array(x: jax.Array) -> bool:

    sh = getattr(x, "sharding", None)
    if sh is None:
        return False
    if not is_distributed_array(x):
        return False

    if getattr(sh, "is_fully_replicated", False):
        return False

    return True
