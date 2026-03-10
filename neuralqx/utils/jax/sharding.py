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
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import netket.jax._utils_tree as _nk_tree

_AXIS = "S"


def _mesh_axis_name(mesh) -> str:
    names = tuple(getattr(mesh, "axis_names", ()) or ())
    if names:
        return str(names[0])
    return _AXIS


def get_abstract_mesh():
    """
    Return the active abstract mesh installed by NetKet/JAX.

    NetKet 3.20+ uses a single mesh axis named 'S'.
    """
    return jax.sharding.get_abstract_mesh()


def _replicate_shmap_callback(f_py, x):
    """
    Execute a Python callback `f_py` on each shard of `x` and stitch the result
    back together using the currently active abstract mesh.
    """

    x_sharding = getattr(x, "sharding", None)
    x_mesh = getattr(x_sharding, "mesh", None)
    active_mesh = get_abstract_mesh()

    # When the input was manually placed on a mesh that differs from the
    # process-global abstract mesh (e.g. axis "d" vs "S"), avoid shard_map
    # because JAX rejects mixed meshes in one traced program.
    if x_mesh is not None and active_mesh is not None:
        if _mesh_axis_name(x_mesh) != _mesh_axis_name(active_mesh):
            bsz, n_sites = x.shape
            xp_spec = ShapeDtypeStruct(
                (bsz, f_py.max_conn, n_sites),
                x.dtype,
            )
            mels_spec = ShapeDtypeStruct(
                (bsz, f_py.max_conn),
                f_py.mel_dtype,
            )
            return pure_callback(
                f_py,
                (xp_spec, mels_spec),
                x,
                vmap_method="legacy_vectorized",
            )

    mesh = x_mesh or active_mesh
    axis = _mesh_axis_name(mesh)

    def _callback_full(x_full):
        bsz, n_sites = x_full.shape
        xp_spec = ShapeDtypeStruct(
            (bsz, f_py.max_conn, n_sites),
            x_full.dtype,
        )
        mels_spec = ShapeDtypeStruct(
            (bsz, f_py.max_conn),
            f_py.mel_dtype,
        )
        return pure_callback(
            f_py,
            (xp_spec, mels_spec),
            x_full,
            vmap_method="legacy_vectorized",
        )

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(axis),),
        out_specs=(P(axis), P(axis)),
    )
    def _per_shard(x_shard):
        return _callback_full(x_shard)

    try:
        return _per_shard(x)
    except ValueError as e:
        # Fallback for mesh-context mismatches between global abstract mesh and
        # incoming NamedSharding (seen with mixed 'S'/'d' axis names).
        if "mesh" in str(e).lower():
            return _callback_full(x)
        raise


def replicate_sharding(f_py):
    """
    Decorator replicating the sharding mechanism for jax.pure_callback-based
    get_conn_padded() implementations.
    """
    return partial(_replicate_shmap_callback, f_py)


def replicated_sharding():
    mesh = get_abstract_mesh()
    return NamedSharding(mesh, P())


def sharded_sharding_1d():
    mesh = get_abstract_mesh()
    return NamedSharding(mesh, P(_mesh_axis_name(mesh)))


def normalize_pspec(pspec, ndim: int) -> tuple:
    ps = tuple(pspec)
    if len(ps) < ndim:
        ps = ps + (None,) * (ndim - len(ps))
    return ps


def derive_output_shardings(in_sharding, in_ndim: int):
    """
    Derive output shardings for:
      xp  : (batch, max_conn, n_sites)
      mels: (batch, max_conn)

    assuming the input x has shape:
      x   : (batch, n_sites)

    We preserve sharding of the batch axis and never shard the sites axis
    or the inserted max_conn axis.
    """
    if not isinstance(in_sharding, NamedSharding):
        return None, None

    in_pspec = normalize_pspec(in_sharding.spec, in_ndim)

    # Never allow sharding along the site axis (last axis of x)
    if in_pspec[-1] is not None:
        return None, None

    xp_pspec = in_pspec[:-1] + (None,) + in_pspec[-1:]
    ml_pspec = in_pspec[:-1] + (None,)

    return (
        NamedSharding(in_sharding.mesh, P(*xp_pspec)),
        NamedSharding(in_sharding.mesh, P(*ml_pspec)),
    )


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


@jax.jit
def _tree_ax_fixed(a, x):
    if hasattr(a, "ndim") and a.ndim == 0:
        return jax.tree_util.tree_map(lambda x_: a * x_, x)
    return jax.tree_util.tree_map(lambda a_, x_: a_ * x_, a, x)


@jax.jit
def _tree_axpy_fixed(a, x, y):
    ax = _tree_ax_fixed(a, x)
    return jax.tree_util.tree_map(lambda ax_, y_: ax_ + y_, ax, y)


_nk_tree.tree_ax = _tree_ax_fixed
_nk_tree.tree_axpy = _tree_axpy_fixed
