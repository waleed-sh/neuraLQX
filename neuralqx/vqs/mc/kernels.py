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

# Copyright 2021 The NetKet Authors - All rights reserved.
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

# pylint: skip-file
# fmt: off

"""
This file overrides some of the implementation of some common kernels used by
MCState and MCMixedState in NetKet

NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax
from netket.operator import DiscreteJaxOperator

from neuralqx.debug import event
from neuralqx.operators.types import ComputationalJaxOperator


def _reshape_conn_args(σ: Array, σp: Array, mels: Array):
    """
    Ensure connected-state arguments have the standard padded shape.
    """
    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])
    return σp, mels


def batch_discrete_kernel(kernel):
    """
    Batch a decorator that only works with 1 sample so that it works with a
    batch of samples.

    Works only for discrete-kernels who take two args as inputs
    """

    def vmapped_kernel(logpsi, pars, σ, args):
        """
        local_value kernel for MCState and generic operators
        """
        σp, mels = args

        if jnp.ndim(σp) != 3:
            σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
            mels = mels.reshape(σp.shape[:-1])

        vkernel = jax.vmap(kernel, in_axes=(None, None, 0, (0, 0)), out_axes=0)
        return vkernel(logpsi, pars, σ, (σp, mels))

    return vmapped_kernel


# Regular Numba and JAX kernels

@batch_discrete_kernel
def local_value_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mel = args
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_kernel_jax(
    logpsi: Callable, pars: PyTree, σ: Array, O: DiscreteJaxOperator
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    σp, mel = O.get_conn_padded(σ)
    logpsi_σ = logpsi(pars, σ)
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_kernel_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
):
    """
    Bra-action local estimator using precomputed `logpsi(pars, σ)`.
    """
    σp, mels = args
    σp, mels = _reshape_conn_args(σ, σp, mels)
    n = σ.shape[-1]
    logpsi_σp = logpsi(pars, σp.reshape((-1, n))).reshape(σp.shape[:-1])
    return jnp.sum(mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_kernel_jax_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
):
    """
    JAX-operator bra-action local estimator using precomputed `logpsi(pars, σ)`.
    """
    σp, mels = O.get_conn_padded(σ)
    n = σ.shape[-1]
    logpsi_σp = logpsi(pars, σp.reshape((-1, n))).reshape(σp.shape[:-1])
    return jnp.sum(mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_squared_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return jnp.abs(local_value_kernel(logpsi, pars, σ, args)) ** 2


def local_value_kernel_squared_jax(
    logpsi: Callable, pars: PyTree, σ: Array, O: DiscreteJaxOperator
):
    """
    Squared‐operator local estimator for any DiscreteJaxOperator
    """
    return jnp.abs(local_value_kernel_jax(logpsi, pars, σ, O)) ** 2


def local_value_squared_kernel_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
):
    return jnp.abs(local_value_kernel_factored(logpsi_σ, logpsi, pars, σ, args)) ** 2


def local_value_kernel_squared_jax_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
):
    return jnp.abs(local_value_kernel_jax_factored(logpsi_σ, logpsi, pars, σ, O)) ** 2


@batch_discrete_kernel
def local_value_op_op_cost(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σp, mel = args

    σ_σp = jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(σp, σ)
    σ_σ = jnp.hstack((σ, σ))
    return jnp.sum(mel * jnp.exp(logpsi(pars, σ_σp) - logpsi(pars, σ_σ)))


# Chunked versions of those kernels are defined below.


def local_value_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_chunked(σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_chunked(σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mels * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)


def local_value_kernel_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    Chunked bra-action estimator using precomputed `logpsi(pars, σ)`.
    """
    σp, mels = args
    σp, mels = _reshape_conn_args(σ, σp, mels)

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )
    n = σ.shape[-1]
    logpsi_σp = logpsi_chunked(σp.reshape((-1, n))).reshape(σp.shape[:-1])

    return jnp.sum(
        mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1
    )


def local_value_kernel_jax_conn_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    chunk_size: int,
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    apply_conn = lambda s: logpsi(pars, s)
    apply_conn = nkjax.apply_chunked(apply_conn, in_axes=0, chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)

    logpsi_σ = apply_conn(σ)
    logpsi_σp = apply_conn(σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_kernel_jax_conn_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    chunk_size: int,
):
    """
    Chunked JAX-operator bra-action estimator using precomputed `logpsi(pars, σ)`.
    """
    apply_conn = lambda s: logpsi(pars, s)
    apply_conn = nkjax.apply_chunked(apply_conn, in_axes=0, chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)
    logpsi_σp = apply_conn(σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_squared_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return (
        jnp.abs(
            local_value_kernel_chunked(logpsi, pars, σ, args, chunk_size=chunk_size)
        )
        ** 2
    )


def local_value_squared_kernel_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    return jnp.abs(
        local_value_kernel_chunked_factored(
            logpsi_σ, logpsi, pars, σ, args, chunk_size=chunk_size
        )
    ) ** 2


def local_value_kernel_squared_jax_conn_chunked(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        O: DiscreteJaxOperator,
        chunk_size: int,
):
    """
    Squared version with the same splitting strategy
    """

    return jnp.abs(
        local_value_kernel_jax_conn_chunked(logpsi, pars, σ, O, chunk_size)
    ) ** 2


def local_value_kernel_squared_jax_conn_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    chunk_size: int,
):
    return jnp.abs(
        local_value_kernel_jax_conn_chunked_factored(
            logpsi_σ, logpsi, pars, σ, O, chunk_size
        )
    ) ** 2


def local_value_op_op_cost_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    σ_σp = jax.vmap(
        lambda σpi, σi: jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(
            σpi, σi
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(σp, σ)
    σ_σ = jax.vmap(lambda σi: jnp.hstack((σi, σi)), in_axes=0)(σ)

    return local_value_kernel_chunked(
        logpsi, pars, σ_σ, (σ_σp, mels), chunk_size=chunk_size
    )


def local_value_kernel_jax_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and jaxcoompatible operators
    """
    if chunk_size >= O.max_conn_size:
        local_value_kernel = lambda s: local_value_kernel_jax(logpsi, pars, s, O)
        local_value_chunked = nkjax.apply_chunked(
            local_value_kernel,
            in_axes=0,
            chunk_size=max(1, chunk_size // O.max_conn_size),
        )
    else:
        local_value_chunked = lambda s: local_value_kernel_jax_conn_chunked(
            logpsi, pars, s, O, chunk_size
        )

    return local_value_chunked(σ)


def local_value_kernel_jax_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
):
    # In factored mode we always use the connected-state chunking variant to
    # avoid recomputing `logpsi(σ)`.
    return local_value_kernel_jax_conn_chunked_factored(
        logpsi_σ, logpsi, pars, σ, O, int(chunk_size)
    )


# PenaltyCost kernels


@batch_discrete_kernel
def local_value_kernel_penalty_cost(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        args: PyTree,
):
    """
    Just the regular kernel for a local operator for now
    """

    σp, mel = args
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_kernel_penalty_cost_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
):
    return local_value_kernel_factored(logpsi_σ, logpsi, pars, σ, args)

#
#
#   Chunked version
def local_value_kernel_penalty_cost_chunked(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        args: PyTree,
        *,
        chunk_size: int | None = None,
):
    """
    Just the regular kernel for a local operator for now
    """

    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_chunked(σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_chunked(σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mels * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)


def local_value_kernel_penalty_cost_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: int | None = None,
):
    return local_value_kernel_chunked_factored(
        logpsi_σ, logpsi, pars, σ, args, chunk_size=chunk_size
    )


"""
NOT batched – operates on the full σ

The VarianceObservable needs the global batch mean ⟨O⟩, wrapping in the
batch_discrete_kernel vmap would hide that information. _expect() already calls the kernel once on 
the whole sample array, so the signature still matches all existing infrastructure
"""
def local_value_kernel_variance(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
):
    """
    Local estimator for a VarianceObservable.

    args = (local_kernel_O, args_O, local_kernel_O2, args_O2)

    Returns  L_σ = O²_loc(σ) − ⟨O⟩_MC²   (real part only).

    The mean ⟨O⟩ is taken over the same mini-batch σ that NetKet's
    Monte-Carlo integration is using, which is the usual unbiased
    MC estimator employed in the original NetKet routine.
    """

    local_O, args_O, local_O2, args_O2 = args

    # shape (n_samples, )
    O_loc = local_O(logpsi, pars, σ, args_O)
    O2_loc = local_O2(logpsi, pars, σ, args_O2)

    # scalar, differentiable
    O_mean = jnp.mean(O_loc)

    # same shape as O_loc
    return (O2_loc - O_mean ** 2).real


@partial(jax.jit, static_argnames=("logpsi", "inner_kernel"))
def volume_cost_kernel(
    logpsi: callable,
    inner_kernel: callable,
    pars: PyTree,
    σ: jnp.ndarray,
    inner_args: PyTree,
    fprime: jnp.ndarray,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """
    Jitted per-sample local estimator for the InverseExpectationCost type

        C_loc(σ) = f'(<V>) * V_loc(σ) + g

    here `logpsi` and `inner_kernel` are static (functions) and `inner_args`, `fprime`, `g` are
    non-static runtime values (device arrays).

    This ensures a single stable compiled executable regardless of how often
    you re-enter training, while still allowing <V> to change batch-to-batch
    """

    # per-sample local estimator of the underlying operator
    V_loc = inner_kernel(logpsi, pars, σ, inner_args)

    # `fprime` and `g` are scalars/broadcast-able arrays computed outside
    # and synchronized across ranks, just form the affine combination
    return fprime * V_loc + g



#
#
#   ket-action local estimators (for ComputationalOperator / ComputationalJaxOperator)

@batch_discrete_kernel
def local_value_kernel_ket_action(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    Local estimator for operators that return KET-action connections:
        O|σ> = Σ_{σ'} O(σ',σ) |σ'>
    Returns:
        O_loc(σ) = Σ_{σ'} O(σ',σ) * (ψ*(σ')/ψ*(σ))
    so that averaging over |ψ|^2 yields <O>.
    """
    event(msg="KET_ACTION_CALLED", tag="local_value_kernel_ket_action")
    σp, mel = args
    return jnp.sum(mel * jnp.exp(jnp.conj(logpsi(pars, σp) - logpsi(pars, σ))))


def local_value_kernel_jax_ket_action(
    logpsi: Callable, pars: PyTree, σ: Array, O: ComputationalJaxOperator
):
    """
    Same as local_value_kernel_ket_action, but for DiscreteJaxOperator-style padded connections.
    """
    event(msg="KET_ACTION_CALLED", tag="local_value_kernel_ket_action_jax")
    σp, mel = O.get_conn_padded(σ)
    logpsi_σ = logpsi(pars, σ)
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    return jnp.sum(
        mel * jnp.exp(jnp.conj(logpsi_σp - jnp.expand_dims(logpsi_σ, -1))),
        axis=-1,
    )


def local_value_kernel_ket_action_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
):
    """
    Ket-action local estimator using precomputed `logpsi(pars, σ)`.
    """
    σp, mels = args
    σp, mels = _reshape_conn_args(σ, σp, mels)
    n = σ.shape[-1]
    logpsi_σp = logpsi(pars, σp.reshape((-1, n))).reshape(σp.shape[:-1])
    return jnp.sum(
        mels * jnp.exp(jnp.conj(logpsi_σp - jnp.expand_dims(logpsi_σ, -1))),
        axis=-1,
    )


def local_value_kernel_jax_ket_action_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: ComputationalJaxOperator,
):
    """
    JAX ket-action local estimator using precomputed `logpsi(pars, σ)`.
    """
    σp, mels = O.get_conn_padded(σ)
    n = σ.shape[-1]
    logpsi_σp = logpsi(pars, σp.reshape((-1, n))).reshape(σp.shape[:-1])
    return jnp.sum(
        mels * jnp.exp(jnp.conj(logpsi_σp - jnp.expand_dims(logpsi_σ, -1))),
        axis=-1,
    )


def local_value_squared_kernel_ket_action(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    return jnp.abs(local_value_kernel_ket_action(logpsi, pars, σ, args)) ** 2


def local_value_kernel_squared_jax_ket_action(
    logpsi: Callable, pars: PyTree, σ: Array, O: ComputationalJaxOperator
):
    return jnp.abs(local_value_kernel_jax_ket_action(logpsi, pars, σ, O)) ** 2


def local_value_squared_kernel_ket_action_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
):
    return jnp.abs(
        local_value_kernel_ket_action_factored(logpsi_σ, logpsi, pars, σ, args)
    ) ** 2


def local_value_kernel_squared_jax_ket_action_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: ComputationalJaxOperator,
):
    return jnp.abs(
        local_value_kernel_jax_ket_action_factored(logpsi_σ, logpsi, pars, σ, O)
    ) ** 2


#
#
#   chunked ket actions

def local_value_kernel_ket_action_chunked(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        args: PyTree,
        *,
        chunk_size: Optional[int] = None,
):
    event(msg="CHUNKED_KET_ACTION_CALLED", tag="local_value_kernel_ket_action_chunked")
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_chunked(σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_chunked(σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mels * jnp.exp(jnp.conj(logpsi_σp - logpsi_σ)), axis=-1)


def local_value_kernel_ket_action_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    σp, mels = args
    σp, mels = _reshape_conn_args(σ, σp, mels)

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )
    n = σ.shape[-1]
    logpsi_σp = logpsi_chunked(σp.reshape((-1, n))).reshape(σp.shape[:-1])

    return jnp.sum(
        mels * jnp.exp(jnp.conj(logpsi_σp - jnp.expand_dims(logpsi_σ, -1))),
        axis=-1,
    )

def local_value_kernel_jax_ket_action_conn_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: ComputationalJaxOperator,
    chunk_size: int,
):
    apply_conn = lambda s: logpsi(pars, s)
    apply_conn = nkjax.apply_chunked(apply_conn, in_axes=0, chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)

    logpsi_σ = apply_conn(σ)
    logpsi_σp = apply_conn(σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    return jnp.sum(mel * jnp.exp(jnp.conj(logpsi_σp - jnp.expand_dims(logpsi_σ, -1))), axis=-1)


def local_value_kernel_jax_ket_action_conn_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: ComputationalJaxOperator,
    chunk_size: int,
):
    apply_conn = lambda s: logpsi(pars, s)
    apply_conn = nkjax.apply_chunked(apply_conn, in_axes=0, chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)
    logpsi_σp = apply_conn(σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    return jnp.sum(
        mel * jnp.exp(jnp.conj(logpsi_σp - jnp.expand_dims(logpsi_σ, -1))),
        axis=-1,
    )


def local_value_kernel_jax_ket_action_chunked(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        O: ComputationalJaxOperator,
        *,
        chunk_size: int | None = None,
):
    event(msg="CHUNKED_KET_ACTION_CALLED", tag="local_value_kernel_ket_action_jax_chunked")

    if chunk_size >= O.max_conn_size:
        local_value_kernel = lambda s: local_value_kernel_jax_ket_action(logpsi, pars, s, O)
        local_value_chunked = nkjax.apply_chunked(
            local_value_kernel,
            in_axes=0,
            chunk_size=max(1, chunk_size // O.max_conn_size),
        )
    else:
        local_value_chunked = lambda s: local_value_kernel_jax_ket_action_conn_chunked(
            logpsi, pars, s, O, chunk_size
        )

    return local_value_chunked(σ)


def local_value_kernel_jax_ket_action_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: ComputationalJaxOperator,
    *,
    chunk_size: int | None = None,
):
    return local_value_kernel_jax_ket_action_conn_chunked_factored(
        logpsi_σ, logpsi, pars, σ, O, int(chunk_size)
    )

#
#
#   chunked Squared kernels

# computational operators with a ket action
def local_value_squared_kernel_ket_action_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    return (
        jnp.abs(
            local_value_kernel_ket_action_chunked(logpsi, pars, σ, args, chunk_size=chunk_size)
        )
        ** 2
    )


def local_value_squared_kernel_ket_action_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    return jnp.abs(
        local_value_kernel_ket_action_chunked_factored(
            logpsi_σ, logpsi, pars, σ, args, chunk_size=chunk_size
        )
    ) ** 2

# discrete jax operators with bra action
def local_value_kernel_squared_jax_chunked(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        O: DiscreteJaxOperator,
        *,
        chunk_size: Optional[int] = None,
):
    return jnp.abs(
        local_value_kernel_jax_chunked(logpsi, pars, σ, O, chunk_size=chunk_size)
    ) ** 2


def local_value_kernel_squared_jax_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: Optional[int] = None,
):
    return jnp.abs(
        local_value_kernel_jax_chunked_factored(
            logpsi_σ, logpsi, pars, σ, O, chunk_size=chunk_size
        )
    ) ** 2

# computational jax operators with ket action
def local_value_kernel_squared_jax_ket_action_chunked(
        logpsi: Callable,
        pars: PyTree,
        σ: Array,
        O: ComputationalJaxOperator,
        *,
        chunk_size: Optional[int] = None,
):
    return jnp.abs(
        local_value_kernel_jax_ket_action_chunked(logpsi, pars, σ, O, chunk_size=chunk_size)
    ) ** 2


def local_value_kernel_squared_jax_ket_action_chunked_factored(
    logpsi_σ: Array,
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: ComputationalJaxOperator,
    *,
    chunk_size: Optional[int] = None,
):
    return jnp.abs(
        local_value_kernel_jax_ket_action_chunked_factored(
            logpsi_σ, logpsi, pars, σ, O, chunk_size=chunk_size
        )
    ) ** 2


_FACTORED_LOCAL_KERNELS = {
    local_value_kernel: local_value_kernel_factored,
    local_value_kernel_jax: local_value_kernel_jax_factored,
    local_value_squared_kernel: local_value_squared_kernel_factored,
    local_value_kernel_squared_jax: local_value_kernel_squared_jax_factored,
    local_value_kernel_penalty_cost: local_value_kernel_penalty_cost_factored,
    local_value_kernel_ket_action: local_value_kernel_ket_action_factored,
    local_value_kernel_jax_ket_action: local_value_kernel_jax_ket_action_factored,
    local_value_squared_kernel_ket_action: local_value_squared_kernel_ket_action_factored,
    local_value_kernel_squared_jax_ket_action: local_value_kernel_squared_jax_ket_action_factored,
}


_FACTORED_LOCAL_KERNELS_CHUNKED = {
    local_value_kernel_chunked: local_value_kernel_chunked_factored,
    local_value_kernel_jax_conn_chunked: local_value_kernel_jax_conn_chunked_factored,
    local_value_kernel_jax_chunked: local_value_kernel_jax_chunked_factored,
    local_value_squared_kernel_chunked: local_value_squared_kernel_chunked_factored,
    local_value_kernel_squared_jax_conn_chunked: local_value_kernel_squared_jax_conn_chunked_factored,
    local_value_kernel_squared_jax_chunked: local_value_kernel_squared_jax_chunked_factored,
    local_value_kernel_penalty_cost_chunked: local_value_kernel_penalty_cost_chunked_factored,
    local_value_kernel_ket_action_chunked: local_value_kernel_ket_action_chunked_factored,
    local_value_kernel_jax_ket_action_conn_chunked: local_value_kernel_jax_ket_action_conn_chunked_factored,
    local_value_kernel_jax_ket_action_chunked: local_value_kernel_jax_ket_action_chunked_factored,
    local_value_squared_kernel_ket_action_chunked: local_value_squared_kernel_ket_action_chunked_factored,
    local_value_kernel_squared_jax_ket_action_chunked: local_value_kernel_squared_jax_ket_action_chunked_factored,
}


def resolve_factored_local_kernel(local_kernel: Callable, *, chunked: bool = False):
    """
    Return a factored local-estimator kernel if available for `local_kernel`.

    Factored kernels consume precomputed `logpsi(pars, σ)` and therefore avoid
    recomputing the wavefunction on the reference samples for every operator.
    """
    if chunked:
        return _FACTORED_LOCAL_KERNELS_CHUNKED.get(local_kernel)
    return _FACTORED_LOCAL_KERNELS.get(local_kernel)
