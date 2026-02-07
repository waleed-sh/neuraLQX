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
This file overrides some of the implementation of some common functions used to compute the
expectation value using an MCState in NetKet

NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""


from typing import Callable, Sequence, Union
from functools import partial

import jax
from jax import Array, numpy as jnp

from netket.utils import mpi
from netket.operator._abstract_observable import AbstractObservable

from netket.stats import Stats, statistics as mpi_statistics
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    Squared,
    ContinuousOperator,
    DiscreteJaxOperator,
)

from netket.experimental.observable import VarianceObservable

from netket.vqs.mc import check_hilbert
from netket.vqs.mc.mc_state.state import MCState

from ..kernels import volume_cost_kernel
from ....debug import trace
from ....operators import InverseExpectationCost, PenaltyCost
from ....operators.types._discrete_operator import DiscreteOperator as DiscreteOperatorNQX
from .. import kernels
from ....utils.errors import ExpectationValueError
from .state import MCState as NQXMCState
from ...mc import get_local_kernel_arguments, get_local_kernel

from ....operators.types.computational_operator import ComputationalOperator, ComputationalJaxOperator
from ....utils.parsing import strict_type


@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: PenaltyCost,
):
    σ = vstate.samples
    σp, mel = Ô.parent.get_conn_padded(σ)
    return σ, (σp, mel)


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: PenaltyCost,
):
    # if the wrapped operator is ket-action, we must use the ket-action estimator
    if isinstance(Ô.parent, (ComputationalOperator, ComputationalJaxOperator)) and Ô.parent.is_ket_action:
        if isinstance(Ô.parent, ComputationalOperator):
            return kernels.local_value_kernel_ket_action

        if isinstance(Ô.parent, ComputationalJaxOperator):
            return kernels.local_value_kernel_jax_ket_action

    return kernels.local_value_kernel_penalty_cost



@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: Squared,
):

    if isinstance(Ô.parent, DiscreteJaxOperator):
        σ = vstate.samples
        # we just pass the whole Squared object downstream, the kernel
        # knows how to use Ô.parent efficiently
        return σ, Ô.parent

    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.parent.get_conn_padded(σ)
    return σ, (σp, mels)


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: Squared,
):
    # computational JAX operators, with ket action
    if isinstance(Ô.parent, ComputationalJaxOperator) and Ô.parent.is_ket_action:
        return kernels.local_value_kernel_squared_jax_ket_action

    # discrete or computational JAX operators with bra action
    if isinstance(Ô.parent, DiscreteJaxOperator):
        return kernels.local_value_kernel_squared_jax

    # non-JAX operators (args are (σp,mels)) with ket action
    if isinstance(Ô.parent, ComputationalOperator) and Ô.parent.is_ket_action:
        return kernels.local_value_squared_kernel_ket_action

    # discrete or computational numba operators with bra action
    return kernels.local_value_squared_kernel



@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: Union[DiscreteOperator, DiscreteOperatorNQX],
):
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(σ)
    return σ, (σp, mels)


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: Union[DiscreteOperator, DiscreteOperatorNQX],
):
    return kernels.local_value_kernel


@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: DiscreteJaxOperator,
):
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    return σ, Ô


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: DiscreteJaxOperator,
):
    return kernels.local_value_kernel_jax


@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: ContinuousOperator,
):
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    args = Ô._pack_arguments()
    return σ, args


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: ContinuousOperator,
):
    # TODO: this should be moved other to dispatch in order to support MCMixedState
    return Ô._expect_kernel


#
#
#   VarianceObservable kernels


@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: VarianceObservable,
):
    σ,  args_O = get_local_kernel_arguments(vstate, Ô.operator)
    _,  args_O2 = get_local_kernel_arguments(vstate, Ô.operator_squared)
    return σ, (args_O, args_O2)


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: VarianceObservable,
):
    # grab the inner kernels once, close over them
    local_O = get_local_kernel(vstate, Ô.operator)
    local_O2 = get_local_kernel(vstate, Ô.operator_squared)

    def _lk_variance(logpsi, pars, σ, args):
        args_O, args_O2 = args
        return kernels.local_value_kernel_variance(
            logpsi,
            pars,
            σ,
            (local_O, args_O, local_O2, args_O2),
        )

    return _lk_variance


#
#
#   Computational Kernels


@dispatch
def get_local_kernel_arguments(vstate: Union[MCState, NQXMCState], Ô: ComputationalOperator):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(σ)
    return σ, (σp, mels)

@dispatch
def get_local_kernel(vstate: Union[MCState, NQXMCState], Ô: ComputationalOperator):
    if Ô.is_ket_action:
        return kernels.local_value_kernel_ket_action
    return kernels.local_value_kernel


@dispatch
def get_local_kernel_arguments(vstate: Union[MCState, NQXMCState], Ô: ComputationalJaxOperator):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    return σ, Ô

@dispatch
def get_local_kernel(vstate: Union[MCState, NQXMCState], Ô: ComputationalJaxOperator):
    if Ô.is_ket_action:
        return kernels.local_value_kernel_jax_ket_action
    return kernels.local_value_kernel_jax



#
#
#   InverseExpectationCost kernels


@dispatch
def get_local_kernel_arguments(
        vstate: Union[MCState, NQXMCState],
        Ô: InverseExpectationCost,
):
    """
    Build the args for the InverseExpectationCost with batch-consistent (σ, inner_args)

    We will:
        - collapse σ like _expect() does
        - build inner_args from that collapsed σ
        - compute <V> from the same pair
        - pack dynamic scalars (fprime, g)
        - return the σ_collapsed so later kernels see the same layout and avoid recompilation
    """

    # gather the σ states
    σ = vstate.samples

    # collapse σ if needed

    σ_collapsed = jax.lax.collapse(σ, 0, 2) if σ.ndim >= 3 else σ

    # get the kernel for E, not the IEC, and args built from the collapsed σ
    inner_kernel = get_local_kernel(vstate, Ô.cost_operator)
    _, inner_args = get_local_kernel_arguments(vstate, Ô.cost_operator)

    # compute <V> from the same (σ_collapsed, inner_args)
    W = {"params": vstate.parameters, **vstate.model_state}
    V_loc = inner_kernel(vstate._apply_fun, W, σ_collapsed, inner_args)
    V_mean = jnp.mean(jnp.real(V_loc))

    # scalar
    V_mean = mpi.mpi_mean_jax(V_mean)[0]

    # scalars for the affine correction
    denom = Ô.alpha + V_mean + Ô.eps
    fprime = (-2.0 * Ô.factor) / (denom ** 3)
    g = (Ô.factor / (denom ** 2)) - fprime * V_mean

    # return σ that matches inner_args, and the packed dynamic args
    return σ_collapsed, (inner_args, fprime, g)


@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: InverseExpectationCost,
):

    inner_kernel = get_local_kernel(vstate, Ô.cost_operator)

    def _wrapped(logpsi, pars, σ, packed):
        inner_args, fprime, g = packed
        return volume_cost_kernel(
            logpsi,
            inner_kernel,
            pars,
            σ,
            inner_args,
            fprime,
            g,
        )

    return _wrapped


# Standard implementation of expect for an MCState (pure) and a generic operator
# The dispatch rule is not strictly needed, as everything currently implemented
# in NetKet only defines a custom get_local_kernel_arguments and get_local_kernel
# but if somebody wants to override behaviour for an existing operator or define
# a completely arbitrary novel type of operator, this makes it much easier.
@dispatch
@trace(tag="EXPECT")
def expect(
        vstate: Union[MCState, NQXMCState],
        Ô: Union[AbstractOperator, AbstractObservable],
        chunk_size: None
) -> Stats:  # noqa: F811
    σ, args = get_local_kernel_arguments(vstate, Ô)
    local_estimator_fun = get_local_kernel(vstate, Ô)

    if strict_type(Ô) is PenaltyCost:
        penalty_factor = Ô.factor
    else:
        penalty_factor = None

    return _expect(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
        penalty_factor,
    )


@dispatch
@trace(tag="EXPECT_SEQUENCE")
def expect(
    vstate: Union[MCState, NQXMCState],
    Ô: Sequence[Union[AbstractOperator, AbstractObservable]],
    chunk_size: None,
) -> Stats:
    """
    Extends the functionality of NetKet's `expect()` to handle a sequence of operators
    [Ô_1, ..., Ô_N]. Essentially, this computes <Ô_1> + ... + <Ô_N> instead of <Ô_1 + ... + Ô_N>.
    """

    # ensure list type
    Ô_list = list(Ô)

    # edge case check: empty list
    if len(Ô_list) == 0:
        raise ExpectationValueError("expect")

    # get the vstate.samples to ensure that we use the same ones for every Ô
    σ = vstate.samples

    # store the number of MC chains for Stats later
    n_chains = σ.shape[0]

    # flatten the chain dimension if needed
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    # the get_local_kernel which returns the local estimator we want to store for every Ô will
    # return a JAX array of shape (σ.shape[0], ), so we create an empty one with that shape
    # this array will hold the sum of all local estimators L_σ returned from every operator
    # dev: this was weak, essentially wrong... we were creating just a scalar zero, and later
    #      JAX changed it into an array after the first sum with the first L_σ, now it should be
    #      more "stable", aka less error prone
    # L_σ_sum = jnp.zeros_like((σ.shape[0],))
    L_σ_sum = jnp.zeros((σ.shape[0],), dtype = jnp.result_type(float))

    # now we want to call the dedicated JAX jitted function to compute L_σ per operator and
    # aggregate the values
    for ô in Ô_list:
        # discared the σ returned from the get_local_kernel_arguments as we want to avoid any
        # potential mismatch in using different σ for different operators in case it changes
        # somewhere during execution
        _, args = get_local_kernel_arguments(vstate, ô)
        local_estimator_fun = get_local_kernel(vstate, ô)

        L_σ = _expect_sequence(
            local_estimator_fun,
            vstate._apply_fun,
            vstate.sampler.machine_pow,
            vstate.parameters,
            vstate.model_state,
            σ,  # use the σ from before which we collected from vstate
            args,
        )

        # check if we are doing a penalty operator of Shannon type
        if strict_type(ô) is PenaltyCost:
            L_σ = ô.factor * L_σ

        # aggregate
        L_σ_sum = L_σ_sum + L_σ

    # now the loop is done, return the Stats for the entire sum of local operators
    return mpi_statistics(L_σ_sum.reshape((n_chains, -1)))


@partial(jax.jit, static_argnums=(0, 1))
def _expect_sequence(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Array:
    """
    This function is identical to NetKet's `_expect()` but it returns the local estimator and not
    the Stats object for it
    """

    # this is removed because it is used only in the removed code block below
    # n_chains = σ.shape[0]

    # this in principle could be removed because it is done once in the calling function and the σ
    # doesnt change but to avoid possible conflicts arising from this function being called from
    # some other function than the expect(), we keep it
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    # this is removed because it is used only in the removed code block below
    # def log_pdf(w, σ):
    #     return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    # TODO: Broken until google/jax#11916 is resolved.
    # should uncomment and remove code below once this is fixed
    # _, Ō_stats = nkjax.expect(
    #    log_pdf,
    #    partial(local_value_kernel, logpsi),
    #    parameters,
    #    σ,
    #    local_value_args,
    #    n_chains=n_chains,
    # )

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)

    # removed because we want to return raw estimators per operator in the sequence
    # Ō_stats = mpi_statistics(L_σ.reshape((n_chains, -1)))

    return L_σ


@partial(jax.jit, static_argnums=(0, 1))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
    penalty_factor: jnp.float64 = None,
) -> Stats:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    # TODO: Broken until google/jax#11916 is resolved.
    # should uncomment and remove code below once this is fixed
    # _, Ō_stats = nkjax.expect(
    #    log_pdf,
    #    partial(local_value_kernel, logpsi),
    #    parameters,
    #    σ,
    #    local_value_args,
    #    n_chains=n_chains,
    # )

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)

    if penalty_factor is not None:
        L_σ = penalty_factor * L_σ

    Ō_stats = mpi_statistics(L_σ.reshape((n_chains, -1)))

    return Ō_stats
