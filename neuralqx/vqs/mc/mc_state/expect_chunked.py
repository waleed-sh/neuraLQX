#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


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

import warnings
from typing import Callable, Sequence, Union
from functools import partial

import jax
from jax import Array, numpy as jnp

from netket.operator._abstract_observable import AbstractObservable

from netket.stats import Stats, statistics as statistics
from netket.utils.types import PyTree
from netket import jax as nkjax
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    Squared,
    ContinuousOperator,
    DiscreteJaxOperator,
)

from netket.experimental.observable import VarianceObservable

from netket.vqs.mc.mc_state.state import MCState

from ....debug import trace
from ....operators import InverseExpectationCost, PenaltyCost
from ....operators.types._discrete_operator import DiscreteOperator as DiscreteOperatorNQX
from ....utils.errors import ExpectationValueError
from .state import MCState as NQXMCState

from neuralqx.vqs.mc import (
    kernels,
    get_local_kernel,
    get_local_kernel_arguments,
)
from ....operators.types.computational_operator import ComputationalOperator, ComputationalJaxOperator
from ....utils.parsing import strict_type
from ....profile import section as prof_section

from neuralqx.vqs import expect

#
#
#   Dummy class for dispatching non-chunked types without chunking

class _NoChunking:
    pass

NO_CHUNKING = _NoChunking()


#
#
#   dispatch the chunked kernels

#
#
#   standard numba operators
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: Union[DiscreteOperator, DiscreteOperatorNQX],
        chunk_size: int,
):
    return kernels.local_value_kernel_chunked

#
#
#   standard JAX operators
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: DiscreteJaxOperator,
        chunk_size: int,
):
    return kernels.local_value_kernel_jax_chunked

#
#
#   continuous operators for completion
def _local_continuous_kernel(kernel, logpsi, pars, σ, args, *, chunk_size=None):
    def _kernel(σ):
        return kernel(logpsi, pars, σ, args)

    return nkjax.apply_chunked(_kernel, in_axes=0, chunk_size=chunk_size)(σ)


@dispatch
def get_local_kernel(  # noqa: F811
    vstate: Union[MCState, NQXMCState], Ô: ContinuousOperator, chunk_size: int
):
    return nkjax.HashablePartial(_local_continuous_kernel, Ô._expect_kernel)

#
#
#   Squared operators
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: Squared,
        chunk_size: int,
):
    # computational JAX operators, with ket action
    if isinstance(Ô.parent, ComputationalJaxOperator) and Ô.parent.is_ket_action:
        return kernels.local_value_kernel_squared_jax_ket_action_chunked

    # discrete or computational JAX operators with bra action
    if isinstance(Ô.parent, DiscreteJaxOperator):
        return kernels.local_value_kernel_squared_jax_chunked

    # non-JAX operators (args are (σp,mels)) with ket action
    if isinstance(Ô.parent, ComputationalOperator) and Ô.parent.is_ket_action:
        return kernels.local_value_squared_kernel_ket_action_chunked

    # discrete or computational numba operators with bra action
    return kernels.local_value_squared_kernel_chunked

#
#
#   standard computational operators
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: ComputationalOperator,
        chunk_size: int,
):
    if Ô.is_ket_action:
        return kernels.local_value_kernel_ket_action_chunked
    return kernels.local_value_kernel_chunked


#
#
#   standard computational JAX operators
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: ComputationalJaxOperator,
        chunk_size: int,
):
    if Ô.is_ket_action:
        return kernels.local_value_kernel_jax_ket_action_chunked
    return kernels.local_value_kernel_jax_chunked

#
#
#   standard PenaltyCost operators
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: PenaltyCost,
        chunk_size: int,
):
    # if the wrapped operator is ket-action, we must use the ket-action estimator
    if isinstance(Ô.parent, (ComputationalOperator, ComputationalJaxOperator)) and Ô.parent.is_ket_action:
        if isinstance(Ô.parent, ComputationalOperator):
            return kernels.local_value_kernel_ket_action_chunked

        if isinstance(Ô.parent, ComputationalJaxOperator):
            return kernels.local_value_kernel_jax_ket_action_chunked

    return kernels.local_value_kernel_penalty_cost_chunked

#
#
#   IECs are not supported for now

@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: InverseExpectationCost,
        chunk_size: int,
):
    return NO_CHUNKING


#
#
#   VarianceObservable not supported for now
@dispatch
def get_local_kernel(
        vstate: Union[MCState, NQXMCState],
        Ô: VarianceObservable,
        chunk_size: int,
):
    return NO_CHUNKING

#
#
#   Fallbacks
@expect.dispatch
@trace(tag="EXPECT_UNSPECIFIED")
def expect_chunking_unspecified(
        vstate: Union[MCState, NQXMCState],
        operator: AbstractObservable,
):
    return expect(vstate, operator, None)


@expect.dispatch(precedence=-10)
@trace(tag="EXPECT_FALLBACK")
def expect_fallback(
    vstate: Union[MCState, NQXMCState], operator: Union[AbstractOperator, AbstractObservable], chunk_size: int | tuple
):  # noqa: F811
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_grad method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists."
    )

    return expect(vstate, operator, None)

@expect.dispatch(precedence=-10)
@trace(tag="EXPECT_FALLBACK_SEQUENCE")
def expect_sequence_fallback(
    vstate: Union[MCState, NQXMCState], operator: Sequence[Union[AbstractOperator, AbstractObservable]], chunk_size: int | tuple
):  # noqa: F811
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_grad method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists."
    )

    return expect(vstate, operator, None)


#
#
#   Non-sequence paths
@expect.dispatch
@trace(tag="EXPECT_CHUNKED")
def expect_mcstate_operator_chunked(
    vstate: Union[MCState, NQXMCState], Ô: Union[AbstractOperator, AbstractObservable], chunk_size: int,
) -> Stats:  # noqa: F811

    with prof_section(
        "expect.chunked.resolve_kernel",
        cat="vqs.expect",
        args={
            "operator_type": type(Ô).__name__,
            "chunk_size": int(chunk_size),
        },
    ):
        local_estimator_fun = get_local_kernel(vstate, Ô, chunk_size)

    if local_estimator_fun is NO_CHUNKING:
        warnings.warn(
            f"Ignoring chunk_size={chunk_size} for operator {type(Ô).__name__} "
            f"because chunking is not supported for this operator type.",
            stacklevel=2,
        )
        return expect(vstate, Ô, None)

    with prof_section(
        "expect.chunked.resolve_args",
        cat="vqs.expect",
        args={
            "operator_type": type(Ô).__name__,
            "chunk_size": int(chunk_size),
        },
    ):
        σ, args = get_local_kernel_arguments(vstate, Ô)

    if strict_type(Ô) is PenaltyCost:
        penalty_factor = Ô.factor
    else:
        penalty_factor = None

    with prof_section(
        "expect.chunked.kernel",
        cat="vqs.expect",
        args={
            "operator_type": type(Ô).__name__,
            "chunk_size": int(chunk_size),
        },
    ) as sec:
        out = _expect_chunking(
            chunk_size,
            local_estimator_fun,
            vstate._apply_fun,
            vstate.sampler.machine_pow,
            vstate.parameters,
            vstate.model_state,
            σ,
            args,
            penalty_factor,
        )
        return sec.sync(out)


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect_chunking(
    chunk_size: int,
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    args: PyTree,
    penalty_factor: jnp.float64 = None,
) -> Stats:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    L_σ = local_value_kernel(logpsi, parameters, σ, args, chunk_size=chunk_size)

    if penalty_factor is not None:
        L_σ = penalty_factor * L_σ

    Ō_stats = statistics(L_σ.reshape((n_chains, -1)))

    return Ō_stats


#
#
#   Sequence paths
@trace(tag="EXPECT_CHUNKED_SEQUENCE")
@expect.dispatch
def expect_mcstate_operator_chunked_sequence(
    vstate: Union[MCState, NQXMCState],
    Ô: Sequence[Union[AbstractOperator, AbstractObservable]],
    chunk_size: int,
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
    n_chains = int(σ.shape[0])

    local_kernels: list[Callable] = []
    local_factored_kernels: list[Callable | None] = []
    use_chunked_kernels: list[bool] = []
    use_factored_kernels: list[bool] = []
    local_args: list[PyTree] = []
    local_scales: list[float] = []

    with prof_section(
        "expect.chunked.sequence.prepare",
        cat="vqs.expect",
        args={"n_operators": int(len(Ô_list)), "chunk_size": int(chunk_size)},
    ):
        for i, ô in enumerate(Ô_list):
            with prof_section(
                "expect.chunked.sequence.operator",
                cat="vqs.expect",
                args={"index": int(i), "operator_type": type(ô).__name__},
            ):
                # Discard σ returned by the argument builder: we enforce one shared sample batch.
                _, args = get_local_kernel_arguments(vstate, ô)
                local_estimator_fun = get_local_kernel(vstate, ô, chunk_size)

                if local_estimator_fun is NO_CHUNKING:
                    warnings.warn(
                        f"Ignoring chunk_size={chunk_size} for operator {type(ô).__name__} "
                        f"because chunking is not supported for this operator type.",
                        stacklevel=2,
                    )
                    local_estimator_fun = get_local_kernel(vstate, ô)
                    use_chunked_kernels.append(False)
                    factored_kernel = kernels.resolve_factored_local_kernel(
                        local_estimator_fun, chunked=False
                    )
                else:
                    use_chunked_kernels.append(True)
                    factored_kernel = kernels.resolve_factored_local_kernel(
                        local_estimator_fun, chunked=True
                    )

                local_kernels.append(local_estimator_fun)
                local_factored_kernels.append(factored_kernel)
                use_factored_kernels.append(factored_kernel is not None)
                local_args.append(args)
                local_scales.append(
                    float(ô.factor) if strict_type(ô) is PenaltyCost else 1.0
                )

    with prof_section(
        "expect.chunked.sequence.kernel",
        cat="vqs.expect",
        args={"n_operators": int(len(Ô_list)), "chunk_size": int(chunk_size)},
    ) as sec:
        out = _expect_sequence_chunked(
            int(chunk_size),
            tuple(local_kernels),
            tuple(local_factored_kernels),
            tuple(use_chunked_kernels),
            tuple(use_factored_kernels),
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            σ,
            n_chains,
            tuple(local_args),
            tuple(local_scales),
        )
        return sec.sync(out)

@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect_sequence_chunked_single_local(
    chunk_size: int,
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

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args, chunk_size=chunk_size)

    return L_σ


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 9))
def _expect_sequence_chunked(
    chunk_size: int,
    local_value_kernels: tuple[Callable, ...],
    local_value_factored_kernels: tuple[Callable | None, ...],
    use_chunked_kernels: tuple[bool, ...],
    use_factored_kernels: tuple[bool, ...],
    model_apply_fun: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    n_chains: int,
    local_value_args: tuple[PyTree, ...],
    local_scales: tuple[float, ...],
) -> Stats:
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    if model_state is None:
        model_state = {}

    logpsi = lambda w, sigma: model_apply_fun({"params": w, **model_state}, sigma)
    has_factored = any(use_factored_kernels)
    if has_factored:
        logpsi_σ = nkjax.apply_chunked(
            lambda sigma: logpsi(parameters, sigma),
            in_axes=0,
            chunk_size=chunk_size,
        )(σ)
    else:
        logpsi_σ = None

    variables = {"params": parameters, **model_state}
    total_loc = jnp.zeros((σ.shape[0],), dtype=jnp.result_type(float))
    for i, local_value_kernel in enumerate(local_value_kernels):
        if use_factored_kernels[i]:
            if use_chunked_kernels[i]:
                loc_i = local_value_factored_kernels[i](
                    logpsi_σ,
                    logpsi,
                    parameters,
                    σ,
                    local_value_args[i],
                    chunk_size=chunk_size,
                )
            else:
                loc_i = local_value_factored_kernels[i](
                    logpsi_σ,
                    logpsi,
                    parameters,
                    σ,
                    local_value_args[i],
                )
        else:
            if use_chunked_kernels[i]:
                loc_i = local_value_kernel(
                    logpsi,
                    parameters,
                    σ,
                    local_value_args[i],
                    chunk_size=chunk_size,
                )
            else:
                loc_i = local_value_kernel(
                    model_apply_fun,
                    variables,
                    σ,
                    local_value_args[i],
                )

        scale_i = jnp.asarray(local_scales[i], dtype=loc_i.dtype)
        total_loc = total_loc + scale_i * loc_i

    return statistics(total_loc.reshape((n_chains, -1)))
