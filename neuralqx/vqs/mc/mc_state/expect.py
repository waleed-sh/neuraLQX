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

from netket.operator._abstract_observable import AbstractObservable

from netket.stats import Stats, statistics as statistics
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
from ....profile import section as prof_section
from neuralqx import cfg


def _use_fused_kernels() -> bool:
    return bool(cfg.get("FUSED_KERNELS"))


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
    with prof_section(
        "expect.resolve_args",
        cat="vqs.expect",
        args={"operator_type": type(Ô).__name__},
    ):
        σ, args = get_local_kernel_arguments(vstate, Ô)
    with prof_section(
        "expect.resolve_kernel",
        cat="vqs.expect",
        args={"operator_type": type(Ô).__name__},
    ):
        local_estimator_fun = get_local_kernel(vstate, Ô)

    if strict_type(Ô) is PenaltyCost:
        penalty_factor = Ô.factor
    else:
        penalty_factor = None

    with prof_section(
        "expect.kernel",
        cat="vqs.expect",
        args={"operator_type": type(Ô).__name__},
    ) as sec:
        out = _expect(
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

    local_kernels: list[Callable] = []
    local_factored_kernels: list[Callable | None] = []
    use_factored_kernels: list[bool] = []
    local_args: list[PyTree] = []
    local_scales: list[float] = []
    use_fused = _use_fused_kernels()

    with prof_section(
        "expect.sequence.prepare",
        cat="vqs.expect",
        args={"n_operators": int(len(Ô_list))},
    ):
        for i, ô in enumerate(Ô_list):
            with prof_section(
                "expect.sequence.operator",
                cat="vqs.expect",
                args={"index": int(i), "operator_type": type(ô).__name__},
            ):
                # Discard σ returned by the argument builder: we enforce one shared sample batch.
                _, args = get_local_kernel_arguments(vstate, ô)
                local_kernel = get_local_kernel(vstate, ô)
                factored_kernel = kernels.resolve_factored_local_kernel(
                    local_kernel, chunked=False
                )

                local_kernels.append(local_kernel)
                local_factored_kernels.append(factored_kernel)
                use_factored_kernels.append(factored_kernel is not None)
                local_args.append(args)
                local_scales.append(
                    float(ô.factor) if strict_type(ô) is PenaltyCost else 1.0
                )

    with prof_section(
        "expect.sequence.kernel",
        cat="vqs.expect",
        args={
            "n_operators": int(len(Ô_list)),
            "fused_kernels": bool(use_fused),
        },
    ) as sec:
        if use_fused:
            out = _expect_sequence_fused(
                tuple(local_kernels),
                tuple(local_factored_kernels),
                tuple(use_factored_kernels),
                vstate._apply_fun,
                vstate.parameters,
                vstate.model_state,
                σ,
                tuple(local_args),
                tuple(local_scales),
            )
        else:
            n_chains = int(σ.shape[0])
            model_state = vstate.model_state or {}
            total_loc = None
            for i, local_kernel in enumerate(local_kernels):
                loc_i = _expect_sequence(
                    local_kernel,
                    vstate._apply_fun,
                    vstate.sampler.machine_pow,
                    vstate.parameters,
                    model_state,
                    σ,
                    local_args[i],
                )
                scale_i = jnp.asarray(local_scales[i], dtype=loc_i.dtype)
                loc_i = scale_i * loc_i
                total_loc = loc_i if total_loc is None else total_loc + loc_i

            out = statistics(total_loc.reshape((n_chains, -1)))
        return sec.sync(out)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _expect_sequence_fused(
    local_value_kernels: tuple[Callable, ...],
    local_value_factored_kernels: tuple[Callable | None, ...],
    use_factored_kernels: tuple[bool, ...],
    model_apply_fun: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: tuple[PyTree, ...],
    local_scales: tuple[float, ...],
) -> Stats:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    if model_state is None:
        model_state = {}

    logpsi = lambda w, sigma: model_apply_fun({"params": w, **model_state}, sigma)
    has_factored = any(use_factored_kernels)
    if has_factored:
        logpsi_σ = logpsi(parameters, σ)
    else:
        logpsi_σ = None

    variables = {"params": parameters, **model_state}
    total_loc = jnp.zeros((σ.shape[0],), dtype=jnp.result_type(float))
    for i, local_value_kernel in enumerate(local_value_kernels):
        if use_factored_kernels[i]:
            loc_i = local_value_factored_kernels[i](
                logpsi_σ,
                logpsi,
                parameters,
                σ,
                local_value_args[i],
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

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)

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

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)

    if penalty_factor is not None:
        L_σ = penalty_factor * L_σ

    Ō_stats = statistics(L_σ.reshape((n_chains, -1)))

    return Ō_stats
