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
NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

from functools import partial
from typing import Callable, Sequence, Union
from weakref import WeakKeyDictionary

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.stats import Stats, statistics
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
)

from netket.vqs.mc.mc_state.state import MCState

from ...mc import get_local_kernel, get_local_kernel_arguments
from ..kernels import resolve_factored_local_kernel
from ....operators import PenaltyCost, penalty_is_linear, penalty_linear_scale
from ....utils.errors import ExpectationValueError
from ....vqs.mc.mc_state.state import MCState as NQXMCState
from ....profile import section as prof_section
from neuralqx import cfg


_LOCAL_KERNEL_CACHE_WEAK = WeakKeyDictionary()
_LOCAL_KERNEL_CACHE_FALLBACK = {}


def _use_fused_kernels() -> bool:
    return bool(cfg.get("FUSED_KERNELS"))


def _penalty_scale(operator: AbstractOperator) -> float:
    """
    Linear PenaltyCost contributes its factor outside the local kernel.

    Nonlinear penalty subclasses, including InverseExpectationCost, fold their
    objective derivative into dynamic affine local-kernel arguments. Applying an
    additional external scale would double-count them.
    """
    if isinstance(operator, PenaltyCost) and penalty_is_linear(operator):
        return float(penalty_linear_scale(operator))
    return 1.0


def _get_local_kernel_cached(vstate, operator):
    """
    Resolve and cache local-kernel dispatch for a (vstate type, operator instance) pair.
    """
    key = (type(vstate),)

    def _factory():
        return get_local_kernel(vstate, operator)

    try:
        cache = _LOCAL_KERNEL_CACHE_WEAK.get(operator)
        if cache is None:
            cache = {}
            _LOCAL_KERNEL_CACHE_WEAK[operator] = cache
        kernel = cache.get(key)
        if kernel is None:
            kernel = _factory()
            cache[key] = kernel
        return kernel
    except TypeError:
        # Fallback for objects not supporting weak references.
        fb_key = (id(operator), type(vstate))
        kernel = _LOCAL_KERNEL_CACHE_FALLBACK.get(fb_key)
        if kernel is None:
            kernel = _factory()
            _LOCAL_KERNEL_CACHE_FALLBACK[fb_key] = kernel
        return kernel


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _forces_expect_hermitian_sequence_fused(
    local_value_kernels: tuple[Callable, ...],
    local_value_factored_kernels: tuple[Callable | None, ...],
    use_factored_kernels: tuple[bool, ...],
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: tuple[PyTree, ...],
    local_scales: tuple[float, ...],
) -> tuple[Stats, PyTree, PyTree]:
    """
    Compute forces for a sequence of Hermitian operators with one VJP.

    Per-operator local estimators are evaluated separately on the same sample batch
    and accumulated as:
        L_total = sum_i scale_i * L_i
    then a single covariance/VJP is performed on L_total.
    """
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    n_samples = σ.shape[0]
    if model_state is None:
        model_state = {}

    is_mutable = mutable is not False
    log_sigma, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    logpsi = lambda pars, sigma: model_apply_fun({"params": pars, **model_state}, sigma)
    variables = {"params": parameters, **model_state}
    total_loc = jnp.zeros((σ.shape[0],), dtype=jnp.result_type(float))

    for i, local_value_kernel in enumerate(local_value_kernels):
        if use_factored_kernels[i]:
            loc_i = local_value_factored_kernels[i](
                log_sigma,
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

    Ō = statistics(total_loc.reshape((n_chains, -1)))
    centered = total_loc - Ō.mean

    Ō_grad = vjp_fun(jnp.conjugate(centered) / n_samples)[0]
    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, Ō_grad, new_model_state


@dispatch
def expect_and_forces(
        vstate: Union[MCState, NQXMCState],
        Ô: Sequence[AbstractOperator],
        chunk_size: None,
        *,
        mutable: CollectionFilter = False,
) -> tuple[Stats, PyTree]:
    """
    Extends the NetKet logic of `expect_and_forces()` to apply to a sequence of operators.
    """

    # ensure list format
    Ô_list = list(Ô)

    # baseline check for empty list of operators
    if len(Ô_list) == 0:
        raise ExpectationValueError("expect_and_forces")

    # get the vstate.samples to ensure that we use the same ones for every Ô
    σ = vstate.samples

    local_kernels: list[Callable] = []
    local_factored_kernels: list[Callable | None] = []
    use_factored_kernels: list[bool] = []
    local_args: list[PyTree] = []
    local_scales: list[float] = []
    use_fused = _use_fused_kernels()

    with prof_section(
        "expect_and_forces.sequence.prepare",
        cat="vqs.forces",
        args={"n_operators": int(len(Ô_list))},
    ):
        for i, ô in enumerate(Ô_list):
            with prof_section(
                "expect_and_forces.sequence.operator",
                cat="vqs.forces",
                args={"index": int(i), "operator_type": type(ô).__name__},
            ):
                # Discard σ returned by the argument builder: we enforce one shared sample batch.
                _, args_op = get_local_kernel_arguments(vstate, ô)
                local_kernel = _get_local_kernel_cached(vstate, ô)
                factored_kernel = resolve_factored_local_kernel(local_kernel, chunked=False)
                local_kernels.append(local_kernel)
                local_factored_kernels.append(factored_kernel)
                use_factored_kernels.append(factored_kernel is not None)
                local_args.append(args_op)
                local_scales.append(_penalty_scale(ô))

    with prof_section(
        "expect_and_forces.sequence.kernel",
        cat="vqs.forces",
        args={
            "n_operators": int(len(Ô_list)),
            "fused_kernels": bool(use_fused),
        },
    ) as sec:
        if use_fused:
            stats_sum, grad_sum, new_model_state = _forces_expect_hermitian_sequence_fused(
                tuple(local_kernels),
                tuple(local_factored_kernels),
                tuple(use_factored_kernels),
                vstate._apply_fun,
                mutable,
                vstate.parameters,
                vstate.model_state,
                σ,
                tuple(local_args),
                tuple(local_scales),
            )
        else:
            n_chains = int(σ.shape[0])
            total_loc = None
            grad_sum = None
            new_model_state = None
            for i, local_kernel in enumerate(local_kernels):
                loc_i, grad_i, op_model_state = forces_expect_hermitian_sequence(
                    local_kernel,
                    vstate._apply_fun,
                    mutable,
                    vstate.parameters,
                    vstate.model_state,
                    σ,
                    local_args[i],
                )
                sf = float(local_scales[i])
                loc_i = sf * loc_i
                grad_i = jax.tree_util.tree_map(lambda g: sf * g, grad_i)

                total_loc = loc_i if total_loc is None else total_loc + loc_i
                grad_sum = (
                    grad_i
                    if grad_sum is None
                    else jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, grad_i)
                )
                if mutable is not False and op_model_state is not None:
                    new_model_state = op_model_state

            stats_sum = statistics(total_loc.reshape((n_chains, -1)))
        sec.sync((stats_sum, grad_sum))

    if mutable is not False:
        vstate.model_state = new_model_state

    return stats_sum, grad_sum


@dispatch
def expect_and_forces(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    Ô: AbstractOperator,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
) -> tuple[Stats, PyTree]:
    with prof_section(
        "expect_and_forces.resolve_args",
        cat="vqs.forces",
        args={"operator_type": type(Ô).__name__},
    ):
        σ, args = get_local_kernel_arguments(vstate, Ô)

    with prof_section(
        "expect_and_forces.resolve_kernel",
        cat="vqs.forces",
        args={"operator_type": type(Ô).__name__},
    ):
        local_estimator_fun = _get_local_kernel_cached(vstate, Ô)

    scale_factor = _penalty_scale(Ô)

    with prof_section(
        "expect_and_forces.kernel",
        cat="vqs.forces",
        args={"operator_type": type(Ô).__name__},
    ) as sec:
        Ō, Ō_grad, new_model_state = forces_expect_hermitian(
            local_estimator_fun,
            vstate._apply_fun,
            mutable,
            vstate.parameters,
            vstate.model_state,
            σ,
            args,
            scale_factor,
        )
        sec.sync((Ō, Ō_grad))

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


@partial(jax.jit, static_argnums=(0, 1, 2))
def forces_expect_hermitian_sequence(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> tuple[PyTree, PyTree, PyTree]:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    n_samples = σ.shape[0]

    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )

    # keep track of the local estimator value before subtracting the mean otherwise the printed
    # value to the user will be wrong
    O_loc_copy = O_loc.copy()

    Ō = statistics(O_loc.reshape((n_chains, -1)))

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    new_model_state = new_model_state[0] if is_mutable else None

    return (
        # Ō,
        O_loc_copy,
        Ō_grad,
        new_model_state,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def forces_expect_hermitian(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
    scale_factor: float = 1.0,
) -> tuple[PyTree, PyTree, PyTree]:
    """
    Follows the NetKet logic but also returns the local estimator
    """

    n_chains = σ.shape[0]

    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    n_samples = σ.shape[0]

    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )

    sf = jnp.asarray(scale_factor, dtype=O_loc.dtype)
    O_loc = sf * O_loc

    Ō = statistics(O_loc.reshape((n_chains, -1)))

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    new_model_state = new_model_state[0] if is_mutable else None

    return (
        Ō,
        Ō_grad,
        new_model_state,
    )
