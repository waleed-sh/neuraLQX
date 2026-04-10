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
NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""


from functools import partial
from typing import Callable, Sequence, Union, Any
import warnings
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
from netket.operator._abstract_observable import AbstractObservable
from netket.vqs.mc.mc_state.state import MCState

from .expect_chunked import NO_CHUNKING
from ..kernels import resolve_factored_local_kernel
from ...mc import get_local_kernel, get_local_kernel_arguments
from ....operators import PenaltyCost, InverseExpectationCost
from ....utils.errors import ExpectationValueError
from ....vqs.mc.mc_state.state import MCState as NQXMCState
from ....profile import section as prof_section
from ....configs import cfg

from neuralqx.vqs import expect_and_forces
from .expect_forces import forces_expect_hermitian_sequence


_LOCAL_KERNEL_CACHE_WEAK = WeakKeyDictionary()
_LOCAL_KERNEL_CACHE_FALLBACK = {}


def _use_fused_kernels() -> bool:
    return bool(cfg.get("FUSED_KERNELS"))


def _penalty_scale(operator: AbstractOperator) -> float:
    """
    PenaltyCost contributes its factor outside the local kernel, except IEC.

    InverseExpectationCost already includes `factor` in its local-kernel
    parametrisation, so an external scaling would double-count it.
    """
    if isinstance(operator, PenaltyCost) and not isinstance(operator, InverseExpectationCost):
        return float(operator.factor)
    return 1.0


def _get_local_kernel_cached(vstate, operator, chunk_size):
    """
    Resolve and cache chunk-aware local-kernel dispatch.
    """
    key = (type(vstate), int(chunk_size))

    def _factory():
        return get_local_kernel(vstate, operator, chunk_size)

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
        fb_key = (id(operator), type(vstate), int(chunk_size))
        kernel = _LOCAL_KERNEL_CACHE_FALLBACK.get(fb_key)
        if kernel is None:
            kernel = _factory()
            _LOCAL_KERNEL_CACHE_FALLBACK[fb_key] = kernel
        return kernel


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 10))
def _forces_expect_hermitian_sequence_fused_chunked(
    chunk_size: int,
    local_value_kernels: tuple[Callable, ...],
    local_value_factored_kernels: tuple[Callable | None, ...],
    use_chunked_kernels: tuple[bool, ...],
    use_factored_kernels: tuple[bool, ...],
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    n_chains: int,
    local_value_args: tuple[PyTree, ...],
    local_scales: tuple[float, ...],
) -> tuple[Stats, PyTree, PyTree]:
    """
    Chunked sequence-force kernel with one VJP across the summed local estimator.
    """
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ.shape[-1]))

    n_samples = σ.shape[0]
    if model_state is None:
        model_state = {}

    logpsi = lambda pars, sigma: model_apply_fun({"params": pars, **model_state}, sigma)
    has_factored = any(use_factored_kernels)
    if has_factored:
        log_sigma = nkjax.apply_chunked(
            lambda sigma: logpsi(parameters, sigma),
            in_axes=0,
            chunk_size=chunk_size,
        )(σ)
    else:
        log_sigma = None

    variables = {"params": parameters, **model_state}
    total_loc = jnp.zeros((σ.shape[0],), dtype=jnp.result_type(float))

    for i, local_value_kernel in enumerate(local_value_kernels):
        if use_factored_kernels[i]:
            if use_chunked_kernels[i]:
                loc_i = local_value_factored_kernels[i](
                    log_sigma,
                    logpsi,
                    parameters,
                    σ,
                    local_value_args[i],
                    chunk_size=chunk_size,
                )
            else:
                loc_i = local_value_factored_kernels[i](
                    log_sigma,
                    logpsi,
                    parameters,
                    σ,
                    local_value_args[i],
                )
        else:
            if use_chunked_kernels[i]:
                loc_i = local_value_kernel(
                    model_apply_fun,
                    variables,
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

    Ō = statistics(total_loc.reshape((n_chains, -1)))
    centered = total_loc - Ō.mean

    if mutable is not False:
        raise NotImplementedError

    vjp_fun_chunked = nkjax.vjp_chunked(
        lambda w, ms, σi: model_apply_fun({"params": w, **ms}, σi),
        parameters,
        model_state,
        σ,
        conjugate=True,
        chunk_size=chunk_size,
        chunk_argnums=2,
        nondiff_argnums=(1, 2),
    )
    (Ō_grad,) = vjp_fun_chunked((jnp.conjugate(centered) / n_samples))

    return Ō, Ō_grad, None


#
#
#   Unspecified batch size

@expect_and_forces.dispatch
def expect_and_forces_chunking_unspecified(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Union[AbstractOperator, AbstractObservable],
    **kwargs,
):
    return expect_and_forces(vstate, operator, None, **kwargs)

# if no implementation exists for batched, run the code unbatched
@expect_and_forces.dispatch(precedence=-10)
def expect_and_forces_fallback(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Union[AbstractOperator, AbstractObservable],
    chunk_size: Any,
    *args,
    **kwargs,
):
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_forces method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists.",
        stacklevel=2,
    )

    return expect_and_forces(vstate, operator, None, *args, **kwargs)

# same for sequnce paths
@expect_and_forces.dispatch(precedence=-10)
def expect_and_forces_sequence_fallback(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Sequence[Union[AbstractOperator, AbstractObservable]],
    chunk_size: Any,
    *args,
    **kwargs,
):
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_forces method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists.",
        stacklevel=2,
    )

    return expect_and_forces(vstate, operator, None, *args, **kwargs)

#
#
#   register the integer chunk size with plum for both paths

@expect_and_forces.dispatch
def expect_and_forces_impl(  # noqa: F811
    vstate: MCState,
    Ô: Union[AbstractOperator, AbstractObservable],
    chunk_size: int,
    *,
    mutable: CollectionFilter = False,
) -> tuple[Stats, PyTree]:

    with prof_section(
        "expect_and_forces.chunked.resolve_args",
        cat="vqs.forces",
        args={"operator_type": type(Ô).__name__, "chunk_size": int(chunk_size)},
    ):
        σ, args = get_local_kernel_arguments(vstate, Ô)
    with prof_section(
        "expect_and_forces.chunked.resolve_kernel",
        cat="vqs.forces",
        args={"operator_type": type(Ô).__name__, "chunk_size": int(chunk_size)},
    ):
        local_estimator_fun = _get_local_kernel_cached(vstate, Ô, chunk_size)

    if local_estimator_fun is NO_CHUNKING:
        warnings.warn(
            f"Ignoring chunk_size={chunk_size} for operator {type(Ô).__name__} "
            f"because chunking is not supported.",
            stacklevel=2,
        )
        return expect_and_forces(vstate, Ô, None, mutable=mutable)

    scale_factor = _penalty_scale(Ô)

    with prof_section(
        "expect_and_forces.chunked.kernel",
        cat="vqs.forces",
        args={"operator_type": type(Ô).__name__, "chunk_size": int(chunk_size)},
    ) as sec:
        Ō, Ō_grad, new_model_state = forces_expect_hermitian_chunked(
            chunk_size,
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

@expect_and_forces.dispatch
def expect_and_forces(
        vstate: Union[MCState, NQXMCState],
        Ô: Sequence[Union[AbstractOperator, AbstractObservable]],
        chunk_size: int,
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

    if mutable is not False:
        warnings.warn(
            f"Ignoring chunk_size={chunk_size} for sequence expect_and_forces with mutable={mutable} "
            "because chunked mutable VJP is not supported.",
            stacklevel=2,
        )
        return expect_and_forces(vstate, Ô, None, mutable=mutable)

    # get the vstate.samples to ensure that we use the same ones for every Ô
    σ = vstate.samples

    # store the number of MC chains
    n_chains = σ.shape[0]

    if σ.ndim >= 3:
        # flatten the chain dimension if needed
        σ = jax.lax.collapse(σ, 0, 2)

    local_kernels: list[Callable] = []
    local_factored_kernels: list[Callable | None] = []
    local_args: list[PyTree] = []
    local_scales: list[float] = []
    use_chunked_kernels: list[bool] = []
    use_factored_kernels: list[bool] = []
    use_fused = _use_fused_kernels()

    with prof_section(
        "expect_and_forces.chunked.sequence.prepare",
        cat="vqs.forces",
        args={"n_operators": int(len(Ô_list)), "chunk_size": int(chunk_size)},
    ):
        for i, ô in enumerate(Ô_list):
            with prof_section(
                "expect_and_forces.chunked.sequence.operator",
                cat="vqs.forces",
                args={"index": int(i), "operator_type": type(ô).__name__},
            ):
                # Discard σ returned by the argument builder: we enforce one shared sample batch.
                _, args_op = get_local_kernel_arguments(vstate, ô)
                local_estimator_fun = _get_local_kernel_cached(vstate, ô, chunk_size)

                if local_estimator_fun is NO_CHUNKING:
                    warnings.warn(
                        f"Ignoring chunk_size={chunk_size} for operator {type(ô).__name__} "
                        f"because chunking is not supported.",
                        stacklevel=2,
                    )
                    local_estimator_fun = get_local_kernel(vstate, ô)
                    use_chunked_kernels.append(False)
                    factored_kernel = resolve_factored_local_kernel(
                        local_estimator_fun, chunked=False
                    )
                else:
                    use_chunked_kernels.append(True)
                    factored_kernel = resolve_factored_local_kernel(
                        local_estimator_fun, chunked=True
                    )

                local_kernels.append(local_estimator_fun)
                local_factored_kernels.append(factored_kernel)
                use_factored_kernels.append(factored_kernel is not None)
                local_args.append(args_op)
                local_scales.append(_penalty_scale(ô))

    with prof_section(
        "expect_and_forces.chunked.sequence.kernel",
        cat="vqs.forces",
        args={
            "n_operators": int(len(Ô_list)),
            "chunk_size": int(chunk_size),
            "fused_kernels": bool(use_fused),
        },
    ) as sec:
        if use_fused:
            stats_sum, grad_sum, _new_model_state = _forces_expect_hermitian_sequence_fused_chunked(
                int(chunk_size),
                tuple(local_kernels),
                tuple(local_factored_kernels),
                tuple(use_chunked_kernels),
                tuple(use_factored_kernels),
                vstate._apply_fun,
                mutable,
                vstate.parameters,
                vstate.model_state,
                σ,
                int(n_chains),
                tuple(local_args),
                tuple(local_scales),
            )
        else:
            total_loc = None
            grad_sum = None
            for i, local_kernel in enumerate(local_kernels):
                if use_chunked_kernels[i]:
                    loc_i, grad_i, _ = forces_expect_hermitian_sequence_chunked(
                        int(chunk_size),
                        local_kernel,
                        vstate._apply_fun,
                        mutable,
                        vstate.parameters,
                        vstate.model_state,
                        σ,
                        local_args[i],
                    )
                else:
                    loc_i, grad_i, _ = forces_expect_hermitian_sequence(
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

            stats_sum = statistics(total_loc.reshape((int(n_chains), -1)))
        sec.sync((stats_sum, grad_sum))

    return stats_sum, grad_sum



#
#
#   Kernels

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def forces_expect_hermitian_chunked(
    chunk_size: int,
    local_value_kernel_chunked: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
    scale_factor: float = 1.0
) -> tuple[PyTree, PyTree, PyTree]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0]

    O_loc = local_value_kernel_chunked(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
        chunk_size=chunk_size,
    )

    sf = jnp.asarray(scale_factor, dtype=O_loc.dtype)
    O_loc = sf * O_loc

    Ō = statistics(O_loc.reshape(σ_shape[:-1]))

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    if mutable is False:
        vjp_fun_chunked = nkjax.vjp_chunked(
            lambda w, ms, σ: model_apply_fun({"params": w, **ms}, σ),
            parameters,
            model_state,
            σ,
            conjugate=True,
            chunk_size=chunk_size,
            chunk_argnums=2,
            nondiff_argnums=(1, 2),
        )
        new_model_state = None
    else:
        raise NotImplementedError

    (Ō_grad,) = vjp_fun_chunked(
        (jnp.conjugate(O_loc) / n_samples),
    )

    return Ō, Ō_grad, new_model_state


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def forces_expect_hermitian_sequence_chunked(
        chunk_size: int,
        local_value_kernel_chunked: Callable,
        model_apply_fun: Callable,
        mutable: CollectionFilter,
        parameters: PyTree,
        model_state: PyTree,
        σ: jnp.ndarray,
        local_value_args: PyTree,
) -> tuple[PyTree, PyTree, PyTree]:
    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0]

    O_loc = local_value_kernel_chunked(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
        chunk_size=chunk_size,
    )

    # keep track of the local estimator value before subtracting the mean otherwise the printed
    # value to the user will be wrong
    O_loc_copy = O_loc.copy()

    Ō = statistics(O_loc.reshape(σ_shape[:-1]))

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    if mutable is False:
        vjp_fun_chunked = nkjax.vjp_chunked(
            lambda w, ms, σ: model_apply_fun({"params": w, **ms}, σ),
            parameters,
            model_state,
            σ,
            conjugate=True,
            chunk_size=chunk_size,
            chunk_argnums=2,
            nondiff_argnums=(1, 2),
        )
        new_model_state = None
    else:
        raise NotImplementedError

    (Ō_grad, ) = vjp_fun_chunked(
        (jnp.conjugate(O_loc) / n_samples),
    )

    return O_loc_copy, Ō_grad, new_model_state
