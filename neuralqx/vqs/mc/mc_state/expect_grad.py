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
from typing import Callable, Optional, Sequence, Union
from weakref import WeakKeyDictionary

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket import config
from netket.utils import dispatch
from netket.utils.types import PyTree
from netket.stats import Stats, statistics

from netket.operator import (
    AbstractOperator,
    Squared,
)

from netket.experimental.observable import VarianceObservable

from netket.vqs.mc.mc_state.state import MCState

from ..common import get_local_kernel_arguments, get_local_kernel
from ..kernels import resolve_factored_local_kernel
from .state import MCState as NQXMCState
from ....operators import InverseExpectationCost, PenaltyCost
from ....operators.types.computational_operator import ComputationalOperator
from ....utils.errors import ExpectationValueError, InvalidOperatorsSequenceError
from ....utils.parsing import strict_type
from ....profile import section as prof_section
from ....vqs import expect_and_grad, expect_and_forces
from ....vqs.mc.common import force_to_grad
from ....configs import cfg


_LOCAL_KERNEL_CACHE_WEAK = WeakKeyDictionary()
_LOCAL_KERNEL_CACHE_FALLBACK = {}


def _use_fused_kernels() -> bool:
    return bool(cfg.get("FUSED_KERNELS"))


def _use_experimental_grad() -> bool:
    """Return whether experimental non-Hermitian gradient routing is enabled.

    The bi-covariance route is available only when the experimental namespace is
    enabled and the dedicated gradient flag is enabled.
    """
    return bool(cfg.get("EXPERIMENTAL")) and bool(cfg.get("EXPERIMENTAL_GRAD"))


def _try_biadjoint_single(vstate, Ô, chunk_size, *, mutable):
    """Attempt the bi-covariance path for one non-Hermitian operator.

    Args:
        vstate: Variational state.
        Ô: Operator to differentiate.
        chunk_size: Optional chunk size passed to the forces path.
        mutable: Flax mutable collection filter.

    Returns:
        ``(stats, grad)`` on success, or ``None`` when an adjoint is unavailable.
    """
    from neuralqx.experimental.vqs.mc.mc_state.expect_grad_biadjoint import (
        expect_and_grad_biadjoint,
    )
    return expect_and_grad_biadjoint(vstate, Ô, chunk_size, mutable=mutable)


def _try_biadjoint_sequence(vstate, Ô_list, chunk_size, *, mutable):
    """Attempt the bi-covariance path for a non-Hermitian operator sequence.

    Args:
        vstate: Variational state.
        Ô_list: Operator sequence.
        chunk_size: Optional chunk size passed to the forces path.
        mutable: Flax mutable collection filter.

    Returns:
        ``(stats, grad)`` on success, or ``None`` when any adjoint is unavailable.
    """
    from neuralqx.experimental.vqs.mc.mc_state.expect_grad_biadjoint import (
        expect_and_grad_biadjoint_sequence,
    )
    return expect_and_grad_biadjoint_sequence(vstate, Ô_list, chunk_size, mutable=mutable)


def _get_local_kernel_cached(vstate, operator, *, chunk_size=None):
    """
    Resolve and cache local-kernel dispatch for a (vstate type, operator, chunk_size) triple.
    """
    key = (type(vstate), None if chunk_size is None else int(chunk_size))

    def _factory():
        if chunk_size is None:
            return get_local_kernel(vstate, operator)
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
        fb_key = (id(operator), type(vstate), key[1])
        kernel = _LOCAL_KERNEL_CACHE_FALLBACK.get(fb_key)
        if kernel is None:
            kernel = _factory()
            _LOCAL_KERNEL_CACHE_FALLBACK[fb_key] = kernel
        return kernel



@partial(jax.jit, static_argnums=(0, 1))
def _locals_only(
        local_kernel,
        model_apply_fun,
        parameters,
        model_state,
        σ,
        local_args,
):
    """
    A helper function to avoid mixing up lambdas and jit-compiled paths
    """
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    return local_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_args,
    )


@partial(jax.jit, static_argnums=(0, 1, 2),)
def _variance_vjp_and_local(
    local_kernel_O,
    local_kernel_O2,
    afun,
    params,
    model_state,
    σ,
    args_O,
    args_O2,
):
    n_chains = σ.shape[0]
    σ_flat = σ.reshape(-1, σ.shape[-1])

    def log_pdf(p, s):
        return 2 * afun({"params": p, **model_state}, s).real

    # expectation of variance (a scalar)
    def variance_expect(p):
        # mean(O)
        O_mean = nkjax.expect(
            log_pdf,
            lambda pp, ss: local_kernel_O(afun, {"params": pp, **model_state}, ss, args_O),
            p,
            σ_flat,
            n_chains = n_chains,
        )[0]

        # local O²
        O2_loc = local_kernel_O2(afun, {"params": p, **model_state}, σ_flat, args_O2)

        return nkjax.expect(
            log_pdf,
            lambda pp, ss: (O2_loc - O_mean**2).real,
            p,
            σ_flat,
            n_chains=n_chains,
        )

    # vjp
    var_val, vjp_fun, var_stats = nkjax.vjp(
        variance_expect, params, has_aux=True, conjugate=True,
    )
    grad = vjp_fun(jnp.ones_like(var_val))[0]

    # local estimator L_σ for downstream summation
    O_loc = local_kernel_O(afun, {"params": params, **model_state}, σ_flat, args_O)
    O2_loc = local_kernel_O2(afun, {"params": params, **model_state}, σ_flat, args_O2)
    L_σ = (O2_loc - (jnp.mean(O_loc))**2).real

    return L_σ, var_stats, grad


@expect_and_grad.dispatch
def expect_and_grad_variance(
    vstate: Union[MCState, NQXMCState],
    Ô: VarianceObservable,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
    use_covariance = None,
):
    """
    Variance cost specific path for the gradient
    """
    with prof_section("expect_and_grad.variance.resolve_args", cat="vqs.grad"):
        σ, (args_O, args_O2) = get_local_kernel_arguments(vstate, Ô)

    with prof_section("expect_and_grad.variance.resolve_kernels", cat="vqs.grad"):
        local_O = get_local_kernel(vstate, Ô.operator)
        local_O2 = get_local_kernel(vstate, Ô.operator_squared)

    with prof_section("expect_and_grad.variance.kernel", cat="vqs.grad") as sec:
        L_σ, stats, grad = _variance_vjp_and_local(
            local_O,
            local_O2,
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            σ,
            args_O,
            args_O2,
        )
        sec.sync((stats, grad))

    return stats, grad


@expect_and_grad.dispatch
def expect_and_grad_default_formula_sequence(
    vstate: Union[MCState, NQXMCState],
    Ô: Sequence[AbstractOperator],
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    """
    Extended to handle multiple operators. If use_covariance=True, we use the new
    expect_and_forces() that can handle a list.
    """

    # sanity check
    if isinstance(Ô, list) and len(Ô) == 0:
        raise ExpectationValueError("expect_and_grad")

    # use_covariance unspecified and we have to see based on operators
    if use_covariance is None:
        # check all operators in the list
        use_covariance = all(op.is_hermitian for op in Ô)

    # now based on that, we will execute different functions either hermitian or nonhermitian
    if use_covariance:
        with prof_section(
            "expect_and_grad.sequence.covariance",
            cat="vqs.grad",
            args={"n_operators": int(len(Ô))},
        ):
            Ō, Ō_grad = expect_and_forces(vstate, Ô, chunk_size, *args, mutable=mutable)
            # convert forces to gradient
            Ō_grad = force_to_grad(Ō_grad, vstate.parameters)
            return Ō, Ō_grad
    else:
        # Experimental bi-covariance path for non-Hermitian operator sequences.
        if _use_experimental_grad():
            with prof_section(
                "expect_and_grad.sequence.biadjoint",
                cat="vqs.grad",
                args={"n_operators": int(len(Ô))},
            ) as sec:
                result = _try_biadjoint_sequence(vstate, Ô, chunk_size, mutable=mutable)
                sec.sync(result)
            if result is not None:
                return result
            # At least one adjoint unavailable — fall through to generic path.

        # non-hermitian path
        with prof_section(
            "expect_and_grad.sequence.nonhermitian",
            cat="vqs.grad",
            args={"n_operators": int(len(Ô))},
        ):
            return expect_and_grad_nonhermitian(
                vstate, Ô, chunk_size, *args, mutable=mutable
            )

@expect_and_grad.dispatch
def expect_and_grad_default_formula(
    vstate: Union[MCState, NQXMCState],
    Ô: AbstractOperator,
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    if use_covariance is None:
        use_covariance = Ô.is_hermitian

    if use_covariance:
        # Implementation of expect_and_grad for `use_covariance == True` (due to the Literal[True]
        # type in the signature).` This case is equivalent to the composition of the
        # `expect_and_forces` and `force_to_grad` functions.
        # return expect_and_grad_from_covariance(vstate, Ô, *args, mutable=mutable)
        with prof_section(
            "expect_and_grad.covariance",
            cat="vqs.grad",
            args={"operator_type": type(Ô).__name__},
        ):
            Ō, Ō_grad = expect_and_forces(vstate, Ô, chunk_size, *args, mutable=mutable)
            Ō_grad = force_to_grad(Ō_grad, vstate.parameters)
            return Ō, Ō_grad
    else:
        # Experimental bi-covariance path for non-Hermitian operators.
        if _use_experimental_grad():
            with prof_section(
                "expect_and_grad.biadjoint",
                cat="vqs.grad",
                args={"operator_type": type(Ô).__name__},
            ) as sec:
                result = _try_biadjoint_single(vstate, Ô, chunk_size, mutable=mutable)
                sec.sync(result)
            if result is not None:
                return result
            # Adjoint unavailable, fall through to generic path.

        with prof_section(
            "expect_and_grad.nonhermitian",
            cat="vqs.grad",
            args={"operator_type": type(Ô).__name__},
        ):
            return expect_and_grad_nonhermitian(
                vstate, Ô, chunk_size, *args, mutable=mutable
            )


# Squared is a special operator...
@expect_and_grad.dispatch
def expect_and_grad_squared_op(
    vstate: Union[MCState, NQXMCState],
    Ô: Squared,
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    if use_covariance is not None:
        raise ValueError(
            "Cannot specify `use_covariance` with Squared[...] operator.\n"
            "This operator must use the same formula as non-hermitian operators to work."
        )
    if _use_experimental_grad():
        with prof_section(
            "expect_and_grad.squared.biadjoint",
            cat="vqs.grad",
            args={"operator_type": type(Ô).__name__},
        ) as sec:
            result = _try_biadjoint_single(vstate, Ô, chunk_size, mutable=mutable)
            sec.sync(result)
        if result is not None:
            return result
    return expect_and_grad_nonhermitian(vstate, Ô, chunk_size, *args, mutable=mutable)


@expect_and_grad.dispatch
def expect_and_grad_squared_op_sequence(
    vstate: Union[MCState, NQXMCState],
    Ô: Sequence[Union[Squared, PenaltyCost, ComputationalOperator]],
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    """
    Extends the squared-op code to handle either a single Squared[...] operator,
    or a list of Squared[...] operators. If it's a list, we do a multi-operator approach.

    It can also accept PenaltyCost operators which are not Squared.
    """

    # We disallow use_covariance for squared, following NetKet's logic
    if use_covariance is not None:
        raise ValueError(
            "Cannot specify `use_covariance` with Squared[...] operator.\n"
            "This operator must use the same formula as non-hermitian operators to work."
        )

    # assume it's a list of Squared[...] operators
    if not isinstance(Ô, list):
        raise InvalidOperatorsSequenceError("expect_and_grad")

    # check for empty lists
    if len(Ô) == 0:
        raise ExpectationValueError("expect_and_grad")

    if _use_experimental_grad():
        with prof_section(
            "expect_and_grad.sequence.squared.biadjoint",
            cat="vqs.grad",
            args={"n_operators": int(len(Ô))},
        ) as sec:
            result = _try_biadjoint_sequence(vstate, Ô, chunk_size, mutable=mutable)
            sec.sync(result)
        if result is not None:
            return result

    # now we have a list of squared operators: we route them to the non-hermitian logic
    # by calling `expect_and_grad_nonhermitian(...)` with the entire list
    # This will dispatch the multi-operator summation approach
    with prof_section(
        "expect_and_grad.sequence.squared.nonhermitian",
        cat="vqs.grad",
        args={"n_operators": int(len(Ô))},
    ):
        return expect_and_grad_nonhermitian(vstate, Ô, chunk_size, *args, mutable=mutable)


@dispatch.dispatch
def expect_and_grad_nonhermitian(
    vstate: Union[MCState, NQXMCState],
    Ô: Union[AbstractOperator, Squared],
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
):
    if not isinstance(Ô, Squared) and not config.netket_experimental:
        raise RuntimeError(
            """
            Computing the gradient of non hermitian operator is an
            experimental feature under development and is known not to
            return wrong values sometimes.

            If you want to debug it, set the environment variable
            NETKET_EXPERIMENTAL=1
            """
        )

    with prof_section(
        "expect_and_grad.nonhermitian.resolve_args",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ):
        σ, args = get_local_kernel_arguments(vstate, Ô)

    with prof_section(
        "expect_and_grad.nonhermitian.resolve_kernel",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ):
        local_estimator_fun = _get_local_kernel_cached(vstate, Ô)

    # remain general here for all costs
    scale_factor = Ô.factor if isinstance(Ô, PenaltyCost) else 1.0

    with prof_section(
        "expect_and_grad.nonhermitian.kernel",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ) as sec:
        Ō, Ō_grad, new_model_state = _grad_expect_nonherm_kernel(
            local_estimator_fun,
            vstate._apply_fun,
            mutable,
            vstate.sampler.machine_pow,
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


@dispatch.dispatch
def expect_and_grad_nonhermitian(
    vstate: Union[MCState, NQXMCState],
    Ô: Sequence[AbstractOperator],
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
):
    """
    Extends the NetKet implementation to handle list of non-Hermitian (or 'Squared') operators.
    """

    # ensure list type
    Ô_list = list(Ô)

    # Sanity check
    if len(Ô_list) == 0:
        raise ExpectationValueError("expect_and_grad")

    # NetKet's `_grad_expect_nonherm_kernel()` does not support updating model_state
    # we either skip it or raise an error if mutable != False. We'll raise an error like NetKet:
    if mutable is not False:
        raise NotImplementedError(
            "Updating model_state (mutable != False) is not implemented for multiple "
            "non-Hermitian operators. Please set mutable=False."
        )

    # get the vstate.samples to ensure that we use the same ones for every Ô
    σ = vstate.samples

    # get the number of MC chains
    n_chains = σ.shape[0]

    # flatten if needed
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    L_σ_sum = jnp.zeros((σ.shape[0],), dtype=jnp.result_type(float))

    # aggregate gradients from IEC and non-IEC branches
    grad_sum = None

    fused_kernels: list[Callable] = []
    fused_factored_kernels: list[Callable | None] = []
    use_factored_kernels: list[bool] = []
    fused_args: list[PyTree] = []
    fused_scales: list[float] = []
    use_fused = _use_fused_kernels()

    with prof_section(
        "expect_and_grad.nonhermitian.sequence.prepare",
        cat="vqs.grad",
        args={"n_operators": int(len(Ô_list))},
    ):
        for i, ô in enumerate(Ô_list):
            with prof_section(
                "expect_and_grad.nonhermitian.sequence.operator",
                cat="vqs.grad",
                args={"index": int(i), "operator_type": type(ô).__name__},
            ):
                # Discard σ returned by the argument builder: we enforce one shared sample batch.
                _, args_op = get_local_kernel_arguments(vstate, ô)
                local_estimator_fun = _get_local_kernel_cached(vstate, ô)

                if strict_type(ô) is InverseExpectationCost:
                    # branch out for the IEC
                    # forward locals, but grad via forces on E only so we avoid backprop through expect()
                    with prof_section(
                        "expect_and_grad.nonhermitian.sequence.iec_forward",
                        cat="vqs.grad",
                    ):
                        # forward locals for IEC (no VJP)
                        L_op = _locals_only(
                            local_estimator_fun,
                            vstate._apply_fun,
                            vstate.parameters,
                            vstate.model_state,
                            σ,
                            args_op,
                        )

                    # gradient via forces on E
                    # dev: we are essentially computing the gradient via covariance, which is fine
                    #      as long as our operator is Hermitian. A safety net has been added in the
                    #      IEC wrapper to ensure this and it is assumed that the wrapped operator is
                    #      Hermitian at this point
                    with prof_section(
                        "expect_and_grad.nonhermitian.sequence.iec_forces",
                        cat="vqs.grad",
                    ):
                        V_stats, V_forces = expect_and_forces(
                            vstate, ô.cost_operator, None, mutable=False
                        )

                    # get the mean and assemble the derivative
                    meanV = jnp.real(V_stats.mean).astype(jnp.result_type(float))
                    denom = ô.alpha + meanV + ô.eps
                    fprime = (-2.0 * ô.factor) / (denom ** 3)

                    # construct the gradient, first forces
                    Ō_forces_i = jax.tree_util.tree_map(lambda g: fprime * g, V_forces)

                    # now gradients
                    Ō_grad_i = force_to_grad(Ō_forces_i, vstate.parameters)
                else:
                    fused_kernels.append(local_estimator_fun)
                    factored_kernel = resolve_factored_local_kernel(
                        local_estimator_fun, chunked=False
                    )
                    fused_factored_kernels.append(factored_kernel)
                    use_factored_kernels.append(factored_kernel is not None)
                    fused_args.append(args_op)
                    fused_scales.append(
                        float(ô.factor) if strict_type(ô) is PenaltyCost else 1.0
                    )

                # aggregate local estimators and partial gradients from IEC branch immediately
                if strict_type(ô) is InverseExpectationCost:
                    L_σ_sum = L_σ_sum + L_op
                    grad_sum = (
                        Ō_grad_i
                        if grad_sum is None
                        else jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, Ō_grad_i)
                    )

    # Optional fused VJP for all non-IEC non-Hermitian terms.
    if fused_kernels:
        if use_fused:
            with prof_section(
                "expect_and_grad.nonhermitian.sequence.fused_kernel",
                cat="vqs.grad",
                args={"n_fused_operators": int(len(fused_kernels))},
            ) as sec:
                L_fused, _Ō_stat, Ō_grad_fused, _new_model_state = _grad_expect_nonherm_kernel_sequence_fused(
                    tuple(fused_kernels),
                    tuple(fused_factored_kernels),
                    tuple(use_factored_kernels),
                    vstate._apply_fun,
                    False,
                    vstate.sampler.machine_pow,
                    vstate.parameters,
                    vstate.model_state,
                    σ,
                    tuple(fused_args),
                    tuple(fused_scales),
                )
                sec.sync((L_fused, Ō_grad_fused))
            L_σ_sum = L_σ_sum + L_fused
            grad_sum = (
                Ō_grad_fused
                if grad_sum is None
                else jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, Ō_grad_fused)
            )
        else:
            with prof_section(
                "expect_and_grad.nonhermitian.sequence.unfused_kernel",
                cat="vqs.grad",
                args={"n_operators": int(len(fused_kernels))},
            ) as sec:
                for i, local_estimator_fun in enumerate(fused_kernels):
                    # Keep local estimators on the original non-fused path.
                    L_i = _locals_only(
                        local_estimator_fun,
                        vstate._apply_fun,
                        vstate.parameters,
                        vstate.model_state,
                        σ,
                        fused_args[i],
                    )
                    sf = jnp.asarray(fused_scales[i], dtype=L_i.dtype)
                    L_i = sf * L_i

                    _Ō_stat_i, Ō_grad_i, _new_model_state = _grad_expect_nonherm_kernel(
                        local_estimator_fun,
                        vstate._apply_fun,
                        False,
                        vstate.sampler.machine_pow,
                        vstate.parameters,
                        vstate.model_state,
                        σ,
                        fused_args[i],
                        fused_scales[i],
                    )
                    L_σ_sum = L_σ_sum + L_i
                    grad_sum = (
                        Ō_grad_i
                        if grad_sum is None
                        else jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, Ō_grad_i)
                    )
                sec.sync((L_σ_sum, grad_sum))

    # compute the final Stats object for L_σ_sum
    # we do "statistics(...)" on the sum of local values
    with prof_section(
        "expect_and_grad.nonhermitian.sequence.statistics",
        cat="vqs.grad",
        args={"n_chains": int(n_chains)},
    ):
        stats_sum = statistics(L_σ_sum.reshape((n_chains, -1)))

    return stats_sum, grad_sum


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4,))
def _grad_expect_nonherm_kernel_sequence_fused(
    local_value_kernels: tuple[Callable, ...],
    local_value_factored_kernels: tuple[Callable | None, ...],
    use_factored_kernels: tuple[bool, ...],
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    machine_pow: float,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: tuple[PyTree, ...],
    local_scales: tuple[float, ...],
) -> tuple[PyTree, Stats, PyTree, PyTree]:
    """
    Fused non-Hermitian sequence gradient for operators with standard expect+VJP path.
    """
    n_chains = σ.shape[0]

    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    if model_state is None:
        model_state = {}

    is_mutable = mutable is not False
    logpsi = lambda w, σi: model_apply_fun(
        {"params": w, **model_state}, σi, mutable=mutable
    )
    log_pdf = (
        lambda w, σi: machine_pow * model_apply_fun({"params": w, **model_state}, σi).real
    )
    has_factored = any(use_factored_kernels)

    def local_total(pars, σi):
        if has_factored:
            log_sigma = logpsi(pars, σi)
        else:
            log_sigma = None

        total = jnp.zeros((σi.shape[0],), dtype=jnp.result_type(float))
        for i, local_value_kernel in enumerate(local_value_kernels):
            if use_factored_kernels[i]:
                loc_i = local_value_factored_kernels[i](
                    log_sigma,
                    logpsi,
                    pars,
                    σi,
                    local_value_args[i],
                )
            else:
                loc_i = local_value_kernel(logpsi, pars, σi, local_value_args[i])
            scale_i = jnp.asarray(local_scales[i], dtype=loc_i.dtype)
            total = total + scale_i * loc_i
        return total

    def expect_closure_pars(pars):
        return nkjax.expect(
            log_pdf,
            local_total,
            pars,
            σ,
            n_chains=n_chains,
        )

    L_total = local_total(parameters, σ)

    Ō, Ō_pb, Ō_stats = nkjax.vjp(
        expect_closure_pars, parameters, has_aux=True, conjugate=True
    )
    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))[0]

    if is_mutable:
        raise NotImplementedError(
            "gradient of non-hermitian operators over mutable models "
            "is not yet implemented."
        )
    new_model_state = None

    return L_total, Ō_stats, Ō_pars_grad, new_model_state


@partial(jax.jit, static_argnums=(0, 1, 2,))
def _grad_expect_nonherm_kernel_sequence(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    machine_pow: float,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> tuple[PyTree, PyTree, Stats]:
    """
    Extends the NetKet logic to return the local estimator instead of the Stats
    """

    n_chains = σ.shape[0]

    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    is_mutable = mutable is not False
    logpsi = lambda w, σ: model_apply_fun(
        {"params": w, **model_state}, σ, mutable=mutable
    )
    log_pdf = (
        lambda w, σ: machine_pow * model_apply_fun({"params": w, **model_state}, σ).real
    )

    def expect_closure_pars(pars):
        return nkjax.expect(
            log_pdf,
            partial(local_value_kernel, logpsi),
            pars,
            σ,
            local_value_args,
            n_chains=n_chains,
        )

    def expect_closure_pars_lop(pars):
        return _nkjax_expect_kernel(
            partial(local_value_kernel, logpsi),
            pars,
            σ,
            local_value_args,
        )

    L_op = expect_closure_pars_lop(parameters)

    Ō, Ō_pb, Ō_stats = nkjax.vjp(
        expect_closure_pars, parameters, has_aux=True, conjugate=True
    )
    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))[0]

    if is_mutable:
        raise NotImplementedError(
            "gradient of non-hermitian operators over mutable models "
            "is not yet implemented."
        )
    new_model_state = None

    return (
        L_op,
        Ō_stats,
        Ō_pars_grad,
        new_model_state,
    )


@partial(jax.jit, static_argnums=(0, 1, 2,))
def _grad_expect_nonherm_kernel(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    machine_pow: float,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
    scale_factor: float = 1.0,
) -> tuple[PyTree, PyTree, Stats]:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    is_mutable = mutable is not False
    logpsi = lambda w, σ: model_apply_fun(
        {"params": w, **model_state}, σ, mutable=mutable
    )
    log_pdf = (
        lambda w, σ: machine_pow * model_apply_fun({"params": w, **model_state}, σ).real
    )

    def expect_closure_pars(pars):
        out = nkjax.expect(
            log_pdf,
            partial(local_value_kernel, logpsi),
            pars,
            σ,
            local_value_args,
            n_chains=n_chains,
        )

        sf = jnp.asarray(scale_factor, dtype=jnp.result_type(float))
        return (sf * out[0], out[1])  # (value, Stats)

    Ō, Ō_pb, Ō_stats = nkjax.vjp(
        expect_closure_pars, parameters, has_aux=True, conjugate=True
    )

    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))[0]

    if is_mutable:
        raise NotImplementedError(
            "gradient of non-hermitian operators over mutable models "
            "is not yet implemented."
        )
    new_model_state = None

    return (
        Ō_stats,
        Ō_pars_grad,
        new_model_state,
    )


def _nkjax_expect_kernel(expected_fun, pars, σ, *expected_fun_args):
    return expected_fun(pars, σ, *expected_fun_args)


@expect_and_grad.dispatch
def expect_and_grad_iec_single(
    vstate: Union[MCState, NQXMCState],
    Ô: InverseExpectationCost,
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    """
    IEC-specific, computes gradient via forces on E only:
        d/dθ <IEC> = f'(<E>) * d/dθ <E>
    while keeping the forward stats of IEC via the (already implemented) local estimator

    Note: the assumption here is that the wrapped operator E is Hermitian. This will produce
    incorrect results if it is not, as we use the covariance method to compute the gradient
    """

    # forward stats for IEC
    with prof_section("expect_and_grad.iec.resolve", cat="vqs.grad"):
        σ_iec, packed_iec = get_local_kernel_arguments(vstate, Ô)
        local_iec = get_local_kernel(vstate, Ô)

    # get the number of chains
    n_chains = vstate.samples.shape[0]

    # collapse the samples if needed
    if σ_iec.ndim >= 3:
        σ_iec = jax.lax.collapse(σ_iec, 0, 2)

    def logpsi(w, σ_):
        return vstate._apply_fun({"params": w, **vstate.model_state}, σ_)

    # compute the local estimator and stats
    with prof_section("expect_and_grad.iec.forward_stats", cat="vqs.grad"):
        L_σ = local_iec(logpsi, vstate.parameters, σ_iec, packed_iec)
        Ō_iec = statistics(L_σ.reshape((n_chains, -1)))

    # compute the forces for V
    with prof_section("expect_and_grad.iec.forces", cat="vqs.grad"):
        V_stats, V_forces = expect_and_forces(
            vstate, Ô.cost_operator, chunk_size, mutable=mutable
        )

    # chain rule at mean(V)
    meanV = jnp.real(V_stats.mean).astype(jnp.result_type(float))
    denom = Ô.alpha + meanV + Ô.eps
    fprime = (-2.0 * Ô.factor) / (denom ** 3)

    # still in forces
    IEC_forces = jax.tree_util.tree_map(lambda g: fprime * g, V_forces)

    # now convert forces to grad
    IEC_grad = force_to_grad(IEC_forces, vstate.parameters)

    return Ō_iec, IEC_grad
