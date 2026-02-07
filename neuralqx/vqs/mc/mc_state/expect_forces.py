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

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
)

from netket.vqs.mc.mc_state.state import MCState

from ...mc import get_local_kernel, get_local_kernel_arguments
from ....operators import PenaltyCost
from ....utils.errors import ExpectationValueError
from ....vqs.mc.mc_state.state import MCState as NQXMCState


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

    # store the number of MC chains
    n_chains = σ.shape[0]

    if σ.ndim >= 3:
        # flatten the chain dimension if needed
        σ = jax.lax.collapse(σ, 0, 2)

    # the get_local_kernel which returns the local estimator we want to store for every Ô will
    # return a JAX array of shape (σ.shape[0], ), so we create an empty one with that shape
    # this array will hold the sum of all local estimators L_σ returned from every operator
    # dev: this was weak, essentially wrong... we were creating just a scalar zero, and later
    #      JAX changed it into an array after the first sum with the first L_σ, now it should be
    #      more "stable", aka less error prone
    # L_σ_sum = jnp.zeros_like(σ.shape[0])
    L_σ_sum = jnp.zeros((σ.shape[0],), dtype = jnp.result_type(float))

    # we will accumulate the total gradient here
    grad_sum = None

    # to keep track of the updated model_state only once at the end (if mutable)
    # for now, we do not change `vstate.model_state` at each operator
    final_model_state = vstate.model_state

    # a boolean to track if we are at the last operator in the list, to which we then update
    # the model_state
    idx_last_op = len(Ô_list) - 1

    # now we want to call the dedicated JAX jitted function to compute L_σ per operator and
    # aggregate the values
    for i, ô in enumerate(Ô_list):
        # discared the σ returned from the get_local_kernel_arguments as we want to avoid any
        # potential mismatch in using different σ for different operators in case it changes
        # somewhere during execution
        _, args_op = get_local_kernel_arguments(vstate, ô)
        local_estimator_fun = get_local_kernel(vstate, ô)

        # single-operator estimator and gradient
        # here, we call the dedicated modified forces_expect_hermitian() but pass `mutable=False`
        # so that we do not do partial updates.
        # we might want to do something else if we truly need partial model_state updates
        #
        # Notes
        # - For model_state: we keep the current model_state for read-only and only update the
        #                    actual vstate's model state at the last ô, this is changeable
        # - For mutable: we avoid partial updates after every operator unless it is the last one
        #                in the list, this is changeable
        L_op, grad_i, _new_model_state = forces_expect_hermitian_sequence(
            local_estimator_fun,
            vstate._apply_fun,
            mutable=True if i == idx_last_op else False,
            parameters=vstate.parameters,
            model_state=vstate.model_state if i == idx_last_op else final_model_state,
            σ=σ,
            local_value_args=args_op,
        )

        # check if we are doing a penalty operator of Shannon type
        if isinstance(ô, PenaltyCost):
            L_op = ô.factor * L_op

        # aggregate local estimators
        L_σ_sum = L_σ_sum + L_op

        # aggregate partial gradients
        if grad_sum is None:
            # check if we are doing a penalty operator of Shannon type
            if isinstance(ô, PenaltyCost):
                grad_i = jax.tree_util.tree_map(
                    lambda v: ô.factor * v, grad_i
                )

            grad_sum = grad_i
        else:
            # check if we are doing a penalty operator of Shannon type
            if isinstance(ô, PenaltyCost):
                grad_i = jax.tree_util.tree_map(
                    lambda v: ô.factor * v, grad_i
                )

            grad_sum = jax.tree_util.tree_map(
                lambda a, b: a + b, grad_sum, grad_i
            )

        # update the model_state if we are at the last operator
        if mutable is not False:
            if idx_last_op == i:
                vstate.model_state = _new_model_state
            else:
                final_model_state = _new_model_state

    # compute the final Stats object for L_σ_sum
    # we do "statistics(...)" on the sum of local values
    stats_sum = statistics(L_σ_sum.reshape((n_chains, -1)))

    # return the final Stats for sum(Ô_list) and the combined gradient
    return stats_sum, grad_sum


@dispatch
def expect_and_forces(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    Ô: AbstractOperator,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
) -> tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô)

    scale_factor = Ô.factor if isinstance(Ô, PenaltyCost) else 1.0

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

    n_samples = σ.shape[0] * mpi.n_nodes

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
        jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad),
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

    n_samples = σ.shape[0] * mpi.n_nodes

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
        jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad),
        new_model_state,
    )
