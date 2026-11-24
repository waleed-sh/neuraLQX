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


"""Framework adapter discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

from .base import AbstractModelFramework
from .callable import CallableFramework
from .flax import FlaxLinenFramework
from .flax import FlaxNNXFramework


def as_framework(
    model: Any | None = None,
    *,
    apply_fn: Callable[..., Any] | None = None,
    init_fn: Callable[..., Any] | None = None,
    expects_variables: bool = False,
) -> tuple[AbstractModelFramework, Any | None]:
    """Return ``(framework, maybe_variables)`` for supported model inputs.

    Initialized NNX models return their split variables. Linen modules and
    JAX callables return ``None`` until their initializer is called. When
    ``network_sharding_enabled`` is set, a tensor adapter selects and validates
    parameter layouts while preserving the model's ordinary global function.
    ``TensorParallel(model, parameter_rules)`` supplies explicit overrides for
    custom networks and remains usable when tensor parallelism is disabled.
    """
    from neuralqx.jax.tensor_parallel import network_sharding_enabled
    from neuralqx.nn.parallel import TensorParallel

    rules = ()
    if isinstance(model, TensorParallel):
        rules = tuple(model.parameter_rules.items())
        init_fn = model.init_fn if model.init_fn is not None else init_fn
        if model.expects_variables is not None:
            expects_variables = model.expects_variables
        model = model.model
    if network_sharding_enabled():
        from .tensor_parallel import TensorParallelFramework
        from .tensor_parallel import wrap_tensor_parallel

        if isinstance(model, TensorParallelFramework):
            return model, None
        base, variables = _as_framework(
            model,
            apply_fn=apply_fn,
            init_fn=init_fn,
            expects_variables=expects_variables,
        )
        return wrap_tensor_parallel(base, variables, model, rules)
    return _as_framework(
        model, apply_fn=apply_fn, init_fn=init_fn, expects_variables=expects_variables
    )


def _as_framework(model=None, *, apply_fn=None, init_fn=None, expects_variables=False):
    if isinstance(model, AbstractModelFramework):
        return model, None

    if model is not None and apply_fn is not None:
        raise NeuralqxValueError("Pass either model or apply_fn, not both.")

    from neuralqx.nn.projectors import ProjectedCallable

    candidate = model if model is not None else apply_fn
    if isinstance(candidate, ProjectedCallable):
        from .projected import ProjectedFramework

        base, variables = _as_framework(
            apply_fn=candidate.model,
            init_fn=init_fn,
            expects_variables=expects_variables,
        )
        return ProjectedFramework(base, candidate.projector), variables

    if model is not None:
        if _is_linen_module(model):
            return FlaxLinenFramework(model), None
        if _is_nnx_module(model):
            return FlaxNNXFramework.from_model(model)
        if callable(model):
            return (
                CallableFramework(
                    model, init_fn=init_fn, expects_variables=expects_variables
                ),
                None,
            )
        raise NeuralqxTypeError(f"Unsupported model type {type(model).__name__}.")

    if apply_fn is None:
        raise NeuralqxValueError("Pass a Flax model, a callable model, or apply_fn.")
    return (
        CallableFramework(
            apply_fn,
            init_fn=init_fn,
            expects_variables=expects_variables,
        ),
        None,
    )


def _is_linen_module(model: Any) -> bool:
    try:
        from flax import linen as nn
    except Exception:
        return False
    return isinstance(model, nn.Module)


def _is_nnx_module(model: Any) -> bool:
    try:
        from flax import nnx
    except Exception:
        return False
    return isinstance(model, nnx.Module)


__all__ = ["as_framework"]
