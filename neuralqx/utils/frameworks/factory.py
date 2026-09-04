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
    """Return ``(framework, maybe_variables)`` for supported model inputs."""
    if isinstance(model, AbstractModelFramework):
        return model, None

    if model is not None and apply_fn is not None:
        raise ValueError("Pass either model or apply_fn, not both.")

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
        raise TypeError(f"Unsupported model type {type(model).__name__}.")

    if apply_fn is None:
        raise ValueError("Pass a Flax model, a callable model, or apply_fn.")
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
