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


"""Differentiation-mode resolution for variational models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

import jax.numpy as jnp

import jax
from neuralqx.jax.tree import tree_any_complex


class HolomorphicUndeclaredWarning(UserWarning):
    """Warning emitted when complex holomorphicity must be declared explicitly."""


def resolve_differentiation_mode(
    apply: Callable[[Any], Any],
    parameters: Any,
    *,
    mode: str | None = None,
    holomorphic: bool | None = None,
    warn: bool = True,
) -> str:
    """Resolve ``real``/``complex``/``holomorphic`` differentiation mode.

    ``mode="real"`` differentiates the real output channel in real parameter
    coordinates. ``mode="complex"`` differentiates both real and imaginary
    output channels in real parameter coordinates. ``mode="holomorphic"`` uses
    complex holomorphic calculus and therefore requires all parameter leaves and
    the model output to be complex.
    """
    if mode is not None and holomorphic is not None:
        raise ValueError("Pass either mode or holomorphic, not both.")
    if mode is not None:
        mode = str(mode)
        if mode not in {"real", "complex", "holomorphic"}:
            raise ValueError(
                "Differentiation mode must be 'real', 'complex', or "
                f"'holomorphic', got {mode!r}."
            )
        if mode == "holomorphic":
            _validate_holomorphic(parameters, apply)
        return mode

    if holomorphic is True:
        _validate_holomorphic(parameters, apply)
        return "holomorphic"

    output_shape = jax.eval_shape(apply, parameters)
    output_is_complex = bool(jnp.iscomplexobj(output_shape))
    params_are_complex = tree_any_complex(parameters)
    if output_is_complex:
        if params_are_complex and holomorphic is None and warn:
            warnings.warn(
                "Complex-output model with complex parameters was not declared "
                "holomorphic; using non-holomorphic real-coordinate mode. Pass "
                "holomorphic=True to use holomorphic calculus, or mode='complex' "
                "to silence this warning.",
                HolomorphicUndeclaredWarning,
                stacklevel=3,
            )
        return "complex"
    return "real"


def _validate_holomorphic(parameters: Any, apply: Callable[[Any], Any]) -> None:
    if not tree_any_complex(parameters):
        raise ValueError(
            "holomorphic=True/mode='holomorphic' requires complex parameters."
        )
    output_shape = jax.eval_shape(apply, parameters)
    if not bool(jnp.iscomplexobj(output_shape)):
        raise ValueError(
            "holomorphic=True/mode='holomorphic' requires complex model output."
        )


__all__ = ["HolomorphicUndeclaredWarning", "resolve_differentiation_mode"]
