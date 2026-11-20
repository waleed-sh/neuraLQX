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


"""Small construction helpers shared by neural network components."""

import operator

import jax.numpy as jnp
from flax import nnx

from neuralqx.utils.errors import NeuralqxValueError


def as_rngs(rngs):
    """Accept an NNX RNG stream, integer seed, or JAX PRNG key.

    Existing streams are shared, so constructing successive layers consumes
    successive keys. An integer or key starts a new reproducible stream.
    """
    return rngs if isinstance(rngs, nnx.Rngs) else nnx.Rngs(rngs)


def positive_int(value, name):
    """Validate a positive static dimension without silently rounding it."""
    if isinstance(value, bool):
        raise NeuralqxValueError(f"{name} must be a positive integer.")
    try:
        value = operator.index(value)
    except TypeError:
        raise NeuralqxValueError(f"{name} must be a positive integer.") from None
    if value < 1:
        raise NeuralqxValueError(f"{name} must be a positive integer.")
    return value


def log_amplitude(head, complex_output):
    """Combine a two-channel amplitude and phase head into one log amplitude."""
    return head[..., 0] + 1j * head[..., 1] if complex_output else head[..., 0]


def input_array(states, features, dtype):
    """Check the final feature axis and cast inputs to the model's compute dtype."""
    states = jnp.asarray(states, dtype=dtype)
    if states.ndim == 0 or states.shape[-1] != features:
        raise NeuralqxValueError(
            f"Expected a final input axis of length {features}, got {states.shape}."
        )
    return states


def compute_dtype(dtype, param_dtype):
    """Resolve a model dtype without silently dropping complex parameters.

    ``None`` selects the parameter dtype. Both dtypes must be floating point
    or complex. Computing with real arrays while storing complex parameters
    would discard their imaginary parts, so that combination is rejected.
    """
    parameter = jnp.dtype(param_dtype)
    resolved = parameter if dtype is None else jnp.dtype(dtype)
    if not jnp.issubdtype(parameter, jnp.inexact) or not jnp.issubdtype(
        resolved, jnp.inexact
    ):
        raise NeuralqxValueError(
            "Model computation and parameter dtypes must be floating point or complex."
        )
    if jnp.issubdtype(parameter, jnp.complexfloating) and not jnp.issubdtype(
        resolved, jnp.complexfloating
    ):
        raise NeuralqxValueError(
            "Complex parameters require a complex computation dtype."
        )
    return resolved
