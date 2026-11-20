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


"""Shape checks and coherent amplitude reductions for projectors."""

import jax.numpy as jnp

import jax
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

from ..utils.connections import map_connections


def projector_state(projector):
    """Normalize numeric data leaves for eager and abstract initialization alike."""
    return jax.tree.map(
        lambda x: x if isinstance(x, jax.ShapeDtypeStruct) else jnp.asarray(x),
        projector,
    )


def log_values(values, states):
    """Require one scalar log amplitude per configuration."""
    values = jnp.asarray(values)
    if values.shape != states.shape[:-1]:
        raise NeuralqxValueError(
            f"Expected log amplitudes of shape {states.shape[:-1]}, got {values.shape}."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise NeuralqxTypeError(
            "A wavefunction must return floating or complex log amplitudes."
        )
    return values


def mask_log_values(values, mask):
    """Set excluded amplitudes to exact zero without differentiating log(0)."""
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_ or mask.shape != values.shape:
        raise NeuralqxValueError(
            "A diagonal projector must return a boolean mask with one entry per state."
        )
    return jnp.where(mask, values, jnp.asarray(-jnp.inf, dtype=values.dtype))


def coherent_logsum(logs, coefficients):
    """Sum signed or complex amplitudes using a detached logarithmic scale.

    Zero coefficients and zero input amplitudes contribute exactly zero.
    Exact cancellations return ``-inf+0j``. The derivative of a log amplitude
    at a node is undefined. We assign zero there to keep structurally zero
    rows inert in automatic differentiation. Away from nodes, network
    derivatives are those of the full coherent sum. Coefficients are fixed
    projector data, not variational parameters.
    """
    dtype = jnp.result_type(logs, coefficients, jnp.complex64)
    logs = logs.astype(dtype)
    coefficients = coefficients.astype(dtype)
    magnitude = jnp.abs(coefficients)
    active = (magnitude != 0) & ~jnp.isneginf(logs.real)
    safe_magnitude = jnp.where(active, magnitude, 1)
    shifted = jnp.where(active, logs.real + jnp.log(safe_magnitude), -jnp.inf)
    scale = jnp.max(shifted, axis=-1, keepdims=True)
    scale = jax.lax.stop_gradient(jnp.where(jnp.isfinite(scale), scale, 0))
    phase = jnp.where(active, logs.imag, 0)
    terms = (
        jnp.where(active, coefficients, 0)
        / safe_magnitude
        * jnp.exp(shifted - scale + 1j * phase)
    )
    total = jnp.sum(terms, axis=-1)
    nonzero = total != 0
    value = jnp.log(jnp.where(nonzero, total, 1)) + scale[..., 0]
    return jnp.where(nonzero, value, jnp.asarray(-jnp.inf, dtype=dtype))


def connected_evaluator(model):
    """Use the model's exact connection capability or a bounded full evaluation."""
    connected = getattr(model, "connected_log_value", None)
    if callable(connected):
        return connected

    def evaluate(sources, targets, owners, *, chunk_size=None):
        del sources
        return map_connections(lambda x, _: model(x), targets, owners, chunk_size)

    return evaluate
