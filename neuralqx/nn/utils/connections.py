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


"""Exact connected-state evaluation with bounded temporary activations."""

from dataclasses import dataclass

import jax.numpy as jnp

import jax
from neuralqx.utils.errors import NeuralqxValueError


@dataclass(frozen=True)
class ConnectionPolicy:
    """Static limits for reusing a model's source activations.

    ``auto`` uses the model's conservative size threshold. ``on`` requests
    reuse even for small inputs, which is useful for measurement. Neither
    setting permits truncating changes. A chunk that exceeds ``max_changes``
    is evaluated by the ordinary model. ``off`` always uses ordinary evaluation.
    """

    mode: str = "auto"
    """One of ``auto``, ``on`` or ``off``."""

    max_changes: int = 4
    """Maximum changed coordinates handled by the sparse update."""

    min_features: int = 256
    """Minimum input size for automatic reuse."""

    def __post_init__(self):
        """Reject invalid static limits before a model is traced."""
        from .initialization import positive_int

        if self.mode not in {"auto", "on", "off"}:
            raise NeuralqxValueError("Connection mode must be 'auto', 'on' or 'off'.")
        positive_int(self.max_changes, "max_changes")
        positive_int(self.min_features, "min_features")

    def enabled(self, features, sources, targets):
        """Choose reuse from static dimensions, without transferring device data."""
        return (
            self.mode != "off"
            and targets > sources
            and (self.mode == "on" or features >= self.min_features)
        )


def sparse_changes(sources, targets, owners, max_changes, *, dtype=None):
    """Find a bounded set of changed coordinates entirely in JAX.

    Returns coordinate indices, differences, and a scalar overflow flag.
    The caller must use full evaluation when overflow is true. Zero changes
    have exactly zero differences, including padded entries in the result.
    Pass the model's computation ``dtype`` to subtract labels at the same
    precision as its input map, including for large integer coordinates.
    """
    base = sources[owners]
    dirty = targets != base
    count = jnp.sum(dirty, axis=-1)
    width = min(max_changes, targets.shape[-1])
    indices = jnp.arange(targets.shape[-1])
    sentinel = targets.shape[-1]
    remaining = jnp.where(dirty, indices, sentinel)
    selected = []
    # A handful of reductions avoids sorting every configuration coordinate.
    # In particular, CPU top_k can lower to a complete stable sort here.
    for _ in range(width):
        first = jnp.min(remaining, axis=-1)
        selected.append(first)
        remaining = jnp.where(indices <= first[:, None], sentinel, remaining)
    selected = jnp.stack(selected, axis=-1)
    sites = jnp.minimum(selected, sentinel - 1)
    # Discrete label types can be narrow integers or booleans. Subtract in an
    # inexact dtype so a label change cannot wrap before reaching the model.
    dtype = jnp.result_type(targets, base, jnp.asarray(0.0)) if dtype is None else dtype
    delta = jnp.take_along_axis(
        targets.astype(dtype) - base.astype(dtype), sites, axis=-1
    )
    delta = jnp.where(selected < sentinel, delta, 0)
    return sites, delta, jnp.any(count > width)


def validate_connections(sources, targets, owners, chunk_size):
    """Validate static connection shapes and return a normalized chunk size.

    Source index bounds remain the caller's responsibility. Checking their
    values here would require synchronization or a checkified JAX caller.
    """
    from .initialization import positive_int

    if sources.ndim != 2 or targets.ndim != 2 or sources.shape[-1] != targets.shape[-1]:
        raise NeuralqxValueError(
            "Sources and targets must be matrices with the same feature dimension."
        )
    if sources.shape[-1] == 0 or (len(sources) == 0 and len(targets) != 0):
        raise NeuralqxValueError(
            "Connections require a nonempty feature axis and valid source rows."
        )
    if owners.shape != (targets.shape[0],) or not jnp.issubdtype(
        owners.dtype, jnp.integer
    ):
        raise NeuralqxValueError(
            "owners must contain one integer source index per target."
        )
    return None if chunk_size is None else positive_int(chunk_size, "chunk_size")


def map_connections(function, targets, owners, chunk_size):
    """Map a target evaluator over bounded chunks with rematerialized activations.

    Source caches captured by ``function`` remain differentiable. Only each
    target chunk is rematerialized during the backward pass. Padding repeats
    a valid target and its owner, and is removed from the returned vector.
    """
    from .initialization import positive_int

    size = None if chunk_size is None else positive_int(chunk_size, "chunk_size")
    count = targets.shape[0]
    if size is None or count <= size:
        return function(targets, owners)
    padding = (-count) % size
    targets = jnp.pad(targets, ((0, padding), (0, 0)), mode="edge")
    owners = jnp.pad(owners, ((0, padding),), mode="edge")
    values = jax.lax.map(
        jax.checkpoint(lambda pair: function(*pair)),
        (targets.reshape((-1, size, targets.shape[-1])), owners.reshape((-1, size))),
    )
    return values.reshape(-1)[:count]
