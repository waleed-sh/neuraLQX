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


"""Periodic convolutional wavefunctions and local receptive-field reuse."""

import math
from itertools import pairwise

import jax.numpy as jnp
from flax import nnx

from ..activations import activation_spec
from ..layers import Conv
from ..layers import Linear
from ..utils import ConnectionPolicy
from ..utils import as_rngs
from ..utils.convolution import conv1d
from ..utils.initialization import compute_dtype
from ..utils.initialization import input_array
from ..utils.initialization import log_amplitude
from ..utils.initialization import positive_int
from .base import LogWavefunction


class CNN(LogWavefunction):
    """Periodic convolutional log wavefunction on a fixed lattice.

    Flattened configurations are reshaped to ``lattice_shape`` with one input
    channel per lattice site. Shared convolutions and mean pooling impose
    translation invariance. A nonlinear pooled readout produces log modulus
    and, optionally, phase. This symmetry is part of the model definition.

    On sufficiently long one-dimensional lattices, a single changed site only
    changes a finite receptive field. The connected evaluator recomputes that
    field and updates the pooled features exactly. Multiple changes, short
    lattices, higher dimensions and non-elementwise activations use the full
    convolution. Reuse includes all source-feature derivatives.
    """

    lattice_shape: tuple
    """Spatial lattice dimensions in the flattened input's row-major order."""

    convolutions: nnx.List
    """Periodic convolution stages with odd kernels and unit stride."""

    readout: Linear
    """Nonlinear pooled-feature projection."""

    head: Linear
    """Scalar log-amplitude and optional phase projection."""

    activation: object
    """Resolved activation shared by spatial layers and the readout."""

    radius: int
    """One-dimensional receptive-field radius after the final convolution."""

    in_features: int
    """Number of sites in the flattened lattice."""

    complex_output: bool
    """Whether a separate phase head is used."""

    dtype: object
    """Computation dtype for inputs and convolutions."""

    def __init__(
        self,
        lattice_shape,
        features=(16, 16),
        *,
        kernel_size=3,
        activation="silu",
        complex_output=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        connection_policy=None,
        rngs,
    ):
        """Build periodic convolutions, pooling and a shared nonlinear readout.

        ``lattice_shape`` may be an integer for a chain or a sequence for a
        higher-dimensional lattice. ``features`` gives each convolution's
        output channels. ``kernel_size`` is the same odd width on every axis.
        """
        shape = (
            (lattice_shape,) if isinstance(lattice_shape, int) else tuple(lattice_shape)
        )
        self.lattice_shape = tuple(positive_int(n, "lattice_shape") for n in shape)
        if not shape:
            raise ValueError("lattice_shape must have at least one dimension.")
        widths = tuple(positive_int(w, "features") for w in features)
        if not widths:
            raise ValueError("CNN requires at least one convolution.")
        kernel_size = positive_int(kernel_size, "kernel_size")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.radius = len(widths) * (kernel_size // 2)
        self.in_features = math.prod(self.lattice_shape)
        dtype = compute_dtype(dtype, param_dtype)
        self.dtype = dtype
        self.complex_output = bool(complex_output)
        spec = activation_spec(activation, dtype=dtype)
        self.activation = spec.function
        policy = connection_policy or ConnectionPolicy(max_changes=1, min_features=64)
        can_reuse = (
            len(shape) == 1 and self.in_features > 4 * self.radius and spec.elementwise
        )
        if can_reuse:
            self.connection_policy = ConnectionPolicy(
                policy.mode, 1, policy.min_features
            )
        else:
            self.connection_policy = ConnectionPolicy(mode="off")
        rngs = as_rngs(rngs)
        options = dict(
            rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        channels = (1, *widths)
        self.convolutions = nnx.List(
            [
                Conv(a, b, (kernel_size,) * len(shape), padding="CIRCULAR", **options)
                for a, b in pairwise(channels)
            ]
        )
        self.readout = Linear(widths[-1], widths[-1], **options)
        self.head = Linear(
            widths[-1],
            2 if complex_output else 1,
            kernel_init=nnx.initializers.normal(0.01),
            **options,
        )

    def _readout(self, pooled):
        return log_amplitude(
            self.head(self.activation(self.readout(pooled))), self.complex_output
        )

    def __call__(self, states):
        """Evaluate flattened lattice configurations with arbitrary batch axes."""
        x = input_array(states, self.in_features, self.dtype)
        leading = x.shape[:-1]
        h = x.reshape((-1, *self.lattice_shape, 1))
        for layer in self.convolutions:
            h = self.activation(layer(h))
        pooled = h.mean(axis=tuple(range(1, len(self.lattice_shape) + 1)))
        return self._readout(pooled).reshape(leading)

    def prepare_connections(self, sources):
        """Cache source inputs, final spatial features and their mean for a chain."""
        x = input_array(sources, self.in_features, self.dtype)
        h = x[..., None]
        for layer in self.convolutions:
            h = self.activation(layer(h))
        return x, h, h.mean(axis=1)

    def apply_connections(self, cache, owners, sites, differences):
        """Recompute the affected one-dimensional receptive field exactly."""
        inputs, features, pooled = cache
        r = self.radius
        center = sites[:, 0]
        indices = (center[:, None] + jnp.arange(-2 * r, 2 * r + 1)) % self.in_features
        patch = inputs[owners[:, None], indices]
        patch = patch.at[:, 2 * r].add(jnp.asarray(differences[:, 0], self.dtype))[
            ..., None
        ]
        for layer in self.convolutions:
            patch = conv1d(
                patch,
                jnp.asarray(layer.kernel[...], self.dtype),
                jnp.asarray(layer.bias[...], self.dtype),
                padding="VALID",
                precision=layer.precision,
            )
            patch = self.activation(patch)
        affected = (center[:, None] + jnp.arange(-r, r + 1)) % self.in_features
        old = features[owners[:, None], affected]
        new_pool = pooled[owners] + (patch - old).sum(axis=1) / self.in_features
        return self._readout(new_pool)
