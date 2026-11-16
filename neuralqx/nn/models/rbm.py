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


"""Restricted Boltzmann wavefunctions with reusable hidden fields."""

import jax.numpy as jnp
from flax import nnx

from ..activations import logcosh
from ..layers import Linear
from ..utils import ConnectionPolicy
from ..utils import as_rngs
from ..utils.initialization import compute_dtype
from ..utils.initialization import input_array
from ..utils.initialization import log_amplitude
from ..utils.initialization import positive_int
from .base import LogWavefunction


class RBM(LogWavefunction):
    r"""RBM wavefunction with separate log-modulus and phase channels by default.

    Each output channel is
    :math:`a\cdot x+c+\sum_j\log\cosh(b_j+W_j\cdot x)/\sqrt{M}`,
    where :math:`M` is ``hidden_features``. With ``complex_output=True``, the
    first channel is the log modulus and the second is the phase when parameters
    are real. Complex computation also permits complex channel values. Omitting
    the constant ``log(2)`` from each hidden unit changes only normalization.

    Source hidden fields and visible terms are reusable for sparse connections.
    Their updates are exact and differentiated together with the source fields.
    ``dtype``, ``param_dtype`` and ``precision`` control the affine kernels.
    """

    in_features: int
    """Number of visible coordinates."""

    hidden_features: int
    """Number of hidden units per output channel."""

    hidden: Linear
    """Affine fields for all hidden units."""

    visible: Linear
    """Visible linear terms and normalization offsets."""

    complex_output: bool
    """Whether an independent phase channel is included."""

    input_scale: float
    """Fixed divisor applied to visible coordinates."""

    dtype: object
    """Computation dtype."""

    def __init__(
        self,
        in_features,
        hidden_features=128,
        *,
        complex_output=True,
        input_scale=1.0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        connection_policy=None,
        rngs,
    ):
        """Initialize RBM fields and visible terms from a shared RNG stream."""
        self.in_features = positive_int(in_features, "in_features")
        self.hidden_features = positive_int(hidden_features, "hidden_features")
        self.complex_output = bool(complex_output)
        self.input_scale = float(input_scale)
        if not 0 < self.input_scale < float("inf"):
            raise ValueError("input_scale must be finite and positive.")
        dtype = compute_dtype(dtype, param_dtype)
        self.dtype = dtype
        self.connection_policy = connection_policy or ConnectionPolicy()
        rngs = as_rngs(rngs)
        channels = 2 if complex_output else 1
        options = dict(
            rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        self.hidden = Linear(in_features, channels * self.hidden_features, **options)
        self.visible = Linear(
            in_features, channels, kernel_init=nnx.initializers.normal(0.01), **options
        )

    def _reduce(self, fields, visible):
        channels = 2 if self.complex_output else 1
        hidden = logcosh(fields).reshape(
            (*fields.shape[:-1], self.hidden_features, channels)
        )
        return log_amplitude(
            hidden.sum(axis=-2) / self.hidden_features**0.5 + visible,
            self.complex_output,
        )

    def __call__(self, states):
        """Return RBM log amplitudes with arbitrary leading batch dimensions."""
        return self._reduce(*self.prepare_connections(states))

    def prepare_connections(self, sources):
        """Cache the current hidden fields and visible linear terms."""
        states = input_array(sources, self.in_features, self.dtype) / self.input_scale
        return self.hidden(states), self.visible(states)

    def apply_connections(self, cache, owners, sites, differences):
        """Update hidden and visible terms without a full input contraction."""
        fields, visible = cache
        delta = differences / self.input_scale
        return self._reduce(
            fields[owners] + self.hidden.sparse_delta(sites, delta),
            visible[owners] + self.visible.sparse_delta(sites, delta),
        )

    def use_connection_reuse(self, source_shape, target_shape):
        """Use automatic field reuse for at least 512 inputs and 512 total hidden fields.

        Smaller dense maps usually cost less than discovering and gathering
        sparse changes. ``ConnectionPolicy(mode="on")`` permits explicit
        profiling on architectures where that tradeoff differs.
        """
        if self.connection_policy.mode == "auto" and (
            self.in_features < 512 or self.hidden.out_features < 512
        ):
            return False
        return super().use_connection_reuse(source_shape, target_shape)
