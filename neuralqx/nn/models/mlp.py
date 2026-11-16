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


"""Dense logarithmic wavefunctions with exact first-layer reuse."""

from itertools import pairwise

import jax.numpy as jnp
from flax import nnx

from ..activations import get_activation
from ..layers import Linear
from ..utils import ConnectionPolicy
from ..utils import as_rngs
from ..utils.initialization import compute_dtype
from ..utils.initialization import input_array
from ..utils.initialization import log_amplitude
from ..utils.initialization import positive_int
from .base import LogWavefunction


class MLP(LogWavefunction):
    r"""A multilayer perceptron representing :math:`\log\psi(x)`.

    With real parameters, a two-channel head represents log modulus and phase
    independently. ``complex_output=False`` uses a single output channel,
    which is real when the computation dtype is real. Inputs are divided
    by ``input_scale`` before entering the network, which can prevent saturation
    when a physical cutoff increases the range of integer labels.

    Connected-state evaluation reuses the first affine map through the exact
    identity :math:`W(x+\delta)+b=Wx+b+W\delta`. Every later layer is evaluated
    normally. Automatic reuse is conservative for small input dimensions.

    Parameters
    ----------
    in_features : int
        Number of scalar configuration coordinates.
    hidden_features : sequence of int
        Width of each hidden layer. At least one layer is required.
    activation : str or callable
        Registered name or pure activation function.
    complex_output : bool
        Use separate log-modulus and phase outputs.
    input_scale : positive float
        Fixed divisor applied to each input coordinate.
    dtype, param_dtype : dtype
        Computation and parameter dtypes, respectively.
    precision : JAX precision setting, optional
        Precision of affine contractions.
    connection_policy : ConnectionPolicy, optional
        Override automatic sparse-update dispatch, for example to benchmark it.
    rngs : NNX RNG stream, integer seed, or PRNG key
        Parameter initialization source.
    """

    in_features: int
    """Number of input coordinates."""

    hidden: nnx.List
    """Ordered affine hidden layers, available as separate model stages."""

    head: Linear
    """Final log-amplitude and optional phase projection."""

    activation: object
    """Activation callable resolved at construction."""

    input_scale: float
    """Fixed divisor for the input labels."""

    complex_output: bool
    """Whether the head has separate real amplitude and phase channels."""

    dtype: object
    """Array dtype used for forward contractions."""

    def __init__(
        self,
        in_features,
        hidden_features=(64, 64),
        *,
        activation="tanh",
        complex_output=True,
        input_scale=1.0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        connection_policy=None,
        rngs,
    ):
        """Initialize a sequence of dense layers and a small output projection."""
        self.in_features = positive_int(in_features, "in_features")
        widths = tuple(positive_int(w, "hidden_features") for w in hidden_features)
        if not widths:
            raise ValueError("MLP requires at least one hidden layer.")
        self.input_scale = float(input_scale)
        if not 0 < self.input_scale < float("inf"):
            raise ValueError("input_scale must be finite and positive.")
        dtype = compute_dtype(dtype, param_dtype)
        self.dtype = dtype
        self.complex_output = bool(complex_output)
        self.activation = get_activation(activation, dtype=dtype)
        self.connection_policy = connection_policy or ConnectionPolicy()
        rngs = as_rngs(rngs)
        dimensions = (self.in_features, *widths)
        self.hidden = nnx.List(
            [
                Linear(
                    a,
                    b,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                )
                for a, b in pairwise(dimensions)
            ]
        )
        self.head = Linear(
            widths[-1],
            2 if complex_output else 1,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.normal(0.01),
        )

    def _tail(self, fields):
        hidden = self.activation(fields)
        for layer in self.hidden[1:]:
            hidden = self.activation(layer(hidden))
        return log_amplitude(self.head(hidden), self.complex_output)

    def __call__(self, states):
        """Evaluate one log amplitude for each input configuration."""
        return self._tail(self.prepare_connections(states))

    def prepare_connections(self, sources):
        """Compute the first affine fields using the current parameters."""
        return self.hidden[0](
            input_array(sources, self.in_features, self.dtype) / self.input_scale
        )

    def apply_connections(self, cache, owners, sites, differences):
        """Update the first fields exactly and evaluate all remaining layers."""
        delta = self.hidden[0].sparse_delta(sites, differences / self.input_scale)
        return self._tail(cache[owners] + delta)

    def use_connection_reuse(self, source_shape, target_shape):
        """Automatically reuse only large input maps with wide first hidden layers.

        Inferring sparse changes and differentiating indexed updates can cost
        more than a small dense contraction. Automatic dispatch therefore
        requires at least 512 inputs and 512 first-layer outputs. An explicit
        ``ConnectionPolicy(mode="on")`` bypasses this cost check for profiling.
        """
        if self.connection_policy.mode == "auto" and (
            self.in_features < 512 or self.hidden[0].out_features < 512
        ):
            return False
        return super().use_connection_reuse(source_shape, target_shape)
