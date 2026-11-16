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


"""Quadratic log amplitudes with exact sparse differences."""

import jax.numpy as jnp
from flax import nnx

from ..utils import ConnectionPolicy
from ..utils import as_rngs
from ..utils import partitioned
from ..utils.initialization import compute_dtype
from ..utils.initialization import input_array
from ..utils.initialization import log_amplitude
from ..utils.initialization import positive_int
from .base import LogWavefunction


class Jastrow(LogWavefunction):
    r"""Represent each log-amplitude channel as a quadratic function of the state.

    For a symmetric matrix :math:`J` and vector :math:`a`, the channel is
    :math:`\ell(x)=a^T x+\tfrac12 x^T Jx`. Complex output uses independent
    real channels for modulus and phase with the default parameter dtype.
    Complex parameters are also supported. The stored matrix is symmetrized in
    the forward calculation, so no optimizer-side symmetry constraint is needed.

    Sparse updates use
    :math:`\ell(x+\delta)-\ell(x)=(Jx+a)^T\delta+\tfrac12\delta^T J\delta`.
    This includes every diagonal and cross term, even when several coordinates
    change. Parameter-axis annotations can be supplied as ``coupling_axes``
    and ``linear_axes``. The final parameter axis indexes output channels.
    """

    coupling: nnx.Param
    """Raw coupling tensor of shape ``(in_features, in_features, channels)``."""

    linear: nnx.Param
    """Linear coefficients of shape ``(in_features, channels)``."""

    in_features: int
    """Number of scalar configuration coordinates."""

    complex_output: bool
    """Whether the second channel represents a phase."""

    dtype: object
    """Computation dtype."""

    precision: object
    """Precision passed to JAX tensor contractions."""

    def __init__(
        self,
        in_features,
        *,
        complex_output=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        coupling_axes=None,
        linear_axes=None,
        connection_policy=None,
        rngs,
    ):
        """Initialize weak quadratic and linear couplings."""
        self.in_features = positive_int(in_features, "in_features")
        self.complex_output = bool(complex_output)
        dtype = compute_dtype(dtype, param_dtype)
        self.dtype = dtype
        self.precision = precision
        self.connection_policy = connection_policy or ConnectionPolicy(min_features=128)
        rngs = as_rngs(rngs)
        n = self.in_features
        channels = 2 if complex_output else 1
        initializer = partitioned(nnx.initializers.normal(0.1 / n), coupling_axes)
        self.coupling = nnx.Param(
            initializer(rngs.params(), (n, n, channels), param_dtype)
        )
        initializer = partitioned(nnx.initializers.normal(0.05 / n**0.5), linear_axes)
        self.linear = nnx.Param(initializer(rngs.params(), (n, channels), param_dtype))

    def __call__(self, states):
        """Return a quadratic log amplitude for each configuration."""
        return self.prepare_connections(states)[2]

    def prepare_connections(self, sources):
        """Compute symmetric couplings, local fields and source log amplitudes."""
        x = input_array(sources, self.in_features, self.dtype)
        raw = jnp.asarray(self.coupling[...], self.dtype)
        matrix = (raw + raw.swapaxes(0, 1)) * 0.5
        linear = jnp.asarray(self.linear[...], self.dtype)
        field = jnp.einsum("...i,ijc->...jc", x, matrix, precision=self.precision)
        value = jnp.einsum(
            "...i,...ic->...c", x, 0.5 * field + linear, precision=self.precision
        )
        return matrix, field + linear, log_amplitude(value, self.complex_output)

    def apply_connections(self, cache, owners, sites, differences):
        """Apply the exact linear and quadratic terms in a configuration change."""
        matrix, field, base = cache
        delta = jnp.asarray(differences, self.dtype)
        local_field = field[owners[:, None], sites]
        local_matrix = matrix[sites[:, :, None], sites[:, None, :]]
        change = jnp.einsum("bi,bic->bc", delta, local_field, precision=self.precision)
        change += 0.5 * jnp.einsum(
            "bi,bijc,bj->bc", delta, local_matrix, delta, precision=self.precision
        )
        return base[owners] + log_amplitude(change, self.complex_output)
