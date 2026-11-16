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


"""Registered standard activations and stable logarithmic wavefunction functions.

Standard real arithmetic and derivatives use JAX's optimized primitives. Each
public callable is registered explicitly with a checked input domain. Functions
based on ordering or real probability normalization are real-only here, even
when a backend happens to accept complex arrays. ``reim`` supplies an explicit
componentwise complex extension when that is the intended model.

Native complex support does not imply holomorphicity. For example, ``soft_sign``
uses the complex modulus, while ``logcosh`` and ``squareplus`` have branch cuts.
"""

import jax.numpy as jnp

from jax import nn as jax_nn

from .registry import activation

celu = activation(jax_nn.celu, name="celu", elementwise=True, supports_complex=False)
"""Continuously differentiable ELU on real inputs."""

elu = activation(jax_nn.elu, name="elu", elementwise=True, supports_complex=False)
"""Exponential linear unit on real inputs."""

gelu = activation(jax_nn.gelu, name="gelu", elementwise=True, supports_complex=False)
"""Real GELU, retaining JAX's exact and approximate formulations."""

glu = activation(jax_nn.glu, name="glu", elementwise=False, supports_complex=True)
"""Gated linear unit on the selected axis, with native complex sigmoid gates."""

hard_sigmoid = activation(
    jax_nn.hard_sigmoid, name="hard_sigmoid", elementwise=True, supports_complex=False
)
"""Piecewise linear sigmoid on real inputs."""

hard_silu = activation(
    jax_nn.hard_silu, name="hard_silu", elementwise=True, supports_complex=False
)
"""Hard SiLU on real inputs."""

hard_swish = activation(
    jax_nn.hard_swish, name="hard_swish", elementwise=True, supports_complex=False
)
"""Hard swish on real inputs."""

hard_tanh = activation(
    jax_nn.hard_tanh, name="hard_tanh", elementwise=True, supports_complex=False
)
"""Clipped hyperbolic tangent approximation on real inputs."""

identity = activation(
    jax_nn.identity, name="identity", elementwise=True, supports_complex=True
)
"""Identity on real or complex arrays."""

leaky_relu = activation(
    jax_nn.leaky_relu, name="leaky_relu", elementwise=True, supports_complex=False
)
"""Real ReLU with a configurable negative slope."""

log_sigmoid = activation(
    jax_nn.log_sigmoid, name="log_sigmoid", elementwise=True, supports_complex=False
)
"""Real log sigmoid evaluated with JAX's stable softplus implementation."""

log_softmax = activation(
    jax_nn.log_softmax, name="log_softmax", elementwise=False, supports_complex=False
)
"""Log-normalized exponentials over the selected real input axis."""

mish = activation(jax_nn.mish, name="mish", elementwise=True, supports_complex=False)
"""Mish on real inputs."""

relu = activation(jax_nn.relu, name="relu", elementwise=True, supports_complex=False)
"""Pointwise maximum of a real input and zero."""

relu6 = activation(jax_nn.relu6, name="relu6", elementwise=True, supports_complex=False)
"""Real ReLU clipped above at six."""

selu = activation(jax_nn.selu, name="selu", elementwise=True, supports_complex=False)
"""Scaled exponential linear unit on real inputs."""

sigmoid = activation(
    jax_nn.sigmoid, name="sigmoid", elementwise=True, supports_complex=True
)
"""Native logistic sigmoid on real or complex arrays."""

silu = activation(jax_nn.silu, name="silu", elementwise=True, supports_complex=True)
"""Native SiLU, the input multiplied by its logistic sigmoid."""

soft_sign = activation(
    jax_nn.soft_sign, name="soft_sign", elementwise=True, supports_complex=True
)
"""The input divided by one plus its modulus, including for complex arrays."""

softmax = activation(
    jax_nn.softmax, name="softmax", elementwise=False, supports_complex=False
)
"""Normalized exponentials over the selected real input axis."""

softplus = activation(
    jax_nn.softplus, name="softplus", elementwise=True, supports_complex=False
)
"""Real softplus evaluated with JAX's stable log-add-exp primitive."""

sparse_plus = activation(
    jax_nn.sparse_plus, name="sparse_plus", elementwise=True, supports_complex=False
)
"""Sparse plus on real inputs."""

sparse_sigmoid = activation(
    jax_nn.sparse_sigmoid,
    name="sparse_sigmoid",
    elementwise=True,
    supports_complex=False,
)
"""Sparse sigmoid on real inputs."""

squareplus = activation(
    jax_nn.squareplus, name="squareplus", elementwise=True, supports_complex=True
)
"""Squareplus with the principal square root for complex inputs."""

swish = activation(jax_nn.swish, name="swish", elementwise=True, supports_complex=True)
"""Native swish on real or complex arrays."""

tanh = activation(jax_nn.tanh, name="tanh", elementwise=True, supports_complex=True)
"""Native hyperbolic tangent on real or complex arrays."""


@activation(elementwise=True, supports_complex=True)
def logcosh(x):
    r"""Evaluate :math:`\log(\cosh x)` without overflowing at large real parts.

    Real inputs use a stable even expression. Complex inputs return the
    principal logarithm, with imaginary part in :math:`[-\pi,\pi]` up to
    roundoff. Its branch cut and the zeros of ``cosh`` are genuine
    singularities. No clipping or smoothing is applied to them.

    Integer inputs are promoted to floating point. The result has the same
    shape as the input and supports JAX differentiation away from singularities.
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.result_type(x, 0.0))
    u = jnp.where(x.real >= 0, x, -x)
    value = u + jnp.log1p(jnp.exp(-2 * u)) - jnp.log(jnp.asarray(2, x.dtype))
    if jnp.iscomplexobj(x):
        phase = jnp.arctan2(jnp.sin(value.imag), jnp.cos(value.imag))
        return value.real + 1j * phase
    return value


__all__ = [
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "identity",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logcosh",
    "mish",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "sparse_plus",
    "sparse_sigmoid",
    "squareplus",
    "swish",
    "tanh",
]
