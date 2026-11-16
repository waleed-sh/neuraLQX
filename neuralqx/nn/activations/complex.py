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


"""Explicit complex extensions of real activation functions."""

import jax.numpy as jnp

import jax

from . import functions
from .registry import _checked_activation
from .registry import activation
from .registry import activation_spec


def reim(function, **options):
    r"""Apply a real activation independently to real and imaginary components.

    For :math:`z=x+iy` and a real-valued function :math:`f`, this constructs
    :math:`g(z)=f(x)+i f(y)`. Real inputs use :math:`f(x)` and remain real.
    Complex inputs with zero imaginary part still receive :math:`i f(0)`, which
    need not vanish. This is an explicit choice of complex activation, not the
    analytic continuation of the original real function.

    Parameters
    ----------
    function : str or callable
        Registered activation name or function that returns real arrays for
        real inputs. Registered elementwise behavior is preserved, which lets
        CNN and GNN models retain their local reuse paths.
    **options
        Keyword arguments bound to the activation, such as
        ``reim("gelu", approximate=False)``. Keywords supplied to the returned
        callable override these defaults.

    Returns
    -------
    callable
        A JAX-compatible activation accepting real or complex arrays. It carries
        its domain and locality contract without adding a global registry entry.
        Use ``nn.activation`` to give a custom composition a registered name.

    Notes
    -----
    The real-coordinate Jacobian is diagonal for an elementwise activation,
    with entries :math:`f'(x)` and :math:`f'(y)`. These are generally different,
    so the result is generally non-holomorphic. JAX differentiates both channels
    normally. Do not assume ``holomorphic=True`` for a model using this wrapper.

    Examples
    --------
    ``nn.reim("relu")`` returns a callable that sends ``-2 + 3j`` to ``3j``.
    Use ``activation=nn.reim("gelu", approximate=False)`` in a model to select
    the exact real GELU independently on both components. The registered name
    ``"reim_gelu"`` selects the same construction with JAX's default approximation.
    """
    spec = activation_spec(function)
    bound_options = dict(options)

    def split(x, *args, **kwargs):
        x = jnp.asarray(x)
        arguments = bound_options | kwargs
        real = jnp.asarray(spec.function(x.real, *args, **arguments))
        if jnp.iscomplexobj(real):
            raise TypeError(
                "reim requires an activation with real outputs for real inputs."
            )
        if not jnp.iscomplexobj(x):
            return real
        imaginary = jnp.asarray(spec.function(x.imag, *args, **arguments))
        if jnp.iscomplexobj(imaginary):
            raise TypeError(
                "reim requires an activation with real outputs for real inputs."
            )
        if real.shape != imaginary.shape:
            raise ValueError(
                "The real and imaginary activation outputs must have the same shape."
            )
        dtype = jnp.result_type(real, imaginary, x.real)
        # Build the components directly so an infinite imaginary output cannot
        # introduce a spurious NaN into the real component through 0 * inf.
        return jax.lax.complex(real.astype(dtype), imaginary.astype(dtype))

    name = f"reim_{spec.name}"
    split.__name__ = name
    split.__doc__ = (
        f"Apply {spec.name} independently to real and imaginary components.\n\n"
        "Real inputs keep the ordinary real output. Complex inputs return\n"
        "f(real(z)) + 1j * f(imag(z)). This is generally non-holomorphic.\n"
        "Both channels are differentiated by ordinary JAX autodiff.\n"
        "See reim for the input-domain and locality contracts."
    )
    return _checked_activation(
        split, name=name, elementwise=spec.elementwise, supports_complex=True
    )


def _named_reim(function):
    fn = reim(function)
    spec = activation_spec(fn)
    return activation(
        fn, name=spec.name, elementwise=spec.elementwise, supports_complex=True
    )


reim_relu = _named_reim(functions.relu)
"""ReLU applied independently to the real and imaginary components."""

reim_gelu = _named_reim(functions.gelu)
"""GELU applied independently, accepting the exact or approximate real formulation."""

reim_selu = _named_reim(functions.selu)
"""SELU applied independently to the real and imaginary components."""

reim_silu = _named_reim(functions.silu)
"""SiLU applied independently, distinct from the native complex SiLU function."""

reim_tanh = _named_reim(functions.tanh)
"""Independent real hyperbolic tangents, distinct from the native complex tangent."""

reim_sigmoid = _named_reim(functions.sigmoid)
"""Independent real sigmoids, including a one-half imaginary output on the complex real axis."""

reim_softplus = _named_reim(functions.softplus)
"""Stable real softplus applied independently to both components."""

reim_logcosh = _named_reim(functions.logcosh)
"""Independent stable real log-cosh functions, distinct from native complex log-cosh."""


__all__ = [
    "reim",
    "reim_gelu",
    "reim_logcosh",
    "reim_relu",
    "reim_selu",
    "reim_sigmoid",
    "reim_silu",
    "reim_softplus",
    "reim_tanh",
]
