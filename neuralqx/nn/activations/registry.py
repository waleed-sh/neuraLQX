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


"""Registered activations with explicit input domains and locality contracts."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import jax.numpy as jnp


@dataclass(frozen=True)
class ActivationSpec:
    """Describe a pure activation and its permitted input domain.

    Local network updates may reuse unaffected features only when the
    activation acts independently on each element. Custom activations are
    conservatively treated as nonlocal until their author declares otherwise.
    Registered callables enforce their complex-input contract during eager
    evaluation or JAX tracing. These dtype checks add no operations to the
    compiled array computation. Locality remains an author-supplied contract.
    """

    name: str
    """Name accepted by :func:`get_activation`."""

    function: Callable
    """Pure JAX callable implementing the activation."""

    elementwise: bool = False
    """Whether each output depends on only the corresponding input element."""

    supports_complex: bool | None = None
    """Whether complex inputs are permitted, or ``None`` for an unregistered callable.

    ``False`` is enforced by registered callables. ``True`` declares an actual
    complex implementation, which may be non-holomorphic. It never enables
    holomorphic differentiation or converts a real activation automatically.
    """

    def validate_dtype(self, dtype):
        """Reject a complex computation dtype for an explicitly real-only activation."""
        if (
            dtype is not None
            and self.supports_complex is False
            and jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)
        ):
            raise TypeError(
                f"Activation {self.name!r} accepts real inputs only. "
                f"Use reim({self.name!r}) to apply it separately to real and imaginary "
                "components, or choose an activation with a native complex implementation."
            )


_REGISTRY: dict[str, ActivationSpec] = {}


def _checked_activation(function, *, name, elementwise, supports_complex):
    try:
        input_name = next(iter(inspect.signature(function).parameters))
    except (TypeError, ValueError, StopIteration):
        input_name = "x"

    @wraps(function)
    def checked(*args, **kwargs):
        if supports_complex is False:
            value = args[0] if args else kwargs.get(input_name)
            if value is not None:
                dtype = getattr(value, "dtype", None)
                if dtype is None:
                    dtype = jnp.asarray(value).dtype
                spec.validate_dtype(dtype)
        return function(*args, **kwargs)

    spec = ActivationSpec(name, checked, elementwise, supports_complex)
    checked._neuralqx_activation_spec = spec
    domain_note = (
        "Complex inputs are rejected. Use neuralqx.nn.reim for an explicit "
        "real/imaginary extension."
        if supports_complex is False
        else "Complex inputs use this activation's declared implementation. "
        "Complex support does not imply holomorphicity."
    )
    documentation = function.__doc__ or f"Evaluate the {name} activation."
    checked.__doc__ = (
        documentation
        if documentation.endswith(domain_note)
        else documentation + "\n\n" + domain_note
    )
    return checked


def activation(
    function=None, *, name=None, elementwise=None, supports_complex=None, replace=False
):
    """Register an activation with a checked input domain and optional locality contract.

    Use ``@nn.activation`` for a custom function, or
    ``@nn.activation(name="my_tanh", elementwise=True, supports_complex=True)``
    to declare additional capabilities. The function must be pure, accept JAX
    arrays, and work under JIT and autodiff. The returned callable preserves the
    function's signature and derivatives. Complex-input validation runs while
    tracing and adds no kernels, transfers or registry lookups to compiled code.

    Parameters
    ----------
    function : callable, optional
        Function to register. Omit it when using decorator arguments.
    name : str, optional
        Registry key. Defaults to the function's name.
    elementwise : bool, optional
        Declare independence between input elements. A false declaration can
        invalidate optimizations that reuse local network features.
        Omission preserves an existing activation contract, or uses false for
        a new function whose locality is unknown.
    supports_complex : bool, optional
        When false, the registered function rejects complex input arrays.
        When true, the supplied implementation is responsible for its complex
        behavior. This declaration does not assert holomorphicity. Use ``reim``
        when the intended behavior is independent real and imaginary activation.
        Omission preserves an existing contract, or uses false for a new function.
    replace : bool
        Allow replacing an existing name. Models already constructed retain
        their resolved callable.
    """

    def register(fn):
        if not callable(fn):
            raise TypeError("An activation must be callable.")
        key = name if name is not None else getattr(fn, "__name__", None)
        if not isinstance(key, str) or not key.strip():
            raise ValueError("An activation needs a nonempty name.")
        if key in _REGISTRY and not replace:
            raise ValueError(f"Activation {key!r} is already registered.")
        if any(
            flag is not None and not isinstance(flag, bool)
            for flag in (elementwise, supports_complex)
        ):
            raise TypeError(
                "elementwise and supports_complex must be booleans or None."
            )
        inherited = getattr(fn, "_neuralqx_activation_spec", None)
        if not isinstance(inherited, ActivationSpec) or inherited.function is not fn:
            inherited = ActivationSpec(key, fn, False, False)
        locality = inherited.elementwise if elementwise is None else elementwise
        complex_domain = (
            inherited.supports_complex if supports_complex is None else supports_complex
        )
        checked = _checked_activation(
            fn, name=key, elementwise=locality, supports_complex=complex_domain
        )
        _REGISTRY[key] = checked._neuralqx_activation_spec
        return checked

    return register if function is None else register(function)


def activation_spec(value, *, dtype=None) -> ActivationSpec:
    """Resolve a registered name or describe an arbitrary callable.

    ``dtype`` optionally checks compatibility before model parameters are
    allocated. Unregistered callables remain usable with unknown complex
    support and conservative locality. Functions returned by ``reim`` retain
    their contracts even when they have not been given a registry name.
    """
    if isinstance(value, str):
        try:
            spec = _REGISTRY[value]
        except KeyError:
            raise ValueError(
                f"Unknown activation {value!r}. Available: {', '.join(available_activations())}"
            ) from None
    else:
        if not callable(value):
            raise TypeError("An activation must be a registered name or a callable.")
        attached = getattr(value, "_neuralqx_activation_spec", None)
        if isinstance(attached, ActivationSpec) and attached.function is value:
            spec = attached
        else:
            spec = next(
                (
                    registered
                    for registered in _REGISTRY.values()
                    if registered.function is value
                    or getattr(registered.function, "__wrapped__", None) is value
                ),
                ActivationSpec(getattr(value, "__name__", type(value).__name__), value),
            )
    spec.validate_dtype(dtype)
    return spec


def get_activation(value, *, dtype=None) -> Callable:
    """Resolve a callable, optionally checking its declared domain against ``dtype``.

    Registered functions retain their dtype guard when called directly. An
    unregistered callable is passed through without guessing its complex domain.
    """
    return activation_spec(value, dtype=dtype).function


def available_activations() -> tuple[str, ...]:
    """Return the registered activation names in alphabetical order."""
    return tuple(sorted(_REGISTRY))
