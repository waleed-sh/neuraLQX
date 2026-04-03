#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Public registration APIs for non-Struct pytree types.

These helpers let you bring arbitrary Python classes under the same pytree and
serialisation contracts used by :class:`neuralqx.utils.struct.Struct`.

Not every domain type should inherit from :class:`Struct`. Sometimes you need to
adapt an existing class hierarchy (third-party objects, legacy models, minimal
runtime wrappers) while still participating in:
    - JAX tree traversal and transformation,
    - Struct I/O export/load pipelines,
    - reference-based reconstruction across sessions.

This module provides two registration levels:

1. :func:`register_pytree_type`
   Full-control API when you want custom flatten/unflatten behaviour.

2. :func:`register_attrs_type`
   Convenience API when adaptation is naturally attribute-based.

Both APIs register adapters in:
- JAX pytree registry (runtime transformation support),
- struct adapter registry (serialisation and lazy reconstruction support).
"""

from __future__ import annotations

import dataclasses

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence

from typing import Any

import jax

from ._internals.pytree_registry import cache_ref
from ._internals.pytree_registry import get_spec_by_class
from ._internals.pytree_registry import get_spec_by_ref
from ._internals.pytree_registry import register_jax_type
from ._internals.pytree_registry import store_spec

from ._internals._types import FlattenFn
from ._internals._types import UnflattenFn
from ._internals._types import FlattenWithKeysFn
from ._internals._types import SerializerFn
from ._internals._types import DeserializerFn

from .registry import class_ref
from .registry import register_class_ref
from .registry import resolve_class

from neuralqx.utils.errors import SerializationError

__all__ = [
    "PyTreeTypeSpec",
    "register_pytree_type",
    "register_attrs_type",
    "is_registered_pytree_type",
    "resolve_pytree_spec",
]


@dataclasses.dataclass(frozen=True, slots=True)
class PyTreeTypeSpec:
    """
    Complete pytree + serialisation spec for a registered type.

    Notes:
        A ``PyTreeTypeSpec`` is effectively an adapter contract for one class.
        Runtime systems use this single specification for:
            - tree flattening/unflattening,
            - optional serialiser-based persistence paths,
            - lazy resolution by reference at load time.
    """

    cls: type[Any]
    """Target class being adapted."""

    flatten: FlattenFn
    """Core pytree conversion functions in JAX style."""

    unflatten: UnflattenFn
    """Core pytree conversion functions in JAX style."""

    flatten_with_keys: FlattenWithKeysFn | None = None
    """Optional key-aware flatten variant for richer tree metadata."""

    serializer: SerializerFn | None = None
    """Optional pair used by struct I/O to serialize types that do not round-trip cleanly through plain ``flatten`` aux-data."""

    deserializer: DeserializerFn | None = None
    """Optional pair used by struct I/O to serialize types that do not round-trip cleanly through plain ``flatten`` aux-data."""


def register_pytree_type(
    cls: type[Any],
    *,
    flatten: FlattenFn,
    unflatten: UnflattenFn,
    flatten_with_keys: FlattenWithKeysFn | None = None,
    serializer: SerializerFn | None = None,
    deserializer: DeserializerFn | None = None,
) -> type[Any]:
    """
    Register an arbitrary class as a pytree-capable and serialisable type.

    This is the low-level adapter API. Use this when you need full control over
    flattening and reconstruction behaviour.

    Args:
        cls: Target class to adapt.
        flatten: Callable mapping ``obj -> (children, aux)``.
        unflatten: Callable mapping ``(aux, children) -> obj``.
        flatten_with_keys: Optional key-aware flatten variant for richer tree metadata.
        serializer: Optional persistence path where raw ``flatten`` aux-data is not
            directly serialisable or where custom wire format is preferred.
        deserializer: Optional persistence path where raw ``flatten`` aux-data is not
            directly serialisable or where custom wire format is preferred.


    Returns:
        The original class, allowing decorator-style usage.

    Raises:
        TypeError: Propagated if provided callables are invalid at runtime use.

    Example:
        .. code-block::

            class Node:
                def __init__(self, data, tag):
                    self.data = data
                    self.tag = tag

            register_pytree_type(
                Node,
                flatten=lambda obj: ([obj.data], obj.tag),
                unflatten=lambda aux, children: Node(children[0], aux),
            )
    """
    spec = PyTreeTypeSpec(
        cls=cls,
        flatten=flatten,
        unflatten=unflatten,
        flatten_with_keys=flatten_with_keys,
        serializer=serializer,
        deserializer=deserializer,
    )
    register_jax_type(
        cls,
        flatten=spec.flatten,
        unflatten=spec.unflatten,
        flatten_with_keys=spec.flatten_with_keys,
    )

    ref = class_ref(cls)
    store_spec(ref, cls, spec)
    register_class_ref(cls)
    return cls


def register_attrs_type(
    cls: type[Any],
    *,
    node_fields: Sequence[str],
    static_fields: Sequence[str] = (),
    constructor: Callable[[Mapping[str, Any]], Any] | None = None,
    serializer: SerializerFn | None = None,
    deserializer: DeserializerFn | None = None,
) -> type[Any]:
    """
    Register a class by naming its node/static attributes.

    This higher-level API is useful when your class is attribute-backed and you do
    not want to implement flatten/unflatten manually.

    Args:
        cls: Class to register.
        node_fields: Attribute names treated as pytree leaves.
        static_fields: Attribute names treated as pytree auxiliary metadata.
        constructor: Optional callable used to construct instances from resolved attribute map.
            If omitted, ``cls(**values)`` is attempted first, then attribute fallback.
        serializer: Optional persistence adapters (same semantics as
            :func:`register_pytree_type`).
        deserializer: Optional persistence adapters (same semantics as
            :func:`register_pytree_type`).

    Returns:
        The original class.

    Notes:
        This function is intentionally permissive for reconstruction:
            - it first tries ``cls(**values)``,
            - then falls back to ``object.__new__(cls)`` + ``setattr`` assignment.
              This makes it robust for both dataclass-like and legacy classes.
    """

    node_names = tuple(node_fields)
    static_names = tuple(static_fields)

    def _flatten_with_keys(obj: Any) -> tuple[list[tuple[Any, Any]], Any]:
        nodes = [
            (jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in node_names
        ]
        aux = tuple(getattr(obj, name) for name in static_names)
        return nodes, aux

    def _flatten(obj: Any) -> tuple[list[Any], Any]:
        nodes = [getattr(obj, name) for name in node_names]
        aux = tuple(getattr(obj, name) for name in static_names)
        return nodes, aux

    def _unflatten(aux: Any, children: Sequence[Any]) -> Any:
        values: dict[str, Any] = {}
        for name, value in zip(node_names, children):
            values[name] = value
        for name, value in zip(static_names, aux):
            values[name] = value

        if constructor is not None:
            return constructor(values)

        try:
            return cls(**values)
        except Exception:
            obj = object.__new__(cls)
            for k, v in values.items():
                setattr(obj, k, v)
            return obj

    return register_pytree_type(
        cls,
        flatten=_flatten,
        unflatten=_unflatten,
        flatten_with_keys=_flatten_with_keys,
        serializer=serializer,
        deserializer=deserializer,
    )


def is_registered_pytree_type(cls: type[Any]) -> bool:
    """
    Return whether ``cls`` currently has an active pytree adapter.

    This is useful for idempotent registration patterns in large applications
    where modules may be imported in different orders.
    """
    return get_spec_by_class(cls) is not None


def resolve_pytree_spec(ref: str) -> PyTreeTypeSpec:
    """
    Resolve a pytree type spec from class reference.

    Resolution order:
        1. In-memory ref cache.
        2. Import class via class registry, then class cache.
        3. Auto-register classes implementing ``tree_flatten/tree_unflatten``.

    Args:
        ref: Class reference string in ``"module:qualname"`` format.

    Returns:
        Resolved adapter specification.

    Raises:
        SerializationError: If no adapter is found/derivable for the given reference.

    Notes:
        If no explicit spec is found but the resolved class implements both
        ``tree_flatten`` and ``tree_unflatten``, a default adapter is registered
        automatically. This mirrors common JAX ecosystem conventions.

    """
    spec = get_spec_by_ref(ref)
    if spec is not None:
        return spec  # type: ignore[return-value]

    cls = resolve_class(ref)
    spec = get_spec_by_class(cls)
    if spec is not None:
        cache_ref(ref, spec)
        return spec  # type: ignore[return-value]

    flatten = getattr(cls, "tree_flatten", None)
    unflatten = getattr(cls, "tree_unflatten", None)
    if callable(flatten) and callable(unflatten):

        def _flatten(obj: Any) -> tuple[Sequence[Any], Any]:
            return flatten(obj)

        def _unflatten(aux: Any, children: Sequence[Any]) -> Any:
            return unflatten(aux, children)

        register_pytree_type(cls, flatten=_flatten, unflatten=_unflatten)
        registered = get_spec_by_class(cls)
        if registered is not None:
            return registered  # type: ignore[return-value]

    raise SerializationError(
        f"No pytree registration found for {ref!r}. "
        "Use register_pytree_type(...) or register_attrs_type(...)."
    )
