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


"""
Public struct base classes and decorators.

This module defines:
    - :class:`StructMeta`: metaclass that processes classes into immutable pytrees.
    - :class:`Struct`: ergonomic user-facing base class.
    - decorator helpers for registering existing classes as structs.

This module is the primary user entry point for defining immutable, traceable
state containers that behave well under JAX transformations and neuralQX
serialisation flows.

Compared to plain dataclasses, Struct adds:
    - explicit node/static/opaque field semantics,
    - deterministic freeze/validation lifecycle,
    - built-in pytree registration,
    - integrated state export/load methods.

Example

.. code-block::

    import neuralqx as nqx
    s = nqx.utils.struct

    class State(s.Struct):
        params: object
        name: str = s.field(static=True, default="exp")

    st = State(params={"w": [1, 2, 3]})
    st2 = st.replace(name="exp-2")
"""

from __future__ import annotations

import abc
import typing
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

from typing import Any
from typing import ClassVar
from typing import Self
from typing import dataclass_transform

import jax

from . import io as _io

from ._internals.runtime import INIT_FLAG
from ._internals.runtime import apply_converter
from ._internals.runtime import assign_initial_fields
from ._internals.runtime import finalize_instance
from ._internals.runtime import process_struct_class
from ._internals.runtime import safe_array_equal
from ._internals.runtime import summarize_value
from ._internals.runtime import DeferredDerivedValue
from ._internals.runtime import materialize_deferred_derived_field


from .fields import FieldSpec
from .fields import field

from neuralqx.utils.errors import FrozenStructError
from neuralqx.utils.errors import SerializationError

__all__ = [
    "Struct",
    "StructMeta",
    "StructABCMeta",
    "dataclass",
    "register_class",
    "fields",
    "node_fields",
    "static_fields",
    "opaque_fields",
    "derived_fields",
]


@dataclass_transform(field_specifiers=(field, FieldSpec))
class StructMeta(type):
    """
    Metaclass that finalises classes into Struct-compatible pytrees.

    Every subclass of :class:`Struct` is processed at class creation time:
        - field definitions are collected and normalised,
        - constructor signature is synthesised if needed,
        - JAX pytree hooks are registered,
        - class reference is cached for serialisation.

    The metaclass guarantees that class-level invariants are established exactly
    once when classes are defined, not at first instance construction. This
    provides:
        - predictable failure modes for invalid field declarations,
        - zero per-instance schema analysis overhead,
        - consistent runtime behaviour regardless of object creation path.
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        return cls

    def __init__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(name, bases, namespace, **kwargs)
        if name == "Struct" and cls.__module__ == __name__:
            return
        process_struct_class(typing.cast(type[Struct], cls))


class StructABCMeta(StructMeta, abc.ABCMeta):
    """Combined metaclass giving both Struct processing and ABC enforcement."""


class Struct(metaclass=StructMeta):
    """
    Immutable, JAX-native container with explicit field semantics.

    ``Struct`` is designed as a forward-looking abstraction for differentiable
    systems code:
        - deterministic construction and validation,
        - strict post-init immutability,
        - first-class pytree registration,
        - stable serialization contracts.

    Field behaviour is driven by :func:`neuralqx.utils.struct.field`.

    For generated constructors and reconstruction paths, the lifecycle is:
        1. assign user/default values to non-derived fields
        2. compute derived fields
        3. run ``__post_init__`` (if defined)
        4. recompute derived fields (post-init may have changed dependencies)
        5. validate values and static constraints
        6. freeze instance (further mutation raises :class:`FrozenStructError`)

    Struct instances are logically (almost) immutable after initialisation. Use
    :meth:`replace` to derive updated copies.

    Pytree contract:
        - Node fields are leaves.
        - Static fields are auxiliary data.
        - Opaque fields are auxiliary identity-preserving references.

    Serialisation contract:
        Struct exposes convenience wrappers over :mod:`neuralqx.utils.struct.io`:
            - :meth:`to_state_dict`
            - :meth:`from_state_dict`
            - :meth:`export`
            - :meth:`load`
    """

    __struct_fields__: ClassVar[Mapping[str, FieldSpec]] = MappingProxyType({})
    __struct_node_fields__: ClassVar[tuple[str, ...]] = ()
    __struct_static_fields__: ClassVar[tuple[str, ...]] = ()
    __struct_opaque_fields__: ClassVar[tuple[str, ...]] = ()
    __struct_derived_fields__: ClassVar[tuple[str, ...]] = ()
    __struct_processed__: ClassVar[bool] = False
    __hash__ = None

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if isinstance(value, DeferredDerivedValue):
            return materialize_deferred_derived_field(type(self), self, name)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name == INIT_FLAG:
            object.__setattr__(self, name, value)
            return
        initializing = getattr(self, INIT_FLAG, True)
        if initializing:
            object.__setattr__(self, name, value)
            return
        raise FrozenStructError(
            f"Cannot set attribute {name!r} on frozen {type(self).__name__}, use .replace()."
        )

    def __delattr__(self, name: str) -> None:
        raise FrozenStructError(
            f"Cannot delete attribute {name!r} from frozen {type(self).__name__}."
        )

    @classmethod
    def fields(cls) -> Mapping[str, FieldSpec]:
        """
        Return ordered mapping of field names to :class:`FieldSpec`.

        This is the canonical schema view for the class. The mapping order
        matches declaration/MRO resolution order used by constructor generation
        and derived-field evaluation.
        """
        return cls.__struct_fields__

    @classmethod
    def node_fields(cls) -> tuple[str, ...]:
        """
        Return names of fields traced as pytree leaves.

        These fields are visible to ``jax.tree_util.tree_leaves`` and therefore
        participate in transformations such as ``jit``, ``grad``, and ``vmap``.
        """
        return cls.__struct_node_fields__

    @classmethod
    def static_fields(cls) -> tuple[str, ...]:
        """
        Return names of static pytree metadata fields.

        Static fields are part of pytree aux data and influence JAX cache keys.
        They must remain hashable and array-free.
        """
        return cls.__struct_static_fields__

    @classmethod
    def opaque_fields(cls) -> tuple[str, ...]:
        """
        Return names of opaque runtime fields excluded from pytree leaves.

        Opaque fields are useful for runtime-only objects that should not be
        traced or serialized by default (for example handles or ephemeral tokens).
        """
        return cls.__struct_opaque_fields__

    @classmethod
    def derived_fields(cls) -> tuple[str, ...]:
        """
        Return names of fields computed from other fields.

        Derived fields are recomputed automatically during construction,
        replacement, and deserialisation.
        """
        return cls.__struct_derived_fields__

    def _init_fields(self, **kwargs: Any) -> None:
        object.__setattr__(self, INIT_FLAG, True)
        assign_initial_fields(
            type(self), self, bound=kwargs, reject_derived_inputs=True
        )
        finalize_instance(type(self), self, run_post_init=True, validate=True)

    @classmethod
    def __struct_unflatten__(
        cls,
        values: Mapping[str, Any],
        *,
        _defer_derived_on_trace_error: bool = False,
        _validate: bool = True,
    ) -> Self:
        """
        Reconstruct instance from decoded non-derived field values.

        This is an internal/public bridge used by pytree unflattening and
        deserialisation. It enforces converters/defaults/validation and then
        recomputes derived fields before returning a frozen instance.

        Args:
            values: Mapping of field names to decoded values for non-derived fields.

        Returns:
            Fully reconstructed and validated instance.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, INIT_FLAG, True)
        for spec in cls.__struct_fields__.values():
            if spec.is_derived:
                continue
            if spec.name in values:
                value = values[spec.name]
            elif spec.has_default:
                value = spec.make_default()
            else:
                raise SerializationError(
                    f"Missing field {spec.name!r} when reconstructing {cls.__name__}."
                )
            object.__setattr__(obj, spec.name, apply_converter(spec, obj, value))
        finalize_instance(
            cls,
            obj,
            run_post_init=False,
            validate=_validate,
            defer_derived_on_trace_error=_defer_derived_on_trace_error,
        )
        return typing.cast(Self, obj)

    def replace(self, **updates: Any) -> Self:
        """
        Return a new instance with selected field updates.

        Derived fields cannot be directly replaced, they are recomputed.
        Non-init fields are restored after constructor-based recreation.

        Args:
            updates: Field values to update.

        Returns:
            New instance with updates applied.

        Raises:
            TypeError: if unknown fields are provided or an update targets a derived field.

        Notes:
            ``replace`` preserves the class-level lifecycle contract: conversion,
            derivation, post-init hooks, and validation are re-executed.
        """
        unknown = set(updates) - set(type(self).__struct_fields__)
        if unknown:
            msg = ", ".join(repr(x) for x in sorted(unknown))
            raise TypeError(f"Unknown field(s) for replace: {msg}")
        for name in updates:
            spec = type(self).__struct_fields__[name]
            if spec.is_derived:
                raise TypeError(
                    f"Derived field {name!r} cannot be updated explicitly, "
                    "update its source fields instead."
                )

        # Exclude derived fields: they are recomputed by the constructor and
        # reading them here would unnecessarily materialise deferred values.
        full_values = {
            name: getattr(self, name)
            for name, spec in type(self).__struct_fields__.items()
            if not spec.is_derived
        }
        full_values.update(updates)

        init_kwargs = {
            spec.name: full_values[spec.name]
            for spec in type(self).__struct_fields__.values()
            if spec.init and not spec.is_derived
        }
        new_obj = type(self)(**init_kwargs)
        return new_obj._restore_non_init(full_values)

    def _restore_non_init(self, full_values: Mapping[str, Any]) -> Self:
        restore_specs = [
            spec
            for spec in type(self).__struct_fields__.values()
            if (not spec.init) and (not spec.is_derived)
        ]
        if not restore_specs:
            return typing.cast(Self, self)

        object.__setattr__(self, INIT_FLAG, True)
        for spec in restore_specs:
            object.__setattr__(
                self, spec.name, apply_converter(spec, self, full_values[spec.name])
            )
        # __post_init__ already ran inside type(self)(**init_kwargs) above.
        # Running it again here would execute it twice per replace() call.
        finalize_instance(type(self), self, run_post_init=False, validate=True)
        return typing.cast(Self, self)

    def rederive(self) -> Self:
        """
        Recompute derived fields in place and return ``self``.

        .. warning::
            This method exists as an escape hatch for advanced use cases where
            mutable objects held in node fields have been mutated externally.
            Relying on it indicates that the struct's immutability contract has
            been violated. Prefer designing fields as truly immutable values and
            using :meth:`replace` to produce updated copies instead.
        """
        object.__setattr__(self, INIT_FLAG, True)
        finalize_instance(type(self), self, run_post_init=False, validate=True)
        return typing.cast(Self, self)

    def to_dict(
        self, *, recursive: bool = False, include_opaque: bool = True
    ) -> dict[str, Any]:
        """
        Convert struct to dictionary representation.

        Args:
            recursive: When ``True``, nested values are converted with :func:`to_python`.
            include_opaque: Whether opaque runtime fields are included.

        Return:
            dict[str, Any]: Mapping from field names to values (raw or recursively converted).

        Notes:
            This is primarily an inspection helper. For persistence/transport,
            prefer :meth:`to_state_dict` or :meth:`export`.
        """
        out: dict[str, Any] = {}
        for name, spec in type(self).__struct_fields__.items():
            if spec.kind.value == "opaque" and not include_opaque:
                continue
            value = getattr(self, name)
            out[name] = (
                _io.to_python(value, include_opaque=include_opaque)
                if recursive
                else value
            )
        return out

    def to_state_dict(self) -> dict[str, Any]:
        """
        Encode this instance into a portable in-memory state payload.

        Returns:
            Payload consumable by :meth:`from_state_dict`.
        """
        return _io.to_state_dict(self)

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> Self:
        """
        Decode an instance from :meth:`to_state_dict` output.

        Args:
            payload: State payload previously produced by ``to_state_dict``.

        Returns:
            Reconstructed instance of ``cls``.
        """
        return typing.cast(Self, _io.from_state_dict(payload, expected_cls=cls))

    def export(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """
        Persist this instance to a directory bundle or ``.zip`` archive.

        Args:
            path: Output path.
            overwrite: Whether existing path may be replaced.

        Returns:
            Output location.
        """
        return _io.export(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load an instance from a previously exported bundle.

        Args:
            path: Directory bundle path or ``.zip`` archive path.

        Returns:
            Reconstructed instance of ``cls``.
        """
        return typing.cast(Self, _io.load(path, expected_cls=cls))

    def tree_size(self) -> int:
        """
        Return number of pytree leaves in this instance.

        Equivalent to ``len(jax.tree_util.tree_leaves(self))``.
        """
        return len(jax.tree_util.tree_leaves(self))

    def __repr__(self) -> str:
        parts = []
        for name, spec in type(self).__struct_fields__.items():
            if not spec.repr:
                continue
            parts.append(f"{name}={summarize_value(getattr(self, name))}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        other = typing.cast(Struct, other)
        for name, spec in type(self).__struct_fields__.items():
            if not spec.compare:
                continue
            if not safe_array_equal(getattr(self, name), getattr(other, name)):
                return False
        return True

    def __iter__(self):
        """
        Yield field values in declaration order.

        Iterates over all fields (node, static, opaque, and derived) in the
        order they were declared. This mirrors the behaviour of
        ``dataclasses.astuple`` and allows tuple-unpacking of simple structs.

        Note:
            Iteration includes opaque and derived fields. If you only want
            pytree leaves, use ``jax.tree_util.tree_leaves(self)`` instead.
        """
        for name in type(self).__struct_fields__:
            yield getattr(self, name)


@typing.overload
def register_class(cls: type[Any]) -> type[Any]: ...


@typing.overload
def register_class(*, name: str | None = None) -> Callable[[type[Any]], type[Any]]: ...


@dataclass_transform(field_specifiers=(field, FieldSpec))
def register_class(
    cls: type[Any] | None = None,
    *,
    name: str | None = None,
) -> type[Any] | Callable[[type[Any]], type[Any]]:
    """
    Register an existing class as a :class:`Struct` type.

    Supports both styles:

    .. code-block::

        @register_class
        class Graph:
            nodes: ...

        @register_class(name="RenamedGraph")
        class Graph:
            ...

    Args:
        cls: Class to register. When omitted, returns a decorator.
        name: Optional override for generated class name.

    Returns:
        Registered class or class decorator depending on invocation style.

    Notes:
        If the class already subclasses :class:`Struct`, registration is idempotent.
        Otherwise, a new class is created with MRO ``(raw_cls, Struct)`` so methods
        and ``super()`` semantics remain valid.

        - Existing class methods are preserved. Note that ``__init__`` is
          always replaced by the generated Struct constructor; any custom
          ``__init__`` on ``raw_cls`` will be discarded.

        - Zero-argument ``super()`` in class methods remains valid.

        - Struct processing (field collection, pytree registration, class reference
          caching) is applied exactly once.
    """

    def decorate(raw_cls: type[Any]) -> type[Any]:
        if issubclass(raw_cls, Struct):
            process_struct_class(raw_cls)
            return raw_cls

        namespace: dict[str, Any] = {
            "__module__": raw_cls.__module__,
            "__qualname__": raw_cls.__qualname__,
            "__doc__": raw_cls.__doc__,
        }

        annotations = dict(getattr(raw_cls, "__annotations__", {}))
        if annotations:
            namespace["__annotations__"] = annotations
            for field_name in annotations:
                if field_name in raw_cls.__dict__:
                    namespace[field_name] = raw_cls.__dict__[field_name]

        # Keep raw_cls first in MRO so zero-arg super() and method resolution
        # stay aligned with the original class definition.
        bases = (raw_cls, Struct)
        new_name = name or raw_cls.__name__
        new_cls = typing.cast(type[Any], StructMeta(new_name, bases, namespace))
        return new_cls

    if cls is None:
        return decorate
    return decorate(cls)


@typing.overload
def dataclass(cls: type[Any]) -> type[Any]: ...


@typing.overload
def dataclass(*, name: str | None = None) -> Callable[[type[Any]], type[Any]]: ...


@dataclass_transform(field_specifiers=(field, FieldSpec))
def dataclass(
    cls: type[Any] | None = None,
    *,
    name: str | None = None,
) -> type[Any] | Callable[[type[Any]], type[Any]]:
    """
    Dataclass-style alias for :func:`register_class`.

    This helper exists for readability in codebases that conceptually treat
    Struct definitions as immutable dataclasses with JAX semantics.
    """
    return register_class(cls, name=name)


def fields(cls: type[Struct[Any]]) -> Mapping[str, FieldSpec]:
    """
    Return class field mapping.

    Thin convenience wrapper over ``cls.fields()``.
    """
    return cls.fields()


def node_fields(cls: type[Struct[Any]]) -> tuple[str, ...]:
    """
    Return tuple of node-field names.

    Thin convenience wrapper over ``cls.node_fields()``.
    """
    return cls.node_fields()


def static_fields(cls: type[Struct[Any]]) -> tuple[str, ...]:
    """
    Return tuple of static-field names.

    Thin convenience wrapper over ``cls.static_fields()``.
    """
    return cls.static_fields()


def opaque_fields(cls: type[Struct[Any]]) -> tuple[str, ...]:
    """
    Return tuple of opaque-field names.

    Thin convenience wrapper over ``cls.opaque_fields()``.
    """
    return cls.opaque_fields()


def derived_fields(cls: type[Struct[Any]]) -> tuple[str, ...]:
    """
    Return tuple of derived-field names.

    Thin convenience wrapper over ``cls.derived_fields()``.
    """
    return cls.derived_fields()
