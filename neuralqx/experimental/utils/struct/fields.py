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
Public field declarations for :mod:`neuralqx.utils.struct`.

This module defines the declarative schema language for Struct classes.

At a high level:
    - :class:`FieldKind` determines how values participate in JAX pytree semantics.
    - :class:`FieldSpec` stores normalized, immutable metadata for a single field.
    - :func:`field` is the primary user API for declaring field behaviour.

Struct fields are split into three runtime categories:

    1. Node fields (``FieldKind.NODE``)
       These are dynamic pytree leaves. JAX traces/transforms these values.

    2. Static fields (``FieldKind.STATIC``)
       These become pytree auxiliary metadata, not leaves. They influence JIT cache
       identity, so values must be hashable and cannot contain array-like objects.

    3. Opaque fields (``FieldKind.OPAQUE``)
       These are intentionally excluded from pytree leaves and are treated as runtime
       side-data. They are useful for handles, caches, tokens, or process-local
       objects that should not be traced.

The ``derived=...`` hook defines computed fields that are recomputed after
construction/replacement/deserialization. Derived fields:
    - must use ``init=False``,
    - cannot define defaults,
    - cannot be node fields,
    - are never serialised (they are recomputed instead).
"""

from __future__ import annotations

import dataclasses
import typing

from collections.abc import Callable
from collections.abc import Mapping

from enum import Enum
from typing import Any

from ._internals.field_helpers import MISSING
from ._internals.field_helpers import MissingType
from ._internals.field_helpers import normalize_validators
from ._internals._types import ValidatorLike

__all__ = [
    "FieldKind",
    "FieldSpec",
    "ValidatorLike",
    "field",
]


class FieldKind(str, Enum):
    """Enum describing a field's role in Struct/pytree behaviour."""

    NODE = "node"
    """Field is a pytree leaf and participates in tracing/transforms."""

    STATIC = "static"
    """Field is pytree auxiliary metadata (hashable, non-array contract)."""

    OPAQUE = "opaque"
    """Field is excluded from pytree leaves and treated as runtime side-data."""


@dataclasses.dataclass(frozen=True, slots=True)
class FieldSpec:
    """
    Immutable metadata describing one struct field.

    ``FieldSpec`` is the normalised internal representation used by the struct
    class processor. In user code, these are typically created by :func:`field`.

    1. You write class annotations/defaults/``field(...)`` declarations.
    2. Struct metaclass collects declarations and builds ``FieldSpec`` objects.
    3. Runtime processing validates and uses specs to drive constructor, pytree,
       equality, repr, and serialisation behaviour.

    Notes:
        ``FieldSpec`` instances are normally created by :func:`field`.
        Advanced users may construct them directly when building custom decorators.
    """

    name: str

    default: Any = MISSING

    kind: FieldKind = FieldKind.NODE
    """Determines node/static/opaque behaviour."""

    factory: Callable[[], Any] | MissingType = MISSING

    init: bool = True
    """Controls whether a value can be provided in the constructor."""

    repr: bool = True

    compare: bool = True

    serialize: bool | None = None
    """Explicit serialization override. Defaults are inferred from field kind."""

    kw_only: bool = False

    doc: str | None = None

    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    converter: Callable[..., Any] | None = None
    """Assignment-time hooks for normalization and validation."""

    validators: tuple[Callable[..., Any], ...] = ()
    """Assignment-time hooks for normalization and validation."""

    derived: Callable[..., Any] | None = None
    """Optional derivation function for computed fields."""

    @property
    def has_default(self) -> bool:
        """
        Return ``True`` if default value or factory is configured.

        This property is used to decide whether constructor inputs may be omitted
        and whether missing serialised fields can be reconstructed.
        """
        return self.default is not MISSING or self.factory is not MISSING

    @property
    def is_derived(self) -> bool:
        """Return ``True`` if the field value is computed via ``derived`` hook."""
        return self.derived is not None

    def make_default(self) -> Any:
        """
        Return field default value, materialising factory when needed.

        Returns:
            Concrete default value.

        Raises:
            TypeError: if neither ``default`` nor ``factory`` is defined.
        """
        if self.factory is not MISSING:
            return typing.cast(Callable[[], Any], self.factory)()
        if self.default is not MISSING:
            return self.default
        raise TypeError(f"Field {self.name!r} requires a value.")

    @property
    def should_serialize(self) -> bool:
        """
        Return effective serialisation policy for this field.

        Rules:
            1. Derived fields are never serialised.
            2. Explicit ``serialize=...`` overrides inferred behaviour.
            3. Otherwise opaque fields default to non-serialised.
            4. Node/static fields default to serialised.
        """
        if self.is_derived:
            return False
        if self.serialize is not None:
            return self.serialize
        return self.kind is not FieldKind.OPAQUE


def field(
    *,
    static: bool = False,
    pytree: bool = True,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | MissingType = MISSING,
    init: bool = True,
    repr: bool = True,
    compare: bool = True,
    serialize: bool | None = None,
    kw_only: bool = False,
    doc: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    converter: Callable[..., Any] | None = None,
    validator: ValidatorLike = None,
    derived: Callable[..., Any] | None = None,
) -> FieldSpec:
    """
    Declare metadata for one :class:`Struct` field.

    This function mirrors the ergonomics of :func:`dataclasses.field` while adding
    pytree-aware semantics and runtime hooks (converter, validator, derived).

    Typical usage
    -------------

    .. code-block::

        class State(Struct):
            x: jax.Array
            tag: str = field(static=True, default="exp-A")
            token: object = field(pytree=False, default_factory=object)
            norm: float = field(
                static=True,
                init=False,
                derived=lambda self: float(np.asarray(self.x).sum()),
            )

    The shape intentionally resembles :func:`dataclasses.field`, but it adds
    JAX/Struct-specific controls:
        - ``static`` and ``pytree`` decide field role.
        - ``converter`` and ``validator`` add runtime hooks.
        - ``derived`` supports deterministic computed fields.


    Args:
        static: Mark this field as static pytree metadata. Static values are part of JAX
            cache keys, so they must be hashable and array-free.

        pytree: When ``False``, the field is opaque runtime data (not a pytree leaf).

        default: Concrete default value used when input is omitted.

        default_factory: Zero-argument callable used to lazily create defaults.

        init: Include field in generated constructor.

        repr: Include field in ``__repr__`` output.

        compare: Include field in ``__eq__`` checks.

        serialize: Override whether field participates in state dict/export I/O.

        kw_only: Make field keyword-only in generated constructor.

        doc: Optional per-field documentation string.

        metadata: Free-form immutable metadata mapping for tooling.

        converter: Optional converter called on assignment paths. Accepts either
            ``converter(value)`` or ``converter(self, value)``.

        validator: Optional validator (or sequence of validators). Validators accept either
            ``validator(value)`` or ``validator(self, value)``. Returning ``False``
            raises :class:`ValidationError`.

        derived: Optional callable for computed fields (must use ``init=False``). Accepts
            ``derived()`` or ``derived(self)`` and is recomputed after construction,
            replacement, and deserialisation.

    Returns:
        Frozen metadata object consumed by struct class processing.

    Raises:
        ValueError: If declaration invariants are violated, for example:
            - both ``default`` and ``default_factory`` are provided,
            - ``static=True`` and ``pytree=False`` are combined,
            - derived-field constraints are broken.

    Notes:
        Validators may either:
            - raise exceptions directly, or
            - return ``False`` to trigger a :class:`ValidationError`.

        Converters/validators support ergonomic signatures:
            - ``fn(value)``
            - ``fn(self, value)``
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both 'default' and 'default_factory'.")
    if not pytree and static:
        raise ValueError("A field cannot be both static and opaque.")
    kind = (
        FieldKind.OPAQUE
        if not pytree
        else FieldKind.STATIC if static else FieldKind.NODE
    )
    if derived is not None:
        if init:
            raise ValueError("Derived fields must set init=False.")
        if default is not MISSING or default_factory is not MISSING:
            raise ValueError(
                "Derived fields cannot define a default or default_factory."
            )
        if kind is FieldKind.NODE:
            raise ValueError(
                "Derived fields must be static or opaque, not pytree nodes."
            )
        if serialize:
            raise ValueError("Derived fields are recomputed and cannot be serialized.")
    return FieldSpec(
        name="",
        kind=kind,
        default=default,
        factory=default_factory,
        init=init,
        repr=repr,
        compare=compare,
        serialize=serialize,
        kw_only=kw_only,
        doc=doc,
        metadata=dict(metadata or {}),
        converter=converter,
        validators=normalize_validators(validator),
        derived=derived,
    )
