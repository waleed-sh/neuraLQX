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
Internal runtime machinery for Struct class processing and lifecycle management.

Responsibilities:
    - Field collection and MRO merging (``collect_fields``).
    - ``__init__`` generation with correct signature metadata (``build_init``).
    - Full instance lifecycle: assign -> derive -> post-init -> validate -> freeze
      (``finalize_instance``).
    - JAX pytree flatten/unflatten factory (``tree_flatten_factory``).
    - Deferred derived-field materialisation for tracing compatibility
      (``materialize_deferred_derived_field``).
    - Static-field validation helpers.
"""

import dataclasses
import inspect
import threading
import typing
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

import jax
from neuralqx.utils.errors import ValidationError

from ..fields import FieldKind
from ..fields import FieldSpec
from ._types import _TRACER_TYPE
from ._types import _TRACING_ERROR_TYPES
from .class_registry import register_class
from .field_helpers import FACTORY_DEFAULT
from .field_helpers import MISSING

INIT_FLAG = "__struct_initializing__"

# Thread-local storage for cycle detection during deferred field materialisation.
_materializing: threading.local = threading.local()


@dataclasses.dataclass(frozen=True, slots=True)
class OpaqueRef:
    """Identity-preserving wrapper used for opaque values in pytree aux data."""

    value: Any

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpaqueRef) and self.value is other.value


@dataclasses.dataclass(frozen=True, slots=True)
class DeferredDerivedValue:
    """Marker used when a derived field is intentionally deferred during tracing."""

    name: str


def _is_classvar(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    return annotation is typing.ClassVar or origin is typing.ClassVar


def _contains_tracer(value: Any) -> bool:
    if _TRACER_TYPE and isinstance(value, _TRACER_TYPE):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tracer(k) or _contains_tracer(v) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_tracer(v) for v in value)
    return False


def _is_tracing_error(exc: BaseException) -> bool:
    if not _TRACING_ERROR_TYPES:
        return False

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None:
        if isinstance(current, _TRACING_ERROR_TYPES):
            return True
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        next_exc = current.__cause__
        if next_exc is None:
            next_exc = current.__context__
        current = next_exc
    return False


def _contains_arraylike(value: Any) -> bool:
    if isinstance(value, (jax.Array, np.ndarray)):
        return True
    if isinstance(value, np.generic):
        return False
    if isinstance(value, Mapping):
        return any(
            _contains_arraylike(k) or _contains_arraylike(v) for k, v in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_arraylike(v) for v in value)
    return False


def ensure_valid_static_value(name: str, value: Any) -> None:
    """Validate that static fields are hashable and array-free."""
    if _contains_arraylike(value):
        raise ValidationError(
            f"Static field {name!r} cannot contain array-like values. "
            "Move arrays to pytree nodes or mark the field opaque."
        )
    try:
        hash(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValidationError(
            f"Static field {name!r} must be hashable for JAX caching."
        ) from exc


def safe_array_equal(a: Any, b: Any) -> bool:
    """An equality that behaves well for arrays and nested containers."""
    if isinstance(a, (jax.Array, np.ndarray)) or isinstance(b, (jax.Array, np.ndarray)):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if len(a) != len(b):
            return False
        for ka, va in a.items():
            if ka not in b:
                return False
            if not safe_array_equal(va, b[ka]):
                return False
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)) and len(a) == len(b):
        return all(safe_array_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def summarize_value(value: Any) -> str:
    """Compact representation for repr output."""
    if isinstance(value, jax.Array):
        shape = ",".join(str(d) for d in value.shape)
        return f"{value.dtype}[{shape}]"
    if isinstance(value, np.ndarray):
        shape = ",".join(str(d) for d in value.shape)
        return f"{value.dtype}[{shape}]"
    if isinstance(value, str):
        return repr(value)
    text = repr(value)
    return text if len(text) <= 88 else text[:85] + "..."


def call_configured_callable(
    fn: Callable[..., Any],
    *,
    obj: Any,
    value: Any = MISSING,
) -> Any:
    """Call converter/validator/derive hooks supporting ergonomic signatures."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        if value is MISSING:
            try:
                return fn(obj)
            except TypeError:
                return fn()
        try:
            return fn(obj, value)
        except TypeError:
            return fn(value)

    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    count = len(positional)

    if value is MISSING:
        if count == 0:
            return fn()
        if count == 1 or has_varargs:
            return fn(obj)
        raise TypeError(f"Callable {fn!r} must accept 0 or 1 positional argument(s).")

    if count == 1:
        return fn(value)
    if count == 2 or has_varargs:
        return fn(obj, value)
    raise TypeError(f"Callable {fn!r} must accept 1 or 2 positional argument(s).")


def apply_converter(spec: FieldSpec, obj: Any, value: Any) -> Any:
    """Apply field converter if present."""
    if spec.converter is None:
        return value
    return call_configured_callable(spec.converter, obj=obj, value=value)


def run_validators(spec: FieldSpec, obj: Any, value: Any) -> None:
    """Run validators for one field value."""
    for validator in spec.validators:
        result = call_configured_callable(validator, obj=obj, value=value)
        if result is False:
            raise ValidationError(f"Validation failed for field {spec.name!r}.")


def collect_fields(cls: type[Any]) -> OrderedDict[str, FieldSpec]:
    """Collect and merge field specs across the MRO."""
    result: OrderedDict[str, FieldSpec] = OrderedDict()
    for base in reversed(cls.__mro__[1:]):
        base_fields = getattr(base, "__struct_fields__", None)
        if base_fields:
            result.update(base_fields)
    annotations = cls.__dict__.get("__annotations__", {})
    for name, annotation in annotations.items():
        if _is_classvar(annotation):
            continue
        raw_default = cls.__dict__.get(name, MISSING)
        if isinstance(raw_default, FieldSpec):
            spec = dataclasses.replace(raw_default, name=name)
            try:
                delattr(cls, name)
            except AttributeError:
                pass
        elif raw_default is not MISSING:
            spec = FieldSpec(name=name, default=raw_default)
            try:
                delattr(cls, name)
            except AttributeError:
                pass
        else:
            spec = FieldSpec(name=name)
        result[name] = spec
    return result


def validate_field_order(
    cls_name: str, all_fields: OrderedDict[str, FieldSpec]
) -> None:
    """Validate constructor ordering for generated ``__init__``."""
    positional_seen_default = False
    for spec in all_fields.values():
        if not spec.init or spec.kw_only:
            continue
        if spec.has_default:
            positional_seen_default = True
        elif positional_seen_default:
            raise ValidationError(
                f"In {cls_name!r}, a required positional field follows a field with a default."
            )


def assign_initial_fields(
    cls: type[Any],
    obj: Any,
    *,
    bound: Mapping[str, Any],
    reject_derived_inputs: bool,
) -> None:
    """Assign initial values for all non-derived fields."""
    for spec in cls.__struct_fields__.values():
        if spec.is_derived:
            if reject_derived_inputs and spec.name in bound:
                raise TypeError(
                    f"Field {spec.name!r} is derived and cannot be supplied explicitly."
                )
            continue
        if spec.name in bound:
            value = bound[spec.name]
        elif spec.has_default:
            value = spec.make_default()
        elif spec.init:
            raise TypeError(f"Missing required field {spec.name!r}.")
        else:
            raise TypeError(
                f"Field {spec.name!r} has init=False and no default/default_factory."
            )
        object.__setattr__(obj, spec.name, apply_converter(spec, obj, value))


def derive_fields(
    cls: type[Any],
    obj: Any,
    *,
    defer_on_trace_error: bool = False,
) -> None:
    """Compute all derived fields, optionally deferring trace-incompatible ones."""
    for spec in cls.__struct_fields__.values():
        if not spec.is_derived:
            continue

        try:
            raw = call_configured_callable(
                typing.cast(Callable[..., Any], spec.derived), obj=obj
            )
            value = apply_converter(spec, obj, raw)
        except Exception as exc:
            if defer_on_trace_error and _is_tracing_error(exc):
                object.__setattr__(obj, spec.name, DeferredDerivedValue(spec.name))
                continue
            raise

        object.__setattr__(obj, spec.name, value)


def materialize_deferred_derived_field(cls: type[Any], obj: Any, name: str) -> Any:
    """Materialise one derived field if it was deferred during tracing."""
    spec = cls.__struct_fields__.get(name)
    if spec is None or not spec.is_derived:
        return object.__getattribute__(obj, name)

    current = object.__getattribute__(obj, name)
    if not isinstance(current, DeferredDerivedValue):
        return current

    # Cycle detection: guard against a derived callable that reads its own field,
    # directly or transitively, which would cause infinite recursion.
    key = (id(obj), name)
    active: set[tuple[int, str]] = _materializing.__dict__.setdefault("active", set())
    if key in active:
        raise RuntimeError(
            f"Cycle detected while materializing derived field {name!r} on "
            f"{cls.__name__}: the derivation callable reads its own output."
        )
    active.add(key)
    try:
        raw = call_configured_callable(
            typing.cast(Callable[..., Any], spec.derived), obj=obj
        )
        value = apply_converter(spec, obj, raw)
        object.__setattr__(obj, name, value)
        return value
    finally:
        active.discard(key)


def validate_instance(cls: type[Any], obj: Any) -> None:
    """Validate static constraints and field validators for an instance."""
    for spec in cls.__struct_fields__.values():
        value = getattr(obj, spec.name)
        if spec.is_derived and isinstance(value, DeferredDerivedValue):
            continue
        if spec.kind is FieldKind.STATIC:
            ensure_valid_static_value(spec.name, value)
        run_validators(spec, obj, value)


def finalize_instance(
    cls: type[Any],
    obj: Any,
    *,
    run_post_init: bool,
    validate: bool,
    defer_derived_on_trace_error: bool = False,
) -> None:
    """Finalise lifecycle: derive -> post-init -> derive -> validate -> freeze."""
    derive_fields(cls, obj, defer_on_trace_error=defer_derived_on_trace_error)
    if run_post_init:
        post_init = getattr(cls, "__post_init__", None)
        if post_init is not None:
            post_init(obj)
            derive_fields(cls, obj, defer_on_trace_error=defer_derived_on_trace_error)
    if validate:
        validate_instance(cls, obj)
    object.__setattr__(obj, INIT_FLAG, False)


def build_init(
    cls_name: str, all_fields: OrderedDict[str, FieldSpec]
) -> Callable[..., None]:
    """Generate an ``__init__`` compatible with field metadata."""
    init_fields = [spec for spec in all_fields.values() if spec.init]
    pos_specs = [spec for spec in init_fields if not spec.kw_only]
    kw_specs = [spec for spec in init_fields if spec.kw_only]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, INIT_FLAG, True)
        if len(args) > len(pos_specs):
            raise TypeError(
                f"{type(self).__name__}() takes at most {len(pos_specs)} positional argument(s) "
                f"but {len(args)} were given"
            )
        bound: dict[str, Any] = {}
        for spec, arg in zip(pos_specs, args, strict=False):
            if spec.name in kwargs:
                raise TypeError(
                    f"{type(self).__name__}() got multiple values for '{spec.name}'"
                )
            bound[spec.name] = arg
        bound.update(kwargs)
        unknown = set(bound) - {spec.name for spec in init_fields}
        if unknown:
            unexpected = ", ".join(repr(k) for k in sorted(unknown))
            raise TypeError(
                f"{type(self).__name__}() got unexpected keyword argument(s): {unexpected}"
            )
        assign_initial_fields(type(self), self, bound=bound, reject_derived_inputs=True)
        finalize_instance(type(self), self, run_post_init=True, validate=True)

    params: list[inspect.Parameter] = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    for spec in pos_specs:
        default = inspect.Parameter.empty
        if spec.has_default:
            default = spec.default if spec.default is not MISSING else FACTORY_DEFAULT
        params.append(
            inspect.Parameter(
                spec.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default
            )
        )
    for spec in kw_specs:
        default = inspect.Parameter.empty
        if spec.has_default:
            default = spec.default if spec.default is not MISSING else FACTORY_DEFAULT
        params.append(
            inspect.Parameter(
                spec.name, inspect.Parameter.KEYWORD_ONLY, default=default
            )
        )
    __init__.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls_name}.__init__"
    __init__.__doc__ = f"Auto-generated __init__ for {cls_name}."
    return __init__


def tree_flatten_factory(
    cls: type[Any],
) -> tuple[
    Callable[[Any], tuple[list[tuple[Any, Any]], Any]],
    Callable[[Any], tuple[list[Any], Any]],
    Callable[[Any, list[Any]], Any],
]:
    """Build pytree flatten/unflatten functions for a struct class."""
    specs = cls.__struct_fields__
    node_names = tuple(
        name for name, spec in specs.items() if spec.kind is FieldKind.NODE
    )
    static_names = tuple(
        name for name, spec in specs.items() if spec.kind is FieldKind.STATIC
    )
    opaque_names = tuple(
        name for name, spec in specs.items() if spec.kind is FieldKind.OPAQUE
    )

    def flatten_with_keys(obj: Any) -> tuple[list[tuple[Any, Any]], Any]:
        nodes = [
            (jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in node_names
        ]
        static_vals = []
        for name in static_names:
            value = getattr(obj, name)
            ensure_valid_static_value(name, value)
            static_vals.append(value)
        opaque_vals = [OpaqueRef(getattr(obj, name)) for name in opaque_names]
        aux = (
            node_names,
            static_names,
            opaque_names,
            tuple(static_vals),
            tuple(opaque_vals),
        )
        return nodes, aux

    def flatten(obj: Any) -> tuple[list[Any], Any]:
        nodes = [getattr(obj, name) for name in node_names]
        static_vals = []
        for name in static_names:
            value = getattr(obj, name)
            ensure_valid_static_value(name, value)
            static_vals.append(value)
        opaque_vals = [OpaqueRef(getattr(obj, name)) for name in opaque_names]
        aux = (
            node_names,
            static_names,
            opaque_names,
            tuple(static_vals),
            tuple(opaque_vals),
        )
        return nodes, aux

    def unflatten(aux: Any, nodes: list[Any]) -> Any:
        node_names_, static_names_, opaque_names_, static_vals, opaque_vals = aux
        values: dict[str, Any] = {}
        for name, value in zip(node_names_, nodes, strict=True):
            values[name] = value
        for name, value in zip(static_names_, static_vals, strict=True):
            values[name] = value
        for name, ref in zip(opaque_names_, opaque_vals, strict=True):
            values[name] = ref.value
        tracing = any(_contains_tracer(node) for node in nodes)
        return cls.__struct_unflatten__(
            values,
            _defer_derived_on_trace_error=tracing,
            _validate=not tracing,
        )

    return flatten_with_keys, flatten, unflatten


def install_pytree_registration(cls: type[Any]) -> None:
    """Register struct class with JAX pytree utilities."""
    flatten_with_keys, flatten, unflatten = tree_flatten_factory(cls)
    try:
        if hasattr(jax.tree_util, "register_pytree_with_keys"):
            jax.tree_util.register_pytree_with_keys(
                cls, flatten_with_keys, unflatten, flatten
            )
        else:  # pragma: no cover - older JAX fallback
            jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    except ValueError:
        pass


def process_struct_class(cls: type[Any]) -> None:
    """Process class definition into a fully registered immutable struct."""
    if cls.__dict__.get("__struct_processed__", False):
        return

    all_fields = collect_fields(cls)
    validate_field_order(cls.__qualname__, all_fields)

    cls.__struct_fields__ = MappingProxyType(all_fields)
    cls.__struct_node_fields__ = tuple(
        name for name, spec in all_fields.items() if spec.kind is FieldKind.NODE
    )
    cls.__struct_static_fields__ = tuple(
        name for name, spec in all_fields.items() if spec.kind is FieldKind.STATIC
    )
    cls.__struct_opaque_fields__ = tuple(
        name for name, spec in all_fields.items() if spec.kind is FieldKind.OPAQUE
    )
    cls.__struct_derived_fields__ = tuple(
        name for name, spec in all_fields.items() if spec.is_derived
    )
    cls.__match_args__ = tuple(
        name for name, spec in all_fields.items() if spec.init and not spec.kw_only
    )

    if "__init__" not in cls.__dict__:
        cls.__init__ = build_init(cls.__qualname__, all_fields)  # type: ignore[method-assign]

    install_pytree_registration(cls)
    register_class(f"{cls.__module__}:{cls.__qualname__}", cls)
    cls.__struct_processed__ = True
