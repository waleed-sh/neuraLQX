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
Process-local class registry and lazy class resolver.

Provides two operations:

    ``register_class(ref, cls)``
        Cache a ``"module:qualname"``: class mapping so that future lookups
        skip the import traversal entirely.

    ``import_class_ref(ref)``
        Resolve a ``"module:qualname"`` reference by importing the module and
        walking the qualified name. Used as the fallback when a class is not
        already cached.

The registry (``STRUCT_REGISTRY``) is a plain module-level dict. CPython's GIL
makes individual dict reads and writes effectively atomic, which is sufficient
for the typical single-threaded class-definition use case.
"""

import importlib
from typing import Any

from neuralqx.utils.errors import SerializationError

from ._types import STRUCT_REGISTRY


def import_class_ref(ref: str) -> type[Any]:
    """Resolve ``module:qualname`` into a runtime class object."""
    if ":" not in ref:
        raise SerializationError(f"Invalid class reference format: {ref!r}")

    module_name, qualname = ref.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment-dependent import errors
        raise SerializationError(
            f"Could not import module {module_name!r} for {ref!r}"
        ) from exc

    obj: Any = module
    try:
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except AttributeError as exc:
        raise SerializationError(
            f"Could not resolve class {qualname!r} in module {module_name!r}"
        ) from exc

    if not isinstance(obj, type):
        raise SerializationError(f"Resolved reference is not a class: {ref!r}")

    return obj


def lookup_class(ref: str) -> type[Any] | None:
    """Return class by reference if present in the in-memory registry."""
    return STRUCT_REGISTRY.get(ref)


def register_class(ref: str, cls: type[Any]) -> None:
    """Store a class reference mapping in the process-local registry."""
    STRUCT_REGISTRY[ref] = cls
