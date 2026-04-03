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
Class-reference registry utilities for struct and pytree serialisation.

This module provides a stable mechanism for converting runtime classes into
importable string references and resolving them back later.

Class references use the canonical format:

    ``"module.path:QualifiedClassName"``

Examples:
- ``"neuralqx.graph.core.graph_handler:GraphHandler"``
- ``"my_pkg.subpkg.models:Outer.Inner"``

Struct and pytree serialisation needs a process-independent way to identify
types. Raw class objects are not portable across sessions, but importable class
references are.

The registry combines:
    - a fast in-memory cache for already-seen classes,
    - a lazy importer fallback for classes not yet loaded in the process.
"""

from __future__ import annotations

from typing import Any

from ._internals.class_registry import import_class_ref
from ._internals.class_registry import lookup_class
from ._internals.class_registry import register_class

__all__ = [
    "class_ref",
    "register_class_ref",
    "resolve_class",
]


def class_ref(cls: type[Any]) -> str:
    """
    Return canonical class reference string for ``cls``.

    Args:
        cls: Runtime class object to encode.

    Returns:
        A deterministic ``"module:qualname"`` reference suitable for manifests.

    Notes:
        The returned value is purely structural and does not validate importability
        at call time. Import validation occurs during :func:`resolve_class`.
    """
    return f"{cls.__module__}:{cls.__qualname__}"


def register_class_ref(cls: type[Any]) -> None:
    """
    Register ``cls`` in the process-local class cache.

    This improves resolution speed and avoids repeated import traversal for
    classes already encountered during struct/pytree processing.
    """
    register_class(class_ref(cls), cls)


def resolve_class(ref: str) -> type[Any]:
    """
    Resolve a class reference into a runtime class object.

    This functions resolves by:
        1. Look up ``ref`` in the in-memory registry cache.
        2. If missing, import module + qualified name lazily.
        3. Cache resolved result for future calls.

    Args:
        ref: Class reference in ``"module:qualname"`` format.

    Returns:
        Imported and validated class object.

    Raises:
        SerializationError: If the reference format is invalid, import fails, attribute traversal
            fails, or the resolved object is not a class.
    """
    cls = lookup_class(ref)
    if cls is not None:
        return cls

    cls = import_class_ref(ref)
    register_class(ref, cls)
    return cls
