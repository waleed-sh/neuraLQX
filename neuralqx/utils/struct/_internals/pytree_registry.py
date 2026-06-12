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
Process-local storage and JAX registration for pytree type adapters.

This module owns the two in-process caches that map registered types to their
``PyTreeTypeSpec`` adapters:

    ``PYTREE_SPECS_BY_CLASS``
        Keyed by the runtime class object. Populated immediately on registration.

    ``PYTREE_SPECS_BY_REF``
        Keyed by the ``"module:qualname"`` class reference string. Populated on
        first resolution by reference and used as a fast-path for subsequent
        lookups without repeating an import traversal.

Both caches are module-level dicts. In CPython the GIL makes individual dict
operations effectively atomic, so the common single-threaded class-definition
path is safe without explicit locking.
"""

from typing import Any

import jax

from ._types import FlattenFn
from ._types import UnflattenFn
from ._types import FlattenWithKeysFn
from ._types import PYTREE_SPECS_BY_CLASS
from ._types import PYTREE_SPECS_BY_REF


def register_jax_type(
    cls: type[Any],
    *,
    flatten: FlattenFn,
    unflatten: UnflattenFn,
    flatten_with_keys: FlattenWithKeysFn | None = None,
) -> None:
    """Register a type with JAX pytree utilities, ignoring duplicate registration."""
    try:
        if flatten_with_keys is not None and hasattr(
            jax.tree_util, "register_pytree_with_keys"
        ):
            jax.tree_util.register_pytree_with_keys(
                cls,
                flatten_with_keys,
                unflatten,
                flatten,
            )
        else:
            jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    except ValueError:
        # JAX errors on duplicate registration in the same process.
        # Local registry entries still update to the most recent spec.
        pass


def store_spec(ref: str, cls: type[Any], spec: Any) -> None:
    """Store pytree spec by both class object and class reference string."""
    PYTREE_SPECS_BY_CLASS[cls] = spec
    PYTREE_SPECS_BY_REF[ref] = spec


def get_spec_by_class(cls: type[Any]) -> Any | None:
    """Return registered pytree spec for a class if available."""
    return PYTREE_SPECS_BY_CLASS.get(cls)


def get_spec_by_ref(ref: str) -> Any | None:
    """Return registered pytree spec for a class reference if available."""
    return PYTREE_SPECS_BY_REF.get(ref)


def cache_ref(ref: str, spec: Any) -> None:
    """Cache ref->spec mapping when resolving lazily by imported class."""
    PYTREE_SPECS_BY_REF[ref] = spec
