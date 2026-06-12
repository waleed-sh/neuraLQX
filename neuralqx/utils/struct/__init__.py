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
Top-level public API for struct and pytree infrastructure.

This is the user-facing namespace for the struct system. It
consolidates all stable, high-level entry points required to:

    1. Define immutable, JAX-native data containers via :class:`Struct`.
    2. Declare rich field behaviour via :func:`field` and :class:`FieldSpec`.
    3. Register non-Struct Python classes as pytrees.
    4. Serialize and restore Struct/registered-pytree object graphs.
    5. Resolve classes and adapters by stable runtime references.

Basic workflow:

.. code-block::

    import neuralqx as nqx
    s = nqx.utils.struct

    class State(s.Struct):
        x: object
        tag: str = s.field(static=True, default="default")

    obj = State(x=[1, 2, 3])
    payload = obj.to_state_dict()
    restored = State.from_state_dict(payload)

Advanced workflow (non-Struct class registration):

.. code-block::

    class MyNode:
        def __init__(self, data, label):
            self.data = data
            self.label = label

    s.register_attrs_type(
        MyNode,
        node_fields=("data",),
        static_fields=("label",),
    )

    # MyNode now participates in pytree traversal and struct I/O.
"""

from .base import Struct
from .base import StructMeta
from .base import StructABCMeta
from .base import dataclass
from .base import derived_fields
from .base import fields
from .base import node_fields
from .base import opaque_fields
from .base import register_class
from .base import static_fields

from .fields import FieldKind
from .fields import FieldSpec
from .fields import field

from .pytree import PyTreeTypeSpec
from .pytree import is_registered_pytree_type
from .pytree import register_attrs_type
from .pytree import register_pytree_type
from .pytree import resolve_pytree_spec

from .registry import class_ref, resolve_class

from . import io

__all__ = [
    "io",
    "Struct",
    "StructMeta",
    "StructABCMeta",
    "dataclass",
    "field",
    "FieldKind",
    "FieldSpec",
    "register_class",
    "fields",
    "node_fields",
    "static_fields",
    "opaque_fields",
    "derived_fields",
    "class_ref",
    "resolve_class",
    "PyTreeTypeSpec",
    "register_pytree_type",
    "register_attrs_type",
    "is_registered_pytree_type",
    "resolve_pytree_spec",
]
