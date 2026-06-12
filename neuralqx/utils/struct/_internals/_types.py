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

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import jax
from neuralqx.utils.typing import Array
from neuralqx.utils.typing import DeserializerFn
from neuralqx.utils.typing import FlattenFn
from neuralqx.utils.typing import FlattenWithKeysFn
from neuralqx.utils.typing import SerializerFn
from neuralqx.utils.typing import UnflattenFn
from neuralqx.utils.typing import ValidatorLike

if TYPE_CHECKING:
    from ..pytree import PyTreeTypeSpec

STRUCT_REGISTRY: dict[str, type[Any]] = {}

PYTREE_SPECS_BY_CLASS: dict[type[Any], PyTreeTypeSpec] = {}

PYTREE_SPECS_BY_REF: dict[str, PyTreeTypeSpec] = {}

_TRACER_TYPE = getattr(getattr(jax, "core", None), "Tracer", ())

_TRACING_ERROR_TYPES = tuple(
    err
    for err in (
        getattr(getattr(jax, "errors", None), "TracerArrayConversionError", None),
        getattr(getattr(jax, "errors", None), "TracerBoolConversionError", None),
        getattr(getattr(jax, "errors", None), "TracerIntegerConversionError", None),
        getattr(getattr(jax, "errors", None), "ConcretizationTypeError", None),
        getattr(getattr(jax, "errors", None), "NonConcreteBooleanIndexError", None),
        getattr(getattr(jax, "errors", None), "UnexpectedTracerError", None),
    )
    if err is not None
)
