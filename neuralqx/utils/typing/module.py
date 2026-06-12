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


from collections.abc import Sequence

from types import ModuleType
from typing import TYPE_CHECKING
from typing import TypeAlias

from packaging.version import Version as PackagingVersion

if TYPE_CHECKING:
    from neuralqx.utils.module.version.neuralqx_version import NeuralqxVersion
    from neuralqx.utils.module.version.version import Version

ModuleReference: TypeAlias = str | ModuleType
"""Importable module name or imported module object."""

VersionInput: TypeAlias = (
    "Version | NeuralqxVersion | str | int | Sequence[object] | PackagingVersion"
)
"""Input accepted by neuraLQX version coercion helpers."""

__all__ = [
    "ModuleReference",
    "VersionInput",
]
