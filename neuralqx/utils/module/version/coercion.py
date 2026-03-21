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
Low-level coercion helpers used by version classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

from typing import TYPE_CHECKING
from typing import TypeAlias

from packaging.version import Version as PackagingVersion

if TYPE_CHECKING:
    from .neuralqx_version import NeuralqxVersion
    from .version import Version


ModuleReference = str | ModuleType
if TYPE_CHECKING:
    VersionInput: TypeAlias = (
        Version | NeuralqxVersion | str | int | Sequence[object] | PackagingVersion
    )
else:
    VersionInput: TypeAlias = object


def _coerce_sequence_to_text(parts: Sequence[object]) -> str:
    if len(parts) == 0:
        raise ValueError("Version sequences must contain at least one component.")

    tokens: list[str] = []
    for item in parts:
        if isinstance(item, bool):
            raise TypeError("Boolean values are not valid version components.")

        if isinstance(item, int):
            if item < 0:
                raise ValueError("Version components must be non-negative integers.")
            tokens.append(str(item))
            continue

        if isinstance(item, str):
            token = item.strip()
            if token == "":
                raise ValueError(
                    "Empty string components are not valid version components."
                )
            tokens.append(token)
            continue

        raise TypeError(
            f"Unsupported version component type {type(item)!r}. "
            "Expected int or str components."
        )

    return ".".join(tokens)


def coerce_version_text(value: object) -> str:
    """
    Normalise a version-like object into a textual representation.
    """
    from .neuralqx_version import NeuralqxVersion
    from .version import Version

    if isinstance(value, Version):
        return value.source

    if isinstance(value, NeuralqxVersion):
        return str(value)

    if isinstance(value, PackagingVersion):
        return str(value)

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            raise ValueError("Version strings must not be empty.")
        return text

    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid versions.")

    if isinstance(value, int):
        if value < 0:
            raise ValueError("Version integers must be non-negative.")
        return str(value)

    if isinstance(value, Sequence):
        return _coerce_sequence_to_text(value)

    raise TypeError(
        f"Unsupported version value {value!r} (type {type(value)!r}). "
        "Expected str, int, sequence, Version, or NeuralqxVersion."
    )
