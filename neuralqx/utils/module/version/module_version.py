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
Helpers for retrieving version metadata from Python modules.
"""

from __future__ import annotations

from importlib import import_module
from packaging.version import InvalidVersion

from .coercion import ModuleReference
from .neuralqx_version import NeuralqxVersion

from .version import Version
from .version import parse_version


def _resolve_module(module: ModuleReference):
    return import_module(module) if isinstance(module, str) else module


def get_module_version_string(module: ModuleReference) -> str:
    """
    Return the module's raw ``__version__`` string or ``"unknown"``.
    """
    resolved = _resolve_module(module)
    raw = getattr(resolved, "__version__", "unknown")
    if raw is None:
        return "unknown"

    text = str(raw).strip()
    return text if text else "unknown"


def get_module_version(
    module: ModuleReference, *, strict: bool = True
) -> Version | None:
    """
    Return the module version as :class:`Version`.

    Missing ``__version__`` is treated as ``0.0.0``.
    If ``strict=False``, invalid versions return ``None``.
    """
    raw = get_module_version_string(module)
    if raw == "unknown":
        return Version("0.0.0")

    if strict:
        return parse_version(raw)

    try:
        return parse_version(raw)
    except (TypeError, ValueError, InvalidVersion):
        return None


def get_module_neuralqx_version(
    module: ModuleReference,
    *,
    strict: bool = True,
) -> NeuralqxVersion | None:
    """
    Return the module version as :class:`NeuralqxVersion`.

    Missing ``__version__`` is treated as ``0.0.0``.
    If ``strict=False``, invalid versions return ``None``.
    """
    raw = get_module_version_string(module)
    if raw == "unknown":
        return NeuralqxVersion("0.0.0")

    if strict:
        return NeuralqxVersion(raw)

    try:
        return NeuralqxVersion(raw)
    except (TypeError, ValueError, InvalidVersion):
        return None
