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
Version parsing and comparison utilities.
"""

from typing import Tuple, Union
from types import ModuleType
import importlib
import re

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def normalize_version(version: str) -> Tuple[int, int, int]:
    """
    Convert a version string into a numeric (major, minor, patch) tuple.

    Non-numeric suffixes (e.g. dev, rc, post) are ignored.
    """
    match = _VERSION_RE.search(version)
    if not match:
        return (0, 0, 0)

    parts = [int(p) if p is not None else 0 for p in match.groups()]
    return tuple(parts)  # type: ignore


def get_module_version(module: Union[str, ModuleType]) -> Tuple[int, int, int]:
    """
    Retrieve a normalised version tuple for a Python module.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)

    version_str = getattr(module, "__version__", "0.0.0")
    return normalize_version(version_str)


def get_module_version_string(module: Union[str, ModuleType]) -> str:
    """
    Retrieve the raw version string of a Python module.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)

    return getattr(module, "__version__", "unknown")
