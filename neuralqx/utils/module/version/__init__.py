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
Versioning subsystem for neuraLQX.
"""

from packaging.version import InvalidVersion

from .coercion import ModuleReference
from .coercion import VersionInput
from .coercion import coerce_version_text

from .module_version import get_module_neuralqx_version
from .module_version import get_module_version
from .module_version import get_module_version_string

from .neuralqx_version import NeuralqxVersion

from .version import Version
from .version import coerce_version
from .version import normalize_version
from .version import parse_version
from .version import try_parse_version

__all__ = [
    "ModuleReference",
    "VersionInput",
    "InvalidVersion",
    "Version",
    "NeuralqxVersion",
    "coerce_version_text",
    "coerce_version",
    "parse_version",
    "try_parse_version",
    "normalize_version",
    "get_module_version",
    "get_module_neuralqx_version",
    "get_module_version_string",
]
