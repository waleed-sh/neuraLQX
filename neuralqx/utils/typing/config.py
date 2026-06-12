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

from collections.abc import Callable

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import TypeVar

if TYPE_CHECKING:
    from neuralqx.config import ConfigMutation

ConfigValueT = TypeVar("ConfigValueT")
"""Generic value parameter used by configuration option declarations."""

MutationHook: TypeAlias = "Callable[[ConfigMutation], None]"
"""Callback invoked after a configuration value changes."""

Parser: TypeAlias = Callable[[Any], Any]
"""Callable that converts raw configuration input into a canonical value."""

Validator: TypeAlias = Callable[[Any], None]
"""Callable that validates a parsed configuration value."""

__all__ = [
    "ConfigValueT",
    "MutationHook",
    "Parser",
    "Validator",
]
