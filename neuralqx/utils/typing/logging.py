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


"""Logging type aliases and protocols."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import TypeAlias

LoggerCallable: TypeAlias = Callable[..., None]
"""Callable shape accepted by logger adapters."""

HistoryNode: TypeAlias = dict[str, Any]
"""Nested mutable history tree used by runtime loggers."""


class Logger(Protocol):
    """Protocol implemented by objects that consume driver log records."""

    def __call__(
        self,
        step: int,
        item: Mapping[str, Any],
        variational_state: Any | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        """Record one driver log item."""

    def flush(
        self,
        variational_state: Any | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        """Flush buffered state to its backing storage."""


__all__ = [
    "HistoryNode",
    "Logger",
    "LoggerCallable",
]
