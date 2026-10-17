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


"""Base interfaces for driver loggers."""

from __future__ import annotations

import abc
from typing import Any
from typing import Mapping

import jax
from neuralqx.utils.typing import Logger


class AbstractLogger(abc.ABC):
    """Base class for neuraLQX driver loggers.

    Loggers are called on every JAX process. Expensive host-side retention,
    plotting, or file/network I/O should normally happen only on process 0.
    """

    log_on_all_processes: bool = False

    @property
    def is_primary_process(self) -> bool:
        """Whether this process should perform primary host-side logging."""
        return int(jax.process_index()) == 0

    @property
    def is_active_process(self) -> bool:
        """Whether this logger should retain data on the current process."""
        return bool(self.log_on_all_processes or self.is_primary_process)

    @abc.abstractmethod
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
        """Flush buffered data."""
        del variational_state, driver

    def close(self) -> None:
        """Release resources held by this logger."""
        self.flush()


__all__ = ["AbstractLogger", "Logger"]
