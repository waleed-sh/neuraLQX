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


"""Logger normalization and fan-out helpers."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

from neuralqx.utils.typing import LoggerCallable

from .base import AbstractLogger


class CallableLogger(AbstractLogger):
    """Adapt a callable into the logger interface."""

    def __init__(
        self,
        fn: LoggerCallable,
        *,
        log_on_all_processes: bool = False,
    ) -> None:
        self.fn = fn
        self.log_on_all_processes = bool(log_on_all_processes)
        self._passes_driver = _accepts_driver_keyword(fn)
        self._n_positional = _accepted_positional_count(fn)

    def __call__(
        self,
        step: int,
        item: Mapping[str, Any],
        variational_state: Any | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        if not self.is_active_process:
            return
        args = (step, item, variational_state)[: self._n_positional]
        if self._passes_driver:
            self.fn(*args, driver=driver)
        else:
            self.fn(*args)


class LoggerList(AbstractLogger):
    """Fan out driver records to several loggers."""

    log_on_all_processes = True

    def __init__(self, loggers: Iterable[Any]) -> None:
        self.loggers = tuple(
            logger
            for logger in (as_logger(item) for item in loggers)
            if logger is not None
        )

    def __bool__(self) -> bool:
        return bool(self.loggers)

    def __call__(
        self,
        step: int,
        item: Mapping[str, Any],
        variational_state: Any | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        for logger in self.loggers:
            logger(step, item, variational_state, driver=driver)

    def flush(
        self,
        variational_state: Any | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        for logger in self.loggers:
            logger.flush(variational_state, driver=driver)


def as_logger(logger: Any) -> AbstractLogger | None:
    """Normalize user-facing logger inputs."""
    if logger is None or logger is False:
        return None
    if logger is True:
        from .runtime import RuntimeLogger

        return RuntimeLogger()
    if isinstance(logger, AbstractLogger):
        return logger
    if isinstance(logger, Iterable) and not isinstance(logger, (str, bytes, Mapping)):
        logger_list = LoggerList(logger)
        return logger_list if logger_list else None
    if callable(logger):
        return CallableLogger(logger)
    raise TypeError(f"Invalid logger object {logger!r}.")


def _accepts_driver_keyword(fn: LoggerCallable) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or name == "driver"
        for name, parameter in parameters.items()
    )


def _accepted_positional_count(fn: LoggerCallable) -> int:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return 3
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return 3
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return max(2, min(3, positional))


__all__ = ["CallableLogger", "LoggerCallable", "LoggerList", "as_logger"]
