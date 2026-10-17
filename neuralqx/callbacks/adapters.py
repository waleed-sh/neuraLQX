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


"""Callback normalization and compatibility adapters."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

from neuralqx.utils.typing import ControlLike
from neuralqx.utils.typing import LegacyCallback

from .base import AbstractCallback
from .base import Callback
from .context import CallbackContext
from .control import CallbackControl


class LegacyCallbackWrapper(AbstractCallback):
    """Adapt ``callback(step, log_data, driver)`` to the callback interface."""

    def __init__(self, callback: LegacyCallback) -> None:
        self.callback = callback

    def on_step_end(
        self, step: int, log_data: dict[str, Any], driver: Any
    ) -> ControlLike:
        result = self.callback(step, log_data, driver)
        if result is False:
            return CallbackControl.stop_run()
        return None


class FunctionCallback(Callback):
    """Adapter for context-style callback functions."""

    requires_log_data = False
    stores_history = False

    def __init__(self, callback: Callable[[CallbackContext], ControlLike]) -> None:
        self.callback = callback

    def step_finished(self, context: CallbackContext) -> ControlLike:
        return self.callback(context)


def as_callback(callback: Any) -> Callback:
    """Normalize one callback-like object."""
    if isinstance(callback, Callback):
        return callback
    if callable(callback):
        return LegacyCallbackWrapper(callback)
    raise TypeError(f"Invalid callback object {callback!r}.")


def as_tuple(callbacks: Any) -> tuple[Any, ...]:
    """Normalize callback containers without treating strings as containers."""
    if callbacks is None:
        return ()
    if isinstance(callbacks, Callback) or callable(callbacks):
        return (callbacks,)
    if isinstance(callbacks, Iterable) and not isinstance(callbacks, (str, bytes)):
        return tuple(callbacks)
    raise TypeError(f"Invalid callbacks object {callbacks!r}.")


__all__ = [
    "FunctionCallback",
    "LegacyCallback",
    "LegacyCallbackWrapper",
    "as_callback",
    "as_tuple",
]
