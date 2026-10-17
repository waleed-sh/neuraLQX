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


"""Ordered callback execution pipeline."""

from __future__ import annotations

from typing import Any

from .adapters import as_callback
from .adapters import as_tuple
from .base import Callback
from .context import CallbackContext
from .control import CallbackControl
from .control import as_control
from .exceptions import StopRun


class CallbackPipeline(Callback):
    """Ordered callback collection with explicit event dispatch."""

    def __init__(self, callbacks: Any = None, *, extra: Any = None) -> None:
        callback_items = as_tuple(callbacks) + as_tuple(extra)
        self.callbacks = tuple(
            sorted(
                (as_callback(callback) for callback in callback_items),
                key=lambda callback: int(getattr(callback, "priority", 0)),
            )
        )

    def __bool__(self) -> bool:
        return bool(self.callbacks)

    @property
    def requires_log_data(self) -> bool:  # type: ignore[override]
        return any(
            bool(getattr(callback, "requires_log_data", False))
            or bool(getattr(callback, "stores_history", False))
            for callback in self.callbacks
        )

    @property
    def stores_history(self) -> bool:  # type: ignore[override]
        return any(
            bool(getattr(callback, "stores_history", False))
            for callback in self.callbacks
        )

    def run_started(self, context: CallbackContext) -> None:
        self._emit_or_stop("run_started", context)

    def step_started(self, context: CallbackContext) -> None:
        self._emit_or_stop("step_started", context)

    def update_started(self, context: CallbackContext) -> None:
        self._emit_or_stop("update_started", context)

    def update_computed(self, context: CallbackContext) -> bool:  # type: ignore[override]
        control = self._emit("update_computed", context)
        self._raise_if_stopped(control)
        return bool(control.skip_update)

    def parameters_updating(self, context: CallbackContext) -> None:
        self._emit_or_stop("parameters_updating", context)

    def step_finished(self, context: CallbackContext) -> None:
        self._emit_or_stop("step_finished", context)

    def run_finished(self, context: CallbackContext) -> None:
        self._emit("run_finished", context)

    def run_failed(self, context: CallbackContext) -> None:
        self._emit("run_failed", context)

    def _emit_or_stop(self, event: str, context: CallbackContext) -> CallbackControl:
        control = self._emit(event, context)
        self._raise_if_stopped(control)
        return control

    def _emit(self, event: str, context: CallbackContext) -> CallbackControl:
        control = CallbackControl.continue_()
        for callback in self.callbacks:
            hook = getattr(callback, event)
            result = as_control(hook(context))
            control = control.merge(result)
            if control.stop:
                break
        return control

    @staticmethod
    def _raise_if_stopped(control: CallbackControl) -> None:
        if control.stop:
            raise StopRun(control.reason)


CallbackList = CallbackPipeline


__all__ = ["CallbackList", "CallbackPipeline"]
