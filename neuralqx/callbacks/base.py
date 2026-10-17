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


"""Base callback APIs."""

from __future__ import annotations

from typing import Any

from neuralqx.utils.typing import ControlLike

from .context import CallbackContext
from .control import CallbackControl


class Callback:
    """Context-based callback interface for driver lifecycle events.

    Subclasses override whichever methods they need. Hooks may return
    :class:`CallbackControl` to request a graceful stop or to skip the parameter
    update of the current step.
    """

    priority: int = 0
    requires_log_data: bool = False
    stores_history: bool = False

    def run_started(self, context: CallbackContext) -> ControlLike:
        return None

    def step_started(self, context: CallbackContext) -> ControlLike:
        return None

    def update_started(self, context: CallbackContext) -> ControlLike:
        return None

    def update_computed(self, context: CallbackContext) -> ControlLike:
        return None

    def parameters_updating(self, context: CallbackContext) -> ControlLike:
        return None

    def step_finished(self, context: CallbackContext) -> ControlLike:
        return None

    def run_finished(self, context: CallbackContext) -> ControlLike:
        return None

    def run_failed(self, context: CallbackContext) -> ControlLike:
        return None


class AbstractCallback(Callback):
    """Compatibility base class for the original step/log-data hook API."""

    requires_log_data = True
    stores_history = True

    @property
    def callback_order(self) -> int:
        return 0

    @property
    def priority(self) -> int:  # type: ignore[override]
        return int(self.callback_order)

    def on_run_start(self, step: int, driver: Any) -> None:
        pass

    def on_step_start(self, step: int, log_data: dict[str, Any], driver: Any) -> None:
        pass

    def on_compute_update_start(
        self, step: int, log_data: dict[str, Any], driver: Any
    ) -> None:
        pass

    def on_compute_update_end(
        self, step: int, log_data: dict[str, Any], driver: Any
    ) -> bool:
        return False

    def before_parameter_update(
        self, step: int, log_data: dict[str, Any], driver: Any
    ) -> None:
        pass

    def on_step_end(self, step: int, log_data: dict[str, Any], driver: Any) -> None:
        pass

    def on_run_end(self, step: int, driver: Any) -> None:
        pass

    def on_run_error(self, step: int, error: BaseException, driver: Any) -> None:
        pass

    def run_started(self, context: CallbackContext) -> ControlLike:
        return self.on_run_start(context.step, context.driver)

    def step_started(self, context: CallbackContext) -> ControlLike:
        return self.on_step_start(
            context.step,
            context.log_data,  # type: ignore[arg-type]
            context.driver,
        )

    def update_started(self, context: CallbackContext) -> ControlLike:
        return self.on_compute_update_start(
            context.step,
            context.log_data,  # type: ignore[arg-type]
            context.driver,
        )

    def update_computed(self, context: CallbackContext) -> ControlLike:
        reject = self.on_compute_update_end(
            context.step,
            context.log_data,  # type: ignore[arg-type]
            context.driver,
        )
        if reject:
            return CallbackControl.skip_parameter_update()
        return None

    def parameters_updating(self, context: CallbackContext) -> ControlLike:
        return self.before_parameter_update(
            context.step,
            context.log_data,  # type: ignore[arg-type]
            context.driver,
        )

    def step_finished(self, context: CallbackContext) -> ControlLike:
        return self.on_step_end(
            context.step,
            context.log_data,  # type: ignore[arg-type]
            context.driver,
        )

    def run_finished(self, context: CallbackContext) -> ControlLike:
        return self.on_run_end(context.driver.step_count, context.driver)

    def run_failed(self, context: CallbackContext) -> ControlLike:
        assert context.error is not None
        return self.on_run_error(
            context.driver.step_count, context.error, context.driver
        )


__all__ = ["AbstractCallback", "Callback"]
