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


"""Native variational-driver abstractions."""

from __future__ import annotations

import abc
import time
from typing import Any

from neuralqx.callbacks import CallbackContext
from neuralqx.callbacks import CallbackPipeline
from neuralqx.callbacks import ProgressBar
from neuralqx.callbacks import StopRun
from neuralqx.utils.io.logging import as_logger


class AbstractVariationalDriver(abc.ABC):
    """Small native base class for optimization drivers."""

    def __init__(self, variational_state: Any, *, loss_name: str = "loss") -> None:
        self._state = variational_state
        self._loss_name = str(loss_name)
        self._loss_stats: Any | None = None
        self._loss_grad: Any | None = None
        self._dp: Any | None = None
        self._logger: Any | None = None
        self._step_count = 0
        self.history: list[dict[str, Any]] = []

    @property
    def state(self) -> Any:
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def loss_name(self) -> str:
        return self._loss_name

    @property
    def loss_stats(self) -> Any:
        return self._loss_stats

    @property
    def logger(self) -> Any | None:
        """Logger used by the most recent :meth:`run` call."""
        return self._logger

    def reset_step(self) -> None:
        """Reset cached sampler data at the beginning of one optimization step."""
        self.state.reset()

    @abc.abstractmethod
    def compute_loss_and_update(self) -> tuple[Any, Any]:
        """Compute loss statistics and the parameter update direction."""

    @abc.abstractmethod
    def update_parameters(self, update: Any) -> None:
        """Apply one accepted parameter update."""

    def iter(self, n_steps: int, *, step_size: int = 1):
        """Yield step numbers while advancing the driver."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        for _ in range(0, int(n_steps), int(step_size)):
            for inner in range(int(step_size)):
                self.reset_step()
                self._loss_stats, self._dp = self.compute_loss_and_update()
                self.update_parameters(self._dp)
                if inner == 0:
                    yield self.step_count
                self._step_count += 1

    def run(
        self,
        n_iter: int,
        *,
        obs: dict[str, Any] | None = None,
        callbacks: Any = None,
        callback: Any = None,
        step_size: int = 1,
        show_progress: bool = False,
        progress_every: int = 1,
        progress_min_interval: float | None = 0.0,
        progress_leave: bool = True,
        store_history: bool | None = None,
        logger: Any = None,
    ) -> list[dict[str, Any]]:
        """Run the optimization loop and return in-memory log records."""
        if callbacks is not None and callback is not None:
            raise ValueError("Pass either callbacks or callback, not both.")
        runtime_logger = as_logger(logger)
        self._logger = runtime_logger
        default_callbacks = (
            (
                ProgressBar(
                    n_iter,
                    every=progress_every,
                    min_interval=progress_min_interval,
                    leave=progress_leave,
                ),
            )
            if show_progress
            else ()
        )
        callback_pipeline = CallbackPipeline(
            callbacks if callbacks is not None else callback,
            extra=default_callbacks,
        )
        has_callbacks = bool(callback_pipeline)
        has_logger = runtime_logger is not None
        has_observables = bool(obs)
        if store_history is None:
            store_history = self._default_store_history(
                has_callbacks=callback_pipeline.stores_history,
                has_observables=has_observables,
            )
        needs_log_data = (
            callback_pipeline.requires_log_data
            or has_observables
            or bool(store_history)
            or has_logger
        )
        context = CallbackContext(
            driver=self,
            step=self.step_count,
            log_data={},
        )

        try:
            callback_pipeline.run_started(context)
            for _ in range(int(n_iter)):
                step = self.step_count
                log_data: dict[str, Any] = {} if needs_log_data else {}
                context.reset_step(step=step, log_data=log_data)
                if has_callbacks:
                    callback_pipeline.step_started(context)
                    callback_pipeline.update_started(context)

                self.reset_step()
                self._loss_stats, self._dp = self.compute_loss_and_update()
                context.update = self._dp
                if needs_log_data:
                    log_data[self.loss_name] = self._loss_stats
                    self._log_observables(log_data, obs)
                    self._log_additional_data(log_data)

                reject = (
                    callback_pipeline.update_computed(context)
                    if has_callbacks
                    else False
                )
                if not reject:
                    if has_callbacks:
                        callback_pipeline.parameters_updating(context)
                    self.update_parameters(self._dp)
                    self._step_count += 1
                    context.accepted = True
                if has_callbacks:
                    callback_pipeline.step_finished(context)
                if runtime_logger is not None:
                    runtime_logger(
                        step,
                        log_data,
                        self.state,
                        driver=self,
                    )
                if store_history:
                    self.history.append(dict(log_data))
        except StopRun:
            pass
        except BaseException as exc:
            context.error = exc
            callback_pipeline.run_failed(context)
            raise
        finally:
            if runtime_logger is not None:
                runtime_logger.flush(self.state, driver=self)
            callback_pipeline.run_finished(context)

        return self.history

    def estimate(self, observables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Estimate observables on the current state."""
        if observables is None:
            return {}
        return {
            name: self.state.expect(operator) for name, operator in observables.items()
        }

    def _log_observables(
        self, log_data: dict[str, Any], observables: dict[str, Any] | None
    ) -> None:
        log_data.update(self.estimate(observables))

    def _log_additional_data(self, log_data: dict[str, Any]) -> None:
        log_data["wallclock"] = time.perf_counter()
        sampler_state = getattr(self.state, "sampler_state", None)
        acceptance = getattr(sampler_state, "acceptance", None)
        if acceptance is not None:
            log_data["acceptance"] = acceptance

    def _default_store_history(
        self, *, has_callbacks: bool, has_observables: bool
    ) -> bool:
        return has_callbacks or has_observables


__all__ = ["AbstractVariationalDriver"]
