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
from neuralqx.profile._driver import driver_capture
from neuralqx.profile._instrumentation import instrument
from neuralqx.profile._instrumentation import region
from neuralqx.profile.session import ProfileResult
from neuralqx.utils.errors import NeuralqxValueError
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
        """Log records retained when history storage is enabled."""

        self.profile_result: ProfileResult | None = None
        """Artifacts from the most recent automatic run capture, or None if disabled.

        An explicitly supplied profiler retains ownership of its own result.
        Only rank zero receives a merged trace path in distributed runs.
        """

    def _profile_completion_values(self):
        """Return current asynchronous results without refreshing sampler caches."""
        states = self.state if isinstance(self.state, (tuple, list)) else (self.state,)
        return (
            tuple(
                (state.parameters, getattr(state, "_sampler_state", None))
                for state in states
            ),
            self._loss_stats,
            self._loss_grad,
            self._dp,
            getattr(self, "optimizer_state", None),
            getattr(self, "optimizer_states", None),
            getattr(self, "_last_energies", None),
            getattr(self, "_last_overlaps", None),
            getattr(getattr(self, "preconditioner", None), "info", None),
        )

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

    @instrument("vmc.reset", detail="developer", category="neuralqx.driver")
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
        """Yield step numbers while advancing the driver.

        An explicit Profiler context records these iterations and advances its
        schedule once per update. Automatic config.profile capture belongs to
        run, whose lifetime is bounded. This iterator never owns a collector
        across a yield or initiates distributed cleanup during finalization.
        """
        if n_steps < 0:
            raise NeuralqxValueError("n_steps must be non-negative.")
        if step_size <= 0:
            raise NeuralqxValueError("step_size must be positive.")
        with driver_capture(self, automatic=False) as capture:
            for _ in range(0, int(n_steps), int(step_size)):
                for inner in range(int(step_size)):
                    with capture.step(self.step_count):
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
        """Run the optimization loop and return in-memory log records.

        Set config.profile=True to capture the entire run automatically.
        config.profile_detail selects summary, detailed, or developer package
        annotations. Results are available in profile_result after completion.
        Existing explicit sessions are reused and their schedules advance once
        per attempted iteration, including rejected updates. Capture cleanup
        preserves application exceptions and includes callback and logger work.
        """
        if callbacks is not None and callback is not None:
            raise NeuralqxValueError("Pass either callbacks or callback, not both.")
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

        with driver_capture(self, extra_results=lambda: context.log_data) as capture:
            try:
                callback_pipeline.run_started(context)
                for _ in range(int(n_iter)):
                    with capture.step(self.step_count):
                        step = self.step_count
                        log_data: dict[str, Any] = {}
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
                            with region("logging.write", category="neuralqx.io"):
                                runtime_logger(
                                    step,
                                    log_data,
                                    self.state,
                                    driver=self,
                                )
                        if store_history:
                            with region(
                                "logging.history",
                                detail="developer",
                                category="neuralqx.io",
                            ):
                                self.history.append(dict(log_data))
            except StopRun:
                pass
            except BaseException as exc:
                context.error = exc
                callback_pipeline.run_failed(context)
                raise
            finally:
                if runtime_logger is not None:
                    with region("logging.flush", category="neuralqx.io"):
                        runtime_logger.flush(self.state, driver=self)
                callback_pipeline.run_finished(context)

            return self.history

    def estimate(self, observables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Estimate observables on the current state."""
        if observables is None:
            return {}
        estimates = {}
        with region("observables", category="neuralqx.driver"):
            for name, operator in observables.items():
                label = name[:160] if isinstance(name, str) else type(name).__name__
                with region(f"observable/{label}", category="neuralqx.observable"):
                    estimates[name] = self.state.expect(operator)
        return estimates

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
