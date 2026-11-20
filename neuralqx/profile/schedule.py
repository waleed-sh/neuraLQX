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


"""Step schedules that keep expensive collection out of ordinary iterations."""

from dataclasses import dataclass
from enum import StrEnum

from neuralqx.utils.errors import NeuralqxValueError


class Action(StrEnum):
    """Collector state for one complete application step.

    Attributes:
        WAIT: Collectors are stopped.
        WARMUP: Native collectors run, but their events are excluded from export.
        RECORD: Application events and native activity are retained.
    """

    WAIT = "wait"
    """Collectors are stopped and application annotations are inactive."""

    WARMUP = "warmup"
    """Native collection runs while events are excluded from the merged trace."""

    RECORD = "record"
    """Application events and native activity are retained."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """Repeat a wait, warmup, and recording cycle at explicit step boundaries.

    Step zero is the first iteration inside a Profiler context. Call
    Profiler.step after each iteration. Warmup initializes the native tracer
    before the measured steps. Application compilation should also be warmed
    up when compilation is not the subject of the measurement.

    Args:
        wait: Unprofiled steps at the beginning of each cycle.
        warmup: Steps with native collection enabled but discarded on export.
        active: Recorded steps per cycle. Must be positive.
        repeat: Number of cycles. Zero repeats until the session closes.
        skip_first: Unprofiled steps before the first cycle.

    Raises:
        NeuralqxValueError: A count is negative, nonintegral, or active is zero.
    """

    wait: int = 0
    """Unprofiled steps at the beginning of each capture cycle."""

    warmup: int = 1
    """Native collector warmup steps discarded during export."""

    active: int = 5
    """Positive number of recorded steps in each cycle."""

    repeat: int = 1
    """Number of cycles. Zero repeats until the session closes."""

    skip_first: int = 0
    """Unprofiled steps before the first cycle."""

    def __post_init__(self) -> None:
        for name in ("wait", "warmup", "active", "repeat", "skip_first"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "active" else 0):
                raise NeuralqxValueError(f"Invalid schedule count {name}={value!r}.")

    def action(self, step: int) -> Action:
        """Return the collector state for a zero-based step index.

        Args:
            step: Nonnegative index of the next application step.

        Returns:
            The action for that step, including WAIT after the final cycle.

        Raises:
            NeuralqxValueError: The step is negative or not an integer.
        """
        if type(step) is not int or step < 0:
            raise NeuralqxValueError("step must be a nonnegative integer.")
        relative = step - self.skip_first
        if relative < 0:
            return Action.WAIT
        length = self.wait + self.warmup + self.active
        cycle, position = divmod(relative, length)
        if self.repeat and cycle >= self.repeat:
            return Action.WAIT
        if position < self.wait:
            return Action.WAIT
        if position < self.wait + self.warmup:
            return Action.WARMUP
        return Action.RECORD

    def cycle(self, step: int) -> int:
        """Return the cycle index, or minus one during the initial skip.

        Args:
            step: Zero-based application step.

        Returns:
            Cycle index used to separate adjacent recording windows.
        """
        self.action(step)
        relative = step - self.skip_first
        return (
            -1 if relative < 0 else relative // (self.wait + self.warmup + self.active)
        )
