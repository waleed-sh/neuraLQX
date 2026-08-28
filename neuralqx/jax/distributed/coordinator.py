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


"""Coordinator and decorator APIs for distributed execution planning."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from neuralqx.utils.typing import DistributedPhase
from neuralqx.utils.typing import DistributedPlan
from neuralqx.utils.typing import DistributedWorkload
from neuralqx.utils.typing import Planner
from neuralqx.utils.typing import WorkloadBuilder


class DistributedCoordinator:
    """Registry-backed coordinator for phase-local distributed layouts.

    The coordinator is deliberately small and explicit. Each execution phase is
    keyed by a ``DistributedPhase`` enum and resolved by a registered planner.
    New downstream or user code can extend the system with the ``register``
    decorator without editing central dispatch code.
    """

    def __init__(self) -> None:
        self._planners: dict[DistributedPhase, Planner] = {}

    def register(
        self,
        phase: DistributedPhase,
        *,
        replace: bool = False,
    ) -> Callable[[Planner], Planner]:
        """Register ``phase`` planner and return the decorated function."""

        def decorator(fn: Planner) -> Planner:
            if not replace and phase in self._planners:
                raise ValueError(
                    f"A distributed planner is already registered for {phase.value!r}."
                )
            self._planners[phase] = fn
            return fn

        return decorator

    def plan(
        self,
        phase: DistributedPhase,
        workload: DistributedWorkload | None = None,
    ) -> DistributedPlan | None:
        """Return the registered plan for ``phase`` and ``workload``."""
        planner = self._planners.get(phase)
        if planner is None:
            return None
        return planner(self, workload or DistributedWorkload())

    def planner_for(self, phase: DistributedPhase) -> Planner | None:
        """Return the planner registered for ``phase``, if any."""
        return self._planners.get(phase)

    def phases(self) -> tuple[DistributedPhase, ...]:
        """Return phases with registered planners."""
        return tuple(self._planners)


DEFAULT_COORDINATOR = DistributedCoordinator()


def distributed_planner(
    phase: DistributedPhase,
    *,
    replace: bool = False,
    coordinator: DistributedCoordinator = DEFAULT_COORDINATOR,
) -> Callable[[Planner], Planner]:
    """Decorator registering a planner on the selected coordinator."""
    return coordinator.register(phase, replace=replace)


def plan_distributed_execution(
    phase: DistributedPhase,
    workload: DistributedWorkload | None = None,
    *,
    coordinator: DistributedCoordinator = DEFAULT_COORDINATOR,
) -> DistributedPlan | None:
    """Resolve the distributed layout for ``phase``."""
    return coordinator.plan(phase, workload)


def with_distributed_plan(
    phase: DistributedPhase,
    workload_builder: WorkloadBuilder,
    *,
    plan_kwarg: str = "distributed_plan",
    coordinator: DistributedCoordinator = DEFAULT_COORDINATOR,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Inject a coordinator plan into a decorated downstream function.

    This decorator is intended for extension points and application code. It
    keeps distributed planning at the call boundary while the wrapped function
    receives a concrete ``DistributedPlan`` through ``plan_kwarg``.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            workload = workload_builder(*args, **kwargs)
            kwargs[plan_kwarg] = coordinator.plan(phase, workload)
            return fn(*args, **kwargs)

        return wrapped

    return decorator


__all__ = [
    "DEFAULT_COORDINATOR",
    "DistributedCoordinator",
    "Planner",
    "WorkloadBuilder",
    "distributed_planner",
    "plan_distributed_execution",
    "with_distributed_plan",
]
